from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from .config import ExperimentConfig, TrainingConfig
from .tape import TapeLayout, interpret_tape, quantize, validate_symbols


def evolution_phase(step: int, training: TrainingConfig) -> str:
    if step < 0:
        raise ValueError("step must be non-negative")
    if step <= training.free_steps:
        return "Free evolution"
    if step <= training.supervision_end:
        return "Supervision window"
    return "Beyond training window"


def _validate_rollout(rollout: torch.Tensor) -> None:
    if rollout.ndim != 4 or min(rollout.shape) < 1:
        raise ValueError("rollout must have shape (time, channels, height, width)")
    if not torch.is_floating_point(rollout) or not torch.isfinite(rollout).all():
        raise ValueError("rollout must contain finite floating-point states")


def rollout_rgb(
    rollout: torch.Tensor, channel_indices: tuple[int | None, int | None, int | None]
) -> torch.Tensor:
    _validate_rollout(rollout)
    colors = []
    for channel in channel_indices:
        if channel is None:
            colors.append(torch.zeros_like(rollout[:, :1]))
        elif not 0 <= channel < rollout.shape[1]:
            raise ValueError("RGB channel index does not exist in rollout")
        else:
            colors.append(rollout[:, channel : channel + 1])
    color = torch.cat(colors, dim=1)
    return (
        ((torch.tanh(color) + 1.0) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1)
    )


def _role_channels(config: ExperimentConfig) -> tuple[int, int, int | None]:
    computation = next(
        (
            channel
            for channel in range(config.model.channels)
            if channel not in {config.model.program_channel, config.model.io_channel}
        ),
        None,
    )
    return config.model.program_channel, config.model.io_channel, computation


def _raw_tapes(
    rollout: torch.Tensor, layout: TapeLayout, io_channel: int
) -> tuple[str, ...]:
    values = layout.extract_tape(rollout[:, io_channel])
    discrete = quantize(values).cpu().tolist()
    symbols = {-1: "0", 0: "B", 1: "1"}
    return tuple("".join(symbols[value] for value in row) for row in discrete)


def save_gif(
    rollout: torch.Tensor,
    path: str | Path,
    *,
    layout: TapeLayout,
    config: ExperimentConfig,
    input_symbols: str,
    target_symbols: str,
    output_mode: str = "single",
    duration_ms: int = 80,
    scale: int = 12,
) -> Path:
    if duration_ms < 1 or scale < 1:
        raise ValueError("duration_ms and scale must be positive")
    if output_mode not in {"single", "multiple"}:
        raise ValueError("output_mode must be 'single' or 'multiple'")
    validate_symbols(input_symbols)
    validate_symbols(target_symbols)
    if len(input_symbols) > layout.config.tape_slots:
        raise ValueError("input exceeds the visualized tape")
    if len(target_symbols) > layout.config.tape_slots:
        raise ValueError("target exceeds the visualized tape")
    config.validate()
    _validate_rollout(rollout)
    if tuple(rollout.shape[1:]) != (
        config.model.channels,
        layout.height,
        layout.width,
    ):
        raise ValueError("rollout dimensions do not match the layout and model")

    states = rollout.detach().cpu()
    roles = _role_channels(config)
    frames = rollout_rgb(states, roles).numpy()
    raw_tapes = _raw_tapes(states, layout, config.model.io_channel)
    images = [
        Image.fromarray(frame).resize(
            (frame.shape[1] * scale, frame.shape[0] * scale),
            Image.Resampling.NEAREST,
        )
        for frame in frames
    ]
    images = [
        _annotate_frame(
            image,
            step,
            len(images) - 1,
            raw_tapes[step],
            layout,
            config,
            input_symbols,
            target_symbols,
            output_mode,
            scale,
        )
        for step, image in enumerate(images)
    ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    return path


def _annotate_frame(
    frame,
    step: int,
    total_steps: int,
    raw_tape: str,
    layout: TapeLayout,
    config: ExperimentConfig,
    input_symbols: str,
    target_symbols: str,
    output_mode: str,
    scale: int,
):
    left, top, bottom = 18, 116, 42
    width = max(620, frame.width + 2 * left)
    image = Image.new("RGB", (width, frame.height + top + bottom), "#161a1f")
    image.paste(frame, (left, top))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    foreground = "#e7ebef"
    values = torch.tensor(
        [
            1.0 if symbol == "1" else -1.0 if symbol == "0" else 0.0
            for symbol in raw_tape
        ]
    )
    interpreted = interpret_tape(values, output_mode)
    decoded = " | ".join(interpreted.binary_strings) if interpreted.valid else "invalid"
    phase = evolution_phase(step, config.training)

    draw.text((left, 6), f"t = {step} / {total_steps}", fill=foreground, font=font)
    draw.text(
        (left, 24),
        f"Status: {phase}",
        fill="#ff8585" if phase == "Free evolution" else foreground,
        font=font,
    )
    draw.text((left, 42), f"Raw tape: {raw_tape}", fill=foreground, font=font)
    draw.text((left, 60), f"Decoded: {decoded}", fill=foreground, font=font)
    draw.text(
        (left, 78),
        f"Input: {input_symbols}    Target: {target_symbols}",
        fill=foreground,
        font=font,
    )

    for index, (row, column) in enumerate(layout.tape_coordinates):
        x, y = left + column * scale, top + row * scale
        draw.rectangle((x, y, x + scale - 1, y + scale - 1), outline="white")
        if scale >= 12:
            draw.text(
                (x + scale // 2, y - 2),
                str(index),
                fill=foreground,
                font=font,
                anchor="ms",
            )

    computation = _role_channels(config)[2]
    blue = "none" if computation is None else f"computation ch{computation}"
    legend = (
        f"R: program ch{config.model.program_channel}   "
        f"G: I/O ch{config.model.io_channel}   B: {blue}   |   white: logical tape"
    )
    draw.text((left, top + frame.height + 10), legend, fill=foreground, font=font)
    return image

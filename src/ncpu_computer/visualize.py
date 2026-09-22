from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from .config import ExperimentConfig, TrainingConfig
from .tape import TapeLayout, interpret_tape, quantize, validate_symbols


_NEGATIVE = (59, 76, 192)
_NEUTRAL = (245, 245, 245)
_POSITIVE = (180, 4, 38)


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


def rollout_rgb(rollout: torch.Tensor, output_channel: int) -> torch.Tensor:
    _validate_rollout(rollout)
    if not 0 <= output_channel < rollout.shape[1]:
        raise ValueError("output channel index does not exist in rollout")
    values = rollout[:, output_channel].clamp(-1.0, 1.0)
    neutral = values.new_tensor(_NEUTRAL)
    negative = values.new_tensor(_NEGATIVE)
    positive = values.new_tensor(_POSITIVE)
    endpoint = torch.where((values < 0).unsqueeze(-1), negative, positive)
    color = neutral + values.abs().unsqueeze(-1) * (endpoint - neutral)
    return color.round().to(torch.uint8)


def _value_color(value: float) -> tuple[int, int, int]:
    value = max(-1.0, min(1.0, value))
    endpoint = _NEGATIVE if value < 0 else _POSITIVE
    weight = abs(value)
    return tuple(
        round(center + weight * (extreme - center))
        for center, extreme in zip(_NEUTRAL, endpoint)
    )


def _raw_tapes(
    rollout: torch.Tensor, layout: TapeLayout, output_channel: int
) -> tuple[str, ...]:
    values = layout.extract_tape(rollout[:, output_channel])
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
    scale: int = 24,
) -> Path:
    if duration_ms < 1 or scale < 1:
        raise ValueError("duration_ms and scale must be positive")
    if output_mode not in {"single", "multiple"}:
        raise ValueError("output_mode must be 'single' or 'multiple'")
    validate_symbols(input_symbols, allow_empty=True)
    validate_symbols(target_symbols, allow_empty=True)
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
    frames = rollout_rgb(states, config.model.output_channel).numpy()
    raw_tapes = _raw_tapes(states, layout, config.model.output_channel)
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
    frame: Image.Image,
    step: int,
    total_steps: int,
    raw_tape: str,
    layout: TapeLayout,
    config: ExperimentConfig,
    input_symbols: str,
    target_symbols: str,
    output_mode: str,
    scale: int,
) -> Image.Image:
    left, top, bottom = 18, 116, 42
    width = max(500, frame.width + 2 * left)
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
    draw.text((left, 42), f"Raw output: {raw_tape}", fill=foreground, font=font)
    draw.text((left, 60), f"Decoded: {decoded}", fill=foreground, font=font)
    draw.text(
        (left, 78),
        f"Input: {input_symbols or '<empty>'}    "
        f"Target: {target_symbols or '<empty>'}",
        fill=foreground,
        font=font,
    )
    for index, (row, column) in enumerate(layout.tape_coordinates):
        x, y = left + column * scale, top + row * scale
        draw.rectangle(
            (x, y, x + scale - 1, y + scale - 1),
            outline="#ffcc33",
            width=max(1, scale // 12),
        )
        if scale >= 12:
            draw.text(
                (x + scale // 2, y - 2),
                str(index),
                fill="#ffcc33",
                font=font,
                anchor="ms",
            )
    _draw_scale(draw, left, top + frame.height + 8, font, foreground)
    return image


def _draw_scale(draw, left: int, top: int, font, foreground) -> None:
    width, height = 160, 8
    for offset in range(width):
        value = -1.0 + 2.0 * offset / (width - 1)
        draw.line(
            (left + offset, top, left + offset, top + height),
            fill=_value_color(value),
        )
    draw.rectangle((left, top, left + width - 1, top + height), outline="#8d949c")
    label_y = top + height + 3
    draw.text((left, label_y), "0 (-1)", fill=foreground, font=font)
    draw.text(
        (left + width // 2, label_y),
        "B (0)",
        fill=foreground,
        font=font,
        anchor="ma",
    )
    draw.text(
        (left + width, label_y),
        "1 (+1)",
        fill=foreground,
        font=font,
        anchor="ra",
    )

from dataclasses import replace

import pytest
import torch
from PIL import Image

from ncpu_computer.config import ExperimentConfig, GeometryConfig
from ncpu_computer.tape import TapeLayout
from ncpu_computer.visualize import evolution_phase, rollout_rgb, save_gif


def test_evolution_phase_follows_training_window():
    training = replace(ExperimentConfig().training, free_steps=1, supervision_steps=2)
    assert evolution_phase(0, training) == "Free evolution"
    assert evolution_phase(1, training) == "Free evolution"
    assert evolution_phase(2, training) == "Supervision window"
    assert evolution_phase(3, training) == "Supervision window"
    assert evolution_phase(4, training) == "Beyond training window"
    with pytest.raises(ValueError):
        evolution_phase(-1, training)


def test_rollout_rgb_uses_fixed_io_channel_scale():
    rollout = torch.zeros(1, 3, 1, 5)
    rollout[0, 0] = 10.0
    rollout[0, 1, 0] = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    rgb = rollout_rgb(rollout, 1)

    assert rgb.shape == (1, 1, 5, 3)
    assert rgb.dtype == torch.uint8
    assert torch.equal(rgb[0, 0, 0], rgb[0, 0, 1])
    assert torch.equal(rgb[0, 0, 3], rgb[0, 0, 4])
    assert rgb[0, 0, 1, 2] > rgb[0, 0, 1, 0]
    assert torch.equal(rgb[0, 0, 2], torch.tensor([245, 245, 245]))
    assert rgb[0, 0, 3, 0] > rgb[0, 0, 3, 2]

    with pytest.raises(ValueError, match="I/O channel"):
        rollout_rgb(rollout, 3)


def test_save_gif_writes_every_rollout_frame(tmp_path):
    geometry = GeometryConfig(
        tape_slots=3,
        stride=2,
        border_left=1,
        border_right=1,
        border_top=1,
        border_bottom=1,
    )
    config = replace(ExperimentConfig(), geometry=geometry)
    layout = TapeLayout(geometry)
    rollout = torch.zeros(3, config.model.channels, layout.height, layout.width)
    rollout[0, config.model.io_channel] = layout.render_tape(
        torch.tensor([[1.0, 0.0, -1.0]])
    )[0]
    rollout[1, config.model.io_channel] = layout.render_tape(
        torch.tensor([[1.0, 0.0, 0.0]])
    )[0]
    rollout[2, config.model.io_channel] = layout.render_tape(
        torch.tensor([[-1.0, 0.0, 0.0]])
    )[0]

    path = save_gif(
        rollout,
        tmp_path / "evolution.gif",
        layout=layout,
        config=config,
        input_symbols="1B0",
        target_symbols="0",
        duration_ms=20,
        scale=4,
    )

    assert path.is_file()
    with Image.open(path) as image:
        assert image.n_frames == 3

    with pytest.raises(ValueError, match="input exceeds"):
        save_gif(
            rollout,
            tmp_path / "invalid.gif",
            layout=layout,
            config=config,
            input_symbols="1111",
            target_symbols="0",
        )

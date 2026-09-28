import pytest
import torch

from ncpu_computer import (
    GeometryConfig,
    ModelConfig,
    NeuralCellularAutomaton,
    TapeLayout,
)
from ncpu_computer.model import Perception, perception_kernel


TASKS = ("copy", "reverse")
GEOMETRY = GeometryConfig()


def make_model(config=ModelConfig()):
    torch.manual_seed(0)
    return NeuralCellularAutomaton(config, GEOMETRY, TASKS)


def test_default_model_has_shared_io_and_zero_periodic_program():
    model = make_model()
    assert model.config.channels == 5
    assert model.config.io_channel == 1
    assert model.perception.kernel_count == 4
    assert model.programs.shape == (2, 1, 3, 2)
    assert model.program_parameters == ()
    assert torch.count_nonzero(model.programs) == 0


def test_program_repeats_from_absolute_zero_for_variable_tapes():
    config = ModelConfig(program_channels=2, program_mode="learned_read_only")
    model = make_model(config)
    with torch.no_grad():
        model.programs.copy_(
            torch.arange(model.programs.numel()).reshape_as(model.programs)
        )
    for slots in (3, 8):
        layout = TapeLayout(GEOMETRY, slots)
        grid = model.program_grid(torch.tensor([1]), layout.width)
        for x in range(0, layout.width, GEOMETRY.stride):
            assert torch.equal(grid[0, :, :, x : x + 2], model.programs[1])
        state = model.initial_state(
            layout.render_tape(torch.ones(1, slots)), torch.tensor([1])
        )
        assert torch.equal(state[0, :2], grid[0])
        assert torch.equal(
            layout.extract_tape(state[:, config.io_channel]), torch.ones(1, slots)
        )


@pytest.mark.parametrize(
    "mode,changes",
    [("zero", False), ("learned_read_only", False), ("learned_mutable", True)],
)
def test_program_mutability_modes(mode, changes):
    model = make_model(ModelConfig(program_mode=mode, max_abs_state=None))
    with torch.no_grad():
        model.rule.output.weight.fill_(0.1)
    layout = TapeLayout(GEOMETRY, 3)
    state = model.initial_state(layout.render_tape(torch.ones(1, 3)), torch.tensor([0]))
    updated = model.step(state)
    assert (not torch.equal(updated[:, :1], state[:, :1])) is changes
    assert not torch.equal(
        updated[:, model.config.io_channel], state[:, model.config.io_channel]
    )


def test_zero_delta_initialization_is_identity_at_every_shape():
    model = make_model()
    for slots in (2, 5):
        layout = TapeLayout(GEOMETRY, slots)
        state = model.initial_state(
            layout.render_tape(torch.randn(2, slots)), torch.tensor([0, 1])
        )
        assert torch.equal(
            model(state, 3), state.unsqueeze(1).expand(-1, 4, -1, -1, -1)
        )


def test_vertical_wrap_never_wraps_horizontally():
    config = ModelConfig(fixed_kernels=("sobel_y",), learnable_kernels=0, hidden_size=2)
    wrapped = Perception(config, wrap_y=True)
    zero = Perception(config, wrap_y=False)
    state = torch.zeros(1, config.channels, 3, 5)
    state[:, :, 0, 2] = 1.0
    assert torch.count_nonzero(wrapped(state)[:, :, 2, 2]) > 0
    assert torch.count_nonzero(zero(state)[:, :, 2, 2]) == 0
    state.zero_()
    state[:, :, 1, 0] = 1.0
    horizontal = Perception(
        ModelConfig(fixed_kernels=("sobel_x",), learnable_kernels=0), wrap_y=True
    )(state)
    assert torch.count_nonzero(horizontal[:, :, :, -1]) == 0


def test_noise_is_training_only_and_zero_avoids_randomness():
    model = make_model()
    with torch.no_grad():
        model.rule.output.weight.fill_(0.1)
    layout = TapeLayout(GEOMETRY, 3)
    state = model.initial_state(layout.render_tape(torch.ones(1, 3)), torch.tensor([0]))
    before = torch.get_rng_state()
    model.step(state, 0.0)
    assert torch.equal(before, torch.get_rng_state())
    model.eval()
    assert torch.equal(model.step(state, 0.2), model.step(state, 0.2))


def test_random_kernel_is_seeded_and_normalized():
    first = perception_kernel("random", 7)
    assert torch.equal(first, perception_kernel("random", 7))
    assert float(first.norm()) == pytest.approx(1.0)

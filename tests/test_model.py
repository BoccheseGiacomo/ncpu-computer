import pytest
import torch

from ncpu_computer.config import GeometryConfig, ModelConfig
from ncpu_computer.model import NeuralCellularAutomaton, perception_kernel
from ncpu_computer.tape import TapeLayout


TASKS = ("copy", "reverse")


def make_model(config=ModelConfig(), geometry=GeometryConfig()):
    torch.manual_seed(0)
    return NeuralCellularAutomaton(config, geometry, TASKS)


def test_default_model_has_expected_roles_and_parameter_count():
    model = make_model()
    assert model.parameter_count == 3279
    assert model.config.channels == 6
    assert model.config.input_channel == 1
    assert model.config.output_channel == 2
    assert model.perception.kernel_count == 4
    assert model.programs.shape == (2, 1, 7, 21)


def test_grid_program_selection_injects_input_and_zero_output():
    model = make_model()
    layout = model.layout
    input_grid = layout.render_tape(torch.randn(2, layout.config.tape_slots))
    state = model.initial_state(input_grid, torch.tensor([1, 0]))
    assert torch.equal(state[:, model.config.input_channel], input_grid)
    assert torch.count_nonzero(state[:, model.config.output_channel]) == 0
    assert torch.equal(state[0, 0], model.programs[1, 0])
    assert torch.equal(state[1, 0], model.programs[0, 0])
    assert not torch.equal(model.programs[0], model.programs[1])


def test_tape_program_is_zero_outside_logical_positions():
    geometry = GeometryConfig(tape_slots=4, stride=2, border_left=1, border_right=1)
    config = ModelConfig(program_channels=2, program_placement="tape")
    model = make_model(config, geometry)
    grid = model.program_grid(torch.tensor([0, 1]))
    layout = TapeLayout(geometry)
    assert grid.shape == (2, 2, layout.height, layout.width)
    assert torch.equal(layout.extract_tape(grid), model.programs)
    occupied = torch.zeros(layout.height, layout.width, dtype=torch.bool)
    occupied[layout.tape_row, layout.tape_slice] = True
    assert torch.count_nonzero(grid[:, :, ~occupied]) == 0


def test_zero_delta_initialization_produces_identity_dynamics():
    model = make_model()
    input_grid = model.layout.render_tape(torch.randn(2, 8))
    state = model.initial_state(input_grid, torch.tensor([0, 1]))
    rollout = model(state, 3)
    assert torch.equal(rollout, state.unsqueeze(1).expand_as(rollout))


@pytest.mark.parametrize("program_mutable", [False, True])
@pytest.mark.parametrize("input_mode", ["mutable", "frozen"])
def test_program_and_separate_input_mutability(program_mutable, input_mode):
    config = ModelConfig(
        program_mutable=program_mutable,
        input_mode=input_mode,
        max_abs_state=None,
    )
    model = make_model(config)
    with torch.no_grad():
        model.rule.output.weight.fill_(0.1)
    state = model.initial_state(
        model.layout.render_tape(torch.randn(2, 8)), torch.tensor([0, 1])
    )
    updated = model.step(state)
    if program_mutable:
        assert not torch.equal(updated[:, : config.program_channels], state[:, :1])
    else:
        assert torch.equal(updated[:, : config.program_channels], state[:, :1])
    if input_mode == "frozen":
        assert torch.equal(
            updated[:, config.input_channel], state[:, config.input_channel]
        )
    else:
        assert not torch.equal(
            updated[:, config.input_channel], state[:, config.input_channel]
        )
    assert not torch.equal(
        updated[:, config.output_channel], state[:, config.output_channel]
    )


def test_shared_io_starts_with_input_and_is_mutable():
    config = ModelConfig(io_mode="shared", input_mode="mutable", max_abs_state=None)
    model = make_model(config)
    input_grid = model.layout.render_tape(torch.randn(2, 8))
    state = model.initial_state(input_grid, torch.tensor([0, 1]))
    assert config.input_channel == config.output_channel
    assert torch.equal(state[:, config.output_channel], input_grid)
    with torch.no_grad():
        model.rule.output.weight.fill_(0.1)
    assert not torch.equal(model.step(state)[:, config.output_channel], input_grid)


@pytest.mark.parametrize("gate", ["none", "linear", "sigmoid", "tanh", "relu"])
@pytest.mark.parametrize("padding", ["zeros", "reflect", "replicate", "circular"])
def test_gates_and_padding_preserve_identity_initialization(gate, padding):
    config = ModelConfig(
        hidden_size=4,
        fixed_kernels=("identity",),
        learnable_kernels=0,
        gate=gate,
        padding=padding,
    )
    model = make_model(config)
    state = model.initial_state(
        model.layout.render_tape(torch.randn(2, 8)), torch.tensor([0, 1])
    )
    assert torch.equal(model.step(state), state)


def test_model_rejects_wrong_geometry_and_task_indices():
    model = make_model()
    with pytest.raises(ValueError, match="fixed model geometry"):
        model.initial_state(torch.zeros(1, 2, 2), torch.tensor([0]))
    input_grid = model.layout.render_tape(torch.zeros(1, 8))
    with pytest.raises(ValueError, match="out of range"):
        model.initial_state(input_grid, torch.tensor([2]))


def test_random_kernel_is_seeded_and_normalized():
    first = perception_kernel("random", 7)
    second = perception_kernel("random", 7)
    assert torch.equal(first, second)
    assert float(first.norm()) == pytest.approx(1.0)

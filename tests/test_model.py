import pytest
import torch

from ncpu_computer.config import ModelConfig
from ncpu_computer.model import NeuralCellularAutomaton, perception_kernel


def test_default_model_has_expected_roles_and_parameter_count():
    model = NeuralCellularAutomaton(ModelConfig())
    assert model.parameter_count == 2505
    assert model.config.channels == 5
    assert model.config.io_channel == 1
    assert model.perception.kernel_count == 4


def test_initial_state_injects_shared_io_and_leaves_other_channels_zero():
    model = NeuralCellularAutomaton(ModelConfig())
    input_grid = torch.randn(2, 4, 7)
    state = model.initial_state(input_grid)
    assert torch.equal(state[:, model.config.io_channel], input_grid)
    assert torch.count_nonzero(state[:, model.config.io_channel + 1 :]) == 0
    assert torch.count_nonzero(state[:, : model.config.program_channels]) == 0


def test_zero_delta_initialization_produces_identity_dynamics():
    model = NeuralCellularAutomaton(ModelConfig())
    state = torch.randn(2, model.config.channels, 4, 7)
    rollout = model(state, 3)
    assert torch.equal(rollout, state.unsqueeze(1).expand_as(rollout))


def test_program_is_frozen_and_io_is_mutable():
    config = ModelConfig(max_abs_state=None)
    model = NeuralCellularAutomaton(config)
    with torch.no_grad():
        model.rule.output.weight.fill_(0.1)
    state = torch.randn(2, config.channels, 4, 5)
    updated = model.step(state)
    assert torch.equal(
        updated[:, : config.program_channels], state[:, : config.program_channels]
    )
    assert not torch.equal(updated[:, config.io_channel], state[:, config.io_channel])


@pytest.mark.parametrize("gate", ["none", "linear", "sigmoid", "tanh", "relu"])
@pytest.mark.parametrize("padding", ["zeros", "reflect", "replicate"])
def test_gates_and_padding_preserve_identity_initialization(gate, padding):
    config = ModelConfig(
        hidden_size=4,
        fixed_kernels=("identity",),
        learnable_kernels=0,
        gate=gate,
        padding=padding,
    )
    model = NeuralCellularAutomaton(config)
    state = model.initial_state(torch.randn(2, 4, 5))
    assert torch.equal(model.step(state), state)


def test_random_kernel_is_seeded_and_normalized():
    first = perception_kernel("random", 7)
    second = perception_kernel("random", 7)
    assert torch.equal(first, second)
    assert float(first.norm()) == pytest.approx(1.0)


def test_vertical_wrap_does_not_wrap_horizontally():
    config = ModelConfig(
        fixed_kernels=("identity",),
        learnable_kernels=1,
        wrap_y=True,
        padding="zeros",
    )
    model = NeuralCellularAutomaton(config)
    with torch.no_grad():
        kernel = model.perception.learnable_0
        kernel.zero_()
        kernel[0, 1] = 1.0
    state = torch.zeros(1, config.channels, 3, 4)
    state[0, config.io_channel, 2, 0] = 1.0
    perceived = model.perception(state)[:, config.io_channel * 2 + 1]
    assert perceived[0, 0, 0] == 1.0
    assert perceived[0, 0, 3] == 0.0

    with torch.no_grad():
        kernel.zero_()
        kernel[1, 0] = 1.0
    state.zero_()
    state[0, config.io_channel, 0, 3] = 1.0
    assert model.perception(state)[0, config.io_channel * 2 + 1, 0, 0] == 0.0

    with torch.no_grad():
        kernel.zero_()
        kernel[0, 1] = 1.0
    state.zero_()
    state[0, config.io_channel, 2, 0] = 1.0

    no_wrap = NeuralCellularAutomaton(
        ModelConfig(
            fixed_kernels=("identity",),
            learnable_kernels=1,
            wrap_y=False,
            padding="zeros",
        )
    )
    with torch.no_grad():
        no_wrap.perception.learnable_0.copy_(kernel)
    assert no_wrap.perception(state)[0, config.io_channel * 2 + 1, 0, 0] == 0.0

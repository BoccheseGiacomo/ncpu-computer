import pytest
import torch

from ncpu_computer.config import ModelConfig
from ncpu_computer.model import NeuralCellularAutomaton, perception_kernel


def test_default_model_has_expected_roles_and_parameter_count():
    model = NeuralCellularAutomaton(ModelConfig())
    assert model.parameter_count == 2985
    assert model.config.channels == 6
    assert model.config.input_channel == 1
    assert model.config.output_channel == 2
    assert model.perception.kernel_count == 4


def test_initial_state_injects_input_and_leaves_output_zero():
    model = NeuralCellularAutomaton(ModelConfig())
    input_grid = torch.randn(2, 4, 7)
    state = model.initial_state(input_grid)
    assert torch.equal(state[:, model.config.input_channel], input_grid)
    assert torch.count_nonzero(state[:, model.config.output_channel]) == 0
    assert torch.count_nonzero(state[:, : model.config.program_channels]) == 0


def test_zero_delta_initialization_produces_identity_dynamics():
    model = NeuralCellularAutomaton(ModelConfig())
    state = torch.randn(2, model.config.channels, 4, 7)
    rollout = model(state, 3)
    assert torch.equal(rollout, state.unsqueeze(1).expand_as(rollout))


@pytest.mark.parametrize("input_mode", ["mutable", "frozen"])
def test_program_is_frozen_and_input_mutability_is_configurable(input_mode):
    config = ModelConfig(input_mode=input_mode, max_abs_state=None)
    model = NeuralCellularAutomaton(config)
    with torch.no_grad():
        model.rule.output.weight.fill_(0.1)
    state = torch.randn(2, config.channels, 4, 5)
    updated = model.step(state)
    assert torch.equal(
        updated[:, : config.program_channels], state[:, : config.program_channels]
    )
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
    model = NeuralCellularAutomaton(config)
    state = model.initial_state(torch.randn(2, 4, 5))
    assert torch.equal(model.step(state), state)


def test_random_kernel_is_seeded_and_normalized():
    first = perception_kernel("random", 7)
    second = perception_kernel("random", 7)
    assert torch.equal(first, second)
    assert float(first.norm()) == pytest.approx(1.0)

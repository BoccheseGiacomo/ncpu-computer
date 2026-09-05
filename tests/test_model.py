import pytest
import torch

from ncpu_computer.config import ModelConfig
from ncpu_computer.model import NeuralCellularAutomaton, perception_kernel


def test_default_model_has_expected_small_parameter_count():
    model = NeuralCellularAutomaton(ModelConfig())
    assert model.parameter_count == 2505
    assert model.perception.kernel_count == 4


def test_zero_delta_initialization_produces_identity_dynamics():
    model = NeuralCellularAutomaton(ModelConfig())
    state = torch.randn(2, 5, 4, 7)
    rollout = model(state, 3)
    assert torch.equal(rollout, state.unsqueeze(1).expand_as(rollout))


@pytest.mark.parametrize("gate", ["none", "linear", "sigmoid", "tanh", "relu"])
@pytest.mark.parametrize("padding", ["zeros", "reflect", "replicate", "circular"])
def test_configurable_gates_and_padding_preserve_identity_initialization(gate, padding):
    config = ModelConfig(
        hidden_size=4,
        fixed_kernels=("identity",),
        learnable_kernels=0,
        gate=gate,
        padding=padding,
    )
    model = NeuralCellularAutomaton(config)
    io_grid = torch.randn(2, 4, 5)
    initial = model.initial_state(io_grid)
    assert torch.equal(initial[:, config.io_channel], io_grid)
    assert torch.count_nonzero(initial[:, config.program_channel]) == 0
    assert torch.equal(model.step(initial), initial)


def test_frozen_program_is_exact_under_nonzero_clipped_updates():
    config = ModelConfig(max_abs_state=0.25)
    model = NeuralCellularAutomaton(config)
    with torch.no_grad():
        model.rule.output.weight.fill_(1.0)
    state = torch.randn(2, 5, 3, 4)
    original = state[:, config.program_channel].clone()
    updated = model.step(state)
    assert torch.equal(updated[:, config.program_channel], original)
    assert updated[:, config.io_channel].abs().max() <= 0.25


def test_random_kernel_is_seeded_and_normalized():
    first = perception_kernel("random", 7)
    second = perception_kernel("random", 7)
    assert torch.equal(first, second)
    assert float(first.norm()) == pytest.approx(1.0)

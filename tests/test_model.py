import math

import pytest
import torch

from ncpu_computer import (
    GeometryConfig,
    ModelConfig,
    NeuralCellularAutomaton,
    TapeLayout,
)
from ncpu_computer.model import LocalAttention, Perception, perception_kernel


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
    assert model.attention.heads == 1
    assert model.attention.attention_dim == 16
    assert model.programs.shape == (2, 1, 3, 2)
    assert model.program_parameters == ()
    assert torch.count_nonzero(model.programs) == 0


@pytest.mark.parametrize("program_start", [0, 1])
def test_program_repeats_from_configured_origin(program_start):
    geometry = GeometryConfig(program_start=program_start)
    config = ModelConfig(program_channels=2, program_mode="learned_read_only")
    model = NeuralCellularAutomaton(config, geometry, TASKS)
    with torch.no_grad():
        model.programs.copy_(
            torch.arange(model.programs.numel()).reshape_as(model.programs)
        )
    for slots in (3, 8):
        layout = TapeLayout(geometry, slots)
        grid = model.program_grid(torch.tensor([1]), layout.width)
        if program_start == 1:
            assert torch.count_nonzero(grid[..., 0]) == 0
            assert (layout.width - 1) % geometry.stride == 0
        for x in range(program_start, layout.width):
            phase = (x - program_start) % geometry.stride
            assert torch.equal(grid[0, :, :, x], model.programs[1, :, :, phase])
        if program_start == 0:
            assert layout.width % geometry.stride == 1
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
        model.rule.hidden.weight.zero_()
        model.rule.hidden.bias.fill_(1.0)
        model.rule.output.weight.fill_(0.1)
    layout = TapeLayout(GEOMETRY, 3)
    state = model.initial_state(layout.render_tape(torch.ones(1, 3)), torch.tensor([0]))
    updated = model.step(state)
    assert (not torch.equal(updated[:, :1], state[:, :1])) is changes
    if mode == "learned_mutable":
        assert torch.count_nonzero(state[:, :1, :, 0]) == 0
        assert torch.count_nonzero(updated[:, :1, :, 0]) > 0
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


def test_attention_masks_padding_and_preserves_raw_value_scale():
    config = ModelConfig(
        program_channels=1,
        computation_channels=0,
        attention_radius=1,
        attention_dim=1,
        attention_heads=1,
        attention_qk_cap=0.1,
    )
    attention = LocalAttention(config, wrap_y=False)
    with torch.no_grad():
        attention.qkv.weight.zero_()
        attention.qkv.weight[2, 0] = 1.0
    state = torch.zeros(1, config.channels, 3, 4)
    state[:, 0] = 100.0
    output, weights, mask = attention.attention(state)
    assert mask[0, 0].sum() == 4
    assert torch.count_nonzero(weights[0, 0, 0, 0, ~mask[0, 0]]) == 0
    assert float(weights[0, 0, 0, 0].sum()) == pytest.approx(1.0)
    assert float(output[0, 0, 0, 0]) == pytest.approx(100.0)


def test_wrapped_attention_radius_exceeding_height_uses_unique_cells():
    config = ModelConfig(
        attention_radius=4,
        attention_dim=12,
        attention_heads=3,
    )
    attention = LocalAttention(config, wrap_y=True)
    state = torch.randn(2, config.channels, 3, 11, requires_grad=True)
    output, weights, mask = attention.attention(state)
    assert output.shape == (2, 12, 3, 11)
    assert weights.shape == (2, 3, 3, 11, 27)
    assert mask[1, 5].sum() == 27
    indices, valid, _ = attention._metadata(3, 11, state.device)
    center_indices = indices[:, 1, 5][valid[:, 1, 5]]
    assert len(center_indices.unique()) == len(center_indices)
    output.square().mean().backward()
    assert torch.isfinite(state.grad).all()
    assert attention.qkv.weight.grad is not None


def test_attention_metadata_is_valid_for_small_heights_and_large_radii():
    device = torch.device("cpu")
    for wrap_y in (False, True):
        for height in (1, 2, 3, 5):
            for radius in range(7):
                config = ModelConfig(attention_radius=radius)
                attention = LocalAttention(config, wrap_y)
                indices, valid, _ = attention._metadata(height, 4, device)
                assert bool(((indices >= 0) & (indices < height * 4)).all())
                assert bool(valid.any(dim=0).all())
                if wrap_y:
                    for row in range(height):
                        for column in range(4):
                            selected = indices[:, row, column][valid[:, row, column]]
                            assert len(selected) == len(selected.unique())


@pytest.mark.parametrize("wrap_y", [False, True])
@pytest.mark.parametrize("radius", [0, 1, 3])
def test_local_attention_matches_naive_unique_neighbor_reference(wrap_y, radius):
    torch.manual_seed(4)
    config = ModelConfig(
        program_channels=1,
        computation_channels=1,
        attention_radius=radius,
        attention_dim=4,
        attention_heads=2,
        attention_distance_bias=True,
        attention_qk_cap=1.7,
    )
    attention = LocalAttention(config, wrap_y)
    state = torch.randn(1, config.channels, 3, 4)
    actual = attention(state)

    batch, _, height, width = state.shape
    qkv = attention.qkv(state).reshape(
        batch, 3, attention.heads, attention.head_dim, height, width
    )
    query, key, value = qkv.unbind(dim=1)
    query = attention._soft_cap(query)
    key = attention._soft_cap(key)
    expected = torch.empty_like(actual).reshape(
        batch, attention.heads, attention.head_dim, height, width
    )
    for y in range(height):
        for x in range(width):
            candidates = []
            for source_y in range(height):
                vertical = abs(source_y - y)
                if wrap_y:
                    vertical = min(vertical, height - vertical)
                if vertical > radius:
                    continue
                for source_x in range(width):
                    horizontal = abs(source_x - x)
                    if horizontal <= radius:
                        candidates.append(
                            (source_y, source_x, max(vertical, horizontal))
                        )
            for head in range(attention.heads):
                logits = torch.stack(
                    [
                        query[0, head, :, y, x].dot(key[0, head, :, source_y, source_x])
                        / math.sqrt(attention.head_dim)
                        + attention.distance_bias[head, distance]
                        for source_y, source_x, distance in candidates
                    ]
                )
                weights = torch.softmax(logits, dim=0)
                expected[0, head, :, y, x] = sum(
                    weight * value[0, head, :, source_y, source_x]
                    for weight, (source_y, source_x, _) in zip(weights, candidates)
                )
    assert torch.allclose(actual, expected.reshape_as(actual), atol=1e-6, rtol=1e-6)


def test_radial_bias_is_per_head_and_isotropic():
    config = ModelConfig(
        attention_radius=1,
        attention_dim=2,
        attention_heads=2,
        attention_distance_bias=True,
    )
    attention = LocalAttention(config, wrap_y=False)
    with torch.no_grad():
        attention.qkv.weight.zero_()
        attention.distance_bias[0] = torch.tensor([math.log(2.0), 0.0])
        attention.distance_bias[1] = torch.tensor([0.0, math.log(2.0)])
    state = torch.zeros(1, config.channels, 3, 5)
    _, weights, mask = attention.attention(state)
    center = weights[0, :, 1, 2]
    assert mask[1, 2].all()
    assert float(center[0, 4]) == pytest.approx(0.2)
    assert torch.allclose(center[0, :4], torch.full((4,), 0.1))
    assert torch.allclose(center[0, 5:], torch.full((4,), 0.1))
    assert float(center[1, 4]) == pytest.approx(1.0 / 17.0)
    assert torch.allclose(center[1, :4], torch.full((4,), 2.0 / 17.0))
    assert torch.allclose(center[1, 5:], torch.full((4,), 2.0 / 17.0))


def test_nope_attention_is_invariant_to_neighbor_permutation():
    config = ModelConfig(attention_dim=4, attention_heads=1)
    attention = LocalAttention(config, wrap_y=False)
    state = torch.randn(1, config.channels, 3, 5)
    swapped = state.clone()
    swapped[..., 0, 2] = state[..., 1, 1]
    swapped[..., 1, 1] = state[..., 0, 2]
    original = attention(state)[..., 1, 2]
    permuted = attention(swapped)[..., 1, 2]
    assert torch.allclose(original, permuted, atol=1e-6, rtol=1e-6)


def test_attention_only_rule_keeps_gating_and_attention_can_be_disabled():
    attention_only_config = ModelConfig(
        convolution_enabled=False,
        attention_enabled=True,
        attention_dim=8,
        attention_heads=2,
        gate="sigmoid",
    )
    attention_only = make_model(attention_only_config)
    assert attention_only.perception is None
    assert attention_only.rule.hidden.in_channels == 8
    assert attention_only.rule.output.out_channels == 2 * attention_only_config.channels

    convolution_only_config = ModelConfig(attention_enabled=False)
    convolution_only = make_model(convolution_only_config)
    assert convolution_only.attention is None
    assert convolution_only.rule.hidden.in_channels == (
        convolution_only_config.channels * convolution_only.perception.kernel_count
    )
    state = torch.randn(2, convolution_only_config.channels, 3, 7)
    expected = state + convolution_only.rule(convolution_only.perception(state)) * (
        convolution_only.update_mask
    )
    assert torch.equal(convolution_only.step(state), expected)


def test_attention_only_gated_rule_backpropagates_through_attention_and_bias():
    config = ModelConfig(
        convolution_enabled=False,
        attention_dim=8,
        attention_heads=2,
        attention_distance_bias=True,
        gate="sigmoid",
        max_abs_state=None,
    )
    model = make_model(config)
    with torch.no_grad():
        model.rule.hidden.weight.fill_(0.1)
        model.rule.hidden.bias.fill_(0.1)
        model.rule.output.weight.fill_(0.1)
        model.rule.output.bias.fill_(0.1)
    state = torch.randn(2, config.channels, 3, 7)
    model.step(state).square().mean().backward()
    assert bool((model.attention.qkv.weight.grad != 0).any())
    assert bool((model.attention.distance_bias.grad != 0).any())


def test_soft_cap_bounds_each_query_and_key_head():
    config = ModelConfig(attention_dim=8, attention_heads=2, attention_qk_cap=2.0)
    attention = LocalAttention(config, wrap_y=False)
    values = torch.randn(3, 2, 4, 3, 5) * 100.0
    capped = attention._soft_cap(values)
    norms = capped.square().sum(dim=2).sqrt()
    assert bool((norms < 2.0).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_attention_cuda_rollouts_are_finite_across_modes_and_radii():
    for convolution_enabled in (False, True):
        for wrap_y in (False, True):
            for radius in (0, 1, 2, 4):
                geometry = GeometryConfig(vertical_space=1, wrap_y=wrap_y)
                config = ModelConfig(
                    convolution_enabled=convolution_enabled,
                    attention_radius=radius,
                    attention_dim=8,
                    attention_heads=2,
                    attention_distance_bias=True,
                    attention_qk_cap=2.0,
                    max_abs_state=None,
                )
                model = NeuralCellularAutomaton(config, geometry, TASKS).cuda()
                with torch.no_grad():
                    model.rule.output.weight.fill_(0.05)
                layout = TapeLayout(geometry, 5)
                inputs = torch.randn(2, 5, device="cuda")
                state = model.initial_state(
                    layout.render_tape(inputs), torch.tensor([0, 1], device="cuda")
                )
                rollout = model(state, 2)
                assert torch.isfinite(rollout).all()
                rollout.square().mean().backward()
                assert torch.isfinite(model.attention.qkv.weight.grad).all()


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

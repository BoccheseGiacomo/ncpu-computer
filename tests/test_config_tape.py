from dataclasses import replace

import pytest
import torch

from ncpu_computer.config import ExperimentConfig, GeometryConfig, ModelConfig
from ncpu_computer.tape import (
    TERNARY_THRESHOLD,
    TapeLayout,
    binary_to_integer,
    encode_strings,
    integer_to_binary,
    interpret_tape,
    quantize,
    tensor_to_symbols,
)


def test_config_round_trip_and_channel_roles():
    config = ExperimentConfig(
        geometry=GeometryConfig(tape_slots=7, stride=3),
        model=ModelConfig(program_channels=2, computation_channels=4),
    )
    assert ExperimentConfig.from_dict(config.to_dict()) == config
    assert config.model.input_channel == 2
    assert config.model.output_channel == 3
    assert config.model.channels == 8
    shared = ModelConfig(
        program_channels=2,
        computation_channels=4,
        io_mode="shared",
        input_mode="mutable",
    )
    assert shared.input_channel == shared.output_channel == 2
    assert shared.channels == 7
    with pytest.raises(ValueError, match="shared I/O"):
        replace(shared, input_mode="frozen").validate()
    with pytest.raises(ValueError, match="tape_slots"):
        replace(config.geometry, tape_slots=0).validate()
    with pytest.raises(TypeError, match="integers"):
        replace(config.geometry, tape_slots=3.5).validate()


def test_layout_renders_only_strided_tape_cells():
    geometry = GeometryConfig(
        tape_slots=4,
        stride=2,
        border_left=1,
        border_right=0,
        border_top=1,
        border_bottom=2,
    )
    layout = TapeLayout(geometry)
    values = torch.tensor([[1.0, -1.0, 0.0, 1.0]])
    grid = layout.render_tape(values)
    assert grid.shape == (1, 4, 8)
    assert torch.equal(layout.extract_tape(grid), values)
    assert layout.tape_coordinates == ((1, 1), (1, 3), (1, 5), (1, 7))
    assert torch.count_nonzero(grid) == 3


def test_string_encoding_is_direct_ternary_with_implicit_blank_padding():
    encoded = encode_strings(("10B", "", "B01"), tape_slots=4)
    assert encoded.tolist() == [
        [1.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 1.0, 0.0],
    ]
    values = torch.tensor(
        [
            -TERNARY_THRESHOLD - 1e-6,
            -TERNARY_THRESHOLD,
            TERNARY_THRESHOLD,
            TERNARY_THRESHOLD + 1e-6,
        ]
    )
    assert quantize(values).tolist() == [-1, 0, 0, 1]
    assert tensor_to_symbols(encoded[0]) == "10BB"


def test_integer_codec_is_minimal_binary():
    assert integer_to_binary(0) == "0"
    assert integer_to_binary(11) == "1011"
    assert binary_to_integer("1011") == 11
    with pytest.raises(ValueError):
        binary_to_integer("1B1")
    with pytest.raises(ValueError, match="minimal"):
        binary_to_integer("00")


def test_interpreters_accept_explicit_blank_or_tape_boundary():
    single = interpret_tape(torch.tensor([1.0, -1.0, 1.0, 1.0, 0.0]))
    assert single.valid and single.terminated
    assert single.binary_strings == ("1011",)
    assert single.integers == (11,)
    full = interpret_tape(torch.tensor([1.0, -1.0, 1.0, 1.0]))
    assert full.valid and not full.terminated
    empty = interpret_tape(torch.zeros(3))
    assert empty.valid and empty.binary_strings == ("",)
    assert not empty.integer_valid

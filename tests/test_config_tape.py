from dataclasses import replace

import pytest
import torch

from ncpu_computer import (
    ExperimentConfig,
    GeometryConfig,
    ModelConfig,
    TapeLayout,
    TestCase,
    TERNARY_THRESHOLD,
    binary_to_integer,
    encode_strings,
    integer_strata,
    integer_to_binary,
    interpret_tape,
    quantize,
    tensor_to_symbols,
)


def test_config_round_trip_and_channel_roles():
    config = ExperimentConfig(
        geometry=GeometryConfig(stride=3, vertical_space=2, horizontal_space=6),
        model=ModelConfig(program_channels=2, computation_channels=4),
        test_cases=(TestCase("test", 8, 4, 3, 5),),
    )
    assert ExperimentConfig.from_dict(config.to_dict()) == config
    assert config.model.io_channel == 2
    assert config.model.channels == 7
    assert config.geometry.height == 5
    with pytest.raises(ValueError, match="divisible"):
        replace(config.geometry, horizontal_space=2).validate()
    with pytest.raises(ValueError, match="zero program"):
        replace(
            config,
            training=replace(config.training, train_program=True),
        ).validate()
    with pytest.raises(ValueError, match="one blank"):
        replace(
            config,
            training=replace(
                config.training,
                n_trials=2,
                tape_slots_min=3,
                tape_slots_max=6,
                input_max_length_min=4,
                input_max_length_max=5,
                free_steps_min=1,
                free_steps_max=2,
            ),
        ).validate()


def test_integer_strata_are_complete_disjoint_and_nonempty():
    assert integer_strata(2, 8, 3) == ((2, 4), (5, 6), (7, 8))
    with pytest.raises(ValueError, match="distinct"):
        integer_strata(1, 2, 3)


def test_layout_is_symmetric_and_centers_strided_cells():
    geometry = GeometryConfig(stride=2, vertical_space=1, horizontal_space=2)
    layout = TapeLayout(geometry, 4)
    values = torch.tensor([[1.0, -1.0, 0.0, 1.0]])
    grid = layout.render_tape(values)
    assert grid.shape == (1, 3, 12)
    assert layout.tape_coordinates == ((1, 3), (1, 5), (1, 7), (1, 9))
    assert torch.equal(layout.extract_tape(grid), values)
    assert torch.count_nonzero(grid) == 3


def test_direct_ternary_codec_and_interpreter():
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
    interpreted = interpret_tape(encoded[0])
    assert interpreted.binary_strings == ("10",)
    assert interpreted.terminated


def test_integer_codec_is_minimal_binary():
    assert integer_to_binary(0) == "0"
    assert integer_to_binary(11) == "1011"
    assert binary_to_integer("1011") == 11
    with pytest.raises(ValueError, match="minimal"):
        binary_to_integer("00")

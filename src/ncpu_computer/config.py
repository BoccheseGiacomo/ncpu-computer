from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


SUPPORTED_FIXED_KERNELS = {"identity", "sobel_x", "sobel_y"}
SUPPORTED_GATES = {"none", "linear", "sigmoid", "tanh", "relu"}
PROGRAM_MODES = {"zero", "learned_read_only", "learned_mutable"}


@dataclass(frozen=True)
class GeometryConfig:
    stride: int = 2
    vertical_space: int = 1
    horizontal_space: int = 2
    wrap_y: bool = False

    def validate(self) -> None:
        values = (self.stride, self.vertical_space, self.horizontal_space)
        if any(type(value) is not int for value in values):
            raise TypeError("geometry dimensions must be integers")
        if self.stride < 1:
            raise ValueError("stride must be positive")
        if self.vertical_space < 0 or self.horizontal_space < 0:
            raise ValueError("spaces cannot be negative")
        if self.horizontal_space % self.stride:
            raise ValueError("horizontal_space must be divisible by stride")
        if type(self.wrap_y) is not bool:
            raise TypeError("wrap_y must be a boolean")

    @property
    def height(self) -> int:
        return 2 * self.vertical_space + 1


@dataclass(frozen=True)
class ModelConfig:
    program_channels: int = 1
    computation_channels: int = 3
    program_mode: str = "zero"
    program_init_std: float = 0.1
    hidden_size: int = 96
    fixed_kernels: tuple[str, ...] = ("identity", "sobel_x", "sobel_y")
    fixed_laplacian: bool = False
    learnable_kernels: int = 1
    learnable_kernel_init: str = "laplacian"
    gate: str = "none"
    gate_bias: float = 1.0
    fire_rate: float = 1.0
    max_abs_state: float | None = 10.0
    random_kernel_seed: int = 0

    @property
    def io_channel(self) -> int:
        return self.program_channels

    @property
    def channels(self) -> int:
        return self.program_channels + 1 + self.computation_channels

    @property
    def program_mutable(self) -> bool:
        return self.program_mode == "learned_mutable"

    def validate(self) -> None:
        integers = (
            self.program_channels,
            self.computation_channels,
            self.hidden_size,
            self.learnable_kernels,
            self.random_kernel_seed,
        )
        if any(type(value) is not int for value in integers):
            raise TypeError("model dimensions, channels, and seeds must be integers")
        if self.program_channels < 1:
            raise ValueError("program_channels must be positive")
        if self.computation_channels < 0:
            raise ValueError("computation_channels cannot be negative")
        if self.program_mode not in PROGRAM_MODES:
            raise ValueError(f"program_mode must be one of {sorted(PROGRAM_MODES)}")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        if self.random_kernel_seed < 0:
            raise ValueError("random_kernel_seed cannot be negative")
        numeric = (self.program_init_std, self.gate_bias, self.fire_rate)
        if self.max_abs_state is not None:
            numeric += (self.max_abs_state,)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("model scalar hyperparameters must be finite")
        if self.program_init_std < 0:
            raise ValueError("program_init_std cannot be negative")
        if (
            not self.fixed_kernels
            and not self.fixed_laplacian
            and not self.learnable_kernels
        ):
            raise ValueError("at least one perception kernel is required")
        unknown = set(self.fixed_kernels) - SUPPORTED_FIXED_KERNELS
        if unknown:
            raise ValueError(f"unsupported fixed kernels: {sorted(unknown)}")
        if len(set(self.fixed_kernels)) != len(self.fixed_kernels):
            raise ValueError("fixed perception kernels must be unique")
        if self.learnable_kernels < 0:
            raise ValueError("learnable_kernels cannot be negative")
        if self.learnable_kernel_init not in {"laplacian", "random"}:
            raise ValueError("learnable_kernel_init must be 'laplacian' or 'random'")
        if self.gate not in SUPPORTED_GATES:
            raise ValueError(f"gate must be one of {sorted(SUPPORTED_GATES)}")
        if not 0.0 < self.fire_rate <= 1.0:
            raise ValueError("fire_rate must be in (0, 1]")
        if self.max_abs_state is not None and self.max_abs_state <= 0:
            raise ValueError("max_abs_state must be positive or None")


@dataclass(frozen=True)
class TestCase:
    __test__ = False

    name: str
    tape_slots: int
    input_length: int
    free_steps: int
    supervision_steps: int

    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("test case name must be a non-empty string")
        values = (
            self.tape_slots,
            self.input_length,
            self.free_steps,
            self.supervision_steps,
        )
        if any(type(value) is not int for value in values):
            raise TypeError("test case sizes and times must be integers")
        if self.tape_slots < 2:
            raise ValueError("test tape must have at least two slots")
        if not 0 <= self.input_length <= self.tape_slots - 1:
            raise ValueError("test input_length must be in [0, tape_slots - 1]")
        if self.free_steps < 0 or self.supervision_steps < 1:
            raise ValueError("test evolution times are invalid")

    @property
    def steps(self) -> int:
        return self.free_steps + self.supervision_steps


@dataclass(frozen=True)
class TrainingConfig:
    updates: int = 3000
    batch_size_per_task: int = 64
    n_trials: int = 3
    tape_slots_min: int = 8
    tape_slots_max: int = 14
    input_max_length_min: int = 2
    input_max_length_max: int = 6
    free_steps_min: int = 40
    free_steps_max: int = 100
    supervision_ratio: float = 1.6
    learning_rate: float = 2e-3
    final_learning_rate: float = 1e-4
    warmup_updates: int = 0
    weight_decay: float = 2e-5
    program_weight_decay: float = 1e-4
    grad_clip: float | None = 0.8
    train_rule: bool = True
    train_program: bool = False
    perception_noise_start: float = 0.0
    perception_noise_end: float = 0.0
    seed: int = 0
    validation_every: int = 50
    checkpoint_every: int = 50
    device: str = "auto"

    def validate(self) -> None:
        integers = (
            self.updates,
            self.batch_size_per_task,
            self.n_trials,
            self.tape_slots_min,
            self.tape_slots_max,
            self.input_max_length_min,
            self.input_max_length_max,
            self.free_steps_min,
            self.free_steps_max,
            self.warmup_updates,
            self.seed,
            self.validation_every,
            self.checkpoint_every,
        )
        if any(type(value) is not int for value in integers):
            raise TypeError("training counts, ranges, and seed must be integers")
        if any(
            type(value) is not bool for value in (self.train_rule, self.train_program)
        ):
            raise TypeError("training flags must be booleans")
        numeric = (
            self.supervision_ratio,
            self.learning_rate,
            self.final_learning_rate,
            self.weight_decay,
            self.program_weight_decay,
            self.perception_noise_start,
            self.perception_noise_end,
        )
        if self.grad_clip is not None:
            numeric += (self.grad_clip,)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("training scalar hyperparameters must be finite")
        if self.updates < 1 or self.batch_size_per_task < 1 or self.n_trials < 1:
            raise ValueError("updates, batch size, and n_trials must be positive")
        ranges = (
            (self.tape_slots_min, self.tape_slots_max, "tape slots"),
            (self.input_max_length_min, self.input_max_length_max, "input length"),
            (self.free_steps_min, self.free_steps_max, "free steps"),
        )
        for lower, upper, name in ranges:
            if lower < 0 or lower > upper:
                raise ValueError(f"invalid {name} range")
            if upper - lower + 1 < self.n_trials:
                raise ValueError(
                    f"{name} range cannot provide n_trials distinct values"
                )
        if self.tape_slots_min < 2:
            raise ValueError("tape_slots_min must be at least two")
        if self.free_steps_min < 1:
            raise ValueError("free_steps_min must be positive")
        if self.supervision_ratio <= 0:
            raise ValueError("supervision_ratio must be positive")
        if round(self.supervision_ratio * self.free_steps_min) < 1:
            raise ValueError("supervision_ratio produces an empty supervision window")
        if self.input_max_length_max > self.tape_slots_max - 1:
            raise ValueError("input range exceeds the largest feasible tape")
        if self.learning_rate <= 0 or self.final_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if not 0 <= self.warmup_updates < self.updates:
            raise ValueError("warmup_updates must be in [0, updates)")
        if self.weight_decay < 0 or self.program_weight_decay < 0:
            raise ValueError("weight decays cannot be negative")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive or None")
        if self.perception_noise_start < 0 or self.perception_noise_end < 0:
            raise ValueError("perception noise cannot be negative")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        if self.validation_every < 1 or self.checkpoint_every < 1:
            raise ValueError("validation and checkpoint intervals must be positive")
        if self.device != "auto" and not (
            self.device == "cpu" or self.device.startswith("cuda")
        ):
            raise ValueError("device must be 'auto', 'cpu', or a CUDA device")


@dataclass(frozen=True)
class ExperimentConfig:
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    test_cases: tuple[TestCase, ...] = (
        TestCase("in_distribution", 10, 6, 80, 128),
        TestCase("longer_input", 10, 8, 100, 160),
        TestCase("larger_grid", 14, 10, 130, 208),
    )

    def validate(self) -> None:
        self.geometry.validate()
        self.model.validate()
        self.training.validate()
        if not self.test_cases:
            raise ValueError("at least one test case is required")
        for case in self.test_cases:
            case.validate()
        if len({case.name for case in self.test_cases}) != len(self.test_cases):
            raise ValueError("test case names must be unique")
        if not self.training.train_rule and not self.training.train_program:
            raise ValueError(
                "at least one of train_rule and train_program must be true"
            )
        if self.training.train_program and self.model.program_mode == "zero":
            raise ValueError("a zero program cannot be trained")
        _validate_stratified_feasibility(self.training)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        model_data = dict(data["model"])
        model_data["fixed_kernels"] = tuple(model_data["fixed_kernels"])
        config = cls(
            geometry=GeometryConfig(**data["geometry"]),
            model=ModelConfig(**model_data),
            training=TrainingConfig(**data["training"]),
            test_cases=tuple(TestCase(**case) for case in data["test_cases"]),
        )
        config.validate()
        return config


def integer_strata(lower: int, upper: int, count: int) -> tuple[tuple[int, int], ...]:
    size = upper - lower + 1
    if count < 1 or size < count:
        raise ValueError("integer range cannot provide the requested distinct strata")
    quotient, remainder = divmod(size, count)
    result = []
    start = lower
    for index in range(count):
        width = quotient + (index < remainder)
        result.append((start, start + width - 1))
        start += width
    return tuple(result)


def _validate_stratified_feasibility(training: TrainingConfig) -> None:
    tape = integer_strata(
        training.tape_slots_min, training.tape_slots_max, training.n_trials
    )
    inputs = integer_strata(
        training.input_max_length_min,
        training.input_max_length_max,
        training.n_trials,
    )
    for (tape_low, tape_high), (input_low, _) in zip(tape, inputs):
        if tape_low < input_low + 1:
            raise ValueError(
                "every tape value must fit its coupled input stratum and one blank"
            )

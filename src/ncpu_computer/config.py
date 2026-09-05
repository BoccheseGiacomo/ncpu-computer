from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


SUPPORTED_FIXED_KERNELS = {"identity", "sobel_x", "sobel_y"}
SUPPORTED_GATES = {"none", "linear", "sigmoid", "tanh", "relu"}


@dataclass(frozen=True)
class GeometryConfig:
    tape_slots: int = 12
    stride: int = 2
    border_left: int = 3
    border_right: int = 3
    border_top: int = 3
    border_bottom: int = 3

    def validate(self) -> None:
        values = (
            self.tape_slots,
            self.stride,
            self.border_left,
            self.border_right,
            self.border_top,
            self.border_bottom,
        )
        if any(type(value) is not int for value in values):
            raise TypeError("geometry dimensions must be integers")
        if self.tape_slots < 1:
            raise ValueError("tape_slots must be at least 1")
        if self.stride < 1:
            raise ValueError("stride must be positive")
        borders = (
            self.border_left,
            self.border_right,
            self.border_top,
            self.border_bottom,
        )
        if any(border < 0 for border in borders):
            raise ValueError("borders cannot be negative")


@dataclass(frozen=True)
class ModelConfig:
    channels: int = 5
    hidden_size: int = 96
    fixed_kernels: tuple[str, ...] = ("identity", "sobel_x", "sobel_y")
    fixed_laplacian: bool = False
    learnable_kernels: int = 1
    learnable_kernel_init: str = "laplacian"
    gate: str = "none"
    gate_bias: float = 1.0
    fire_rate: float = 1.0
    program_channel: int = 0
    io_channel: int = 1
    padding: str = "zeros"
    max_abs_state: float | None = 10.0
    random_kernel_seed: int = 0

    def validate(self) -> None:
        integers = (
            self.channels,
            self.hidden_size,
            self.learnable_kernels,
            self.program_channel,
            self.io_channel,
            self.random_kernel_seed,
        )
        if any(type(value) is not int for value in integers):
            raise TypeError("model dimensions, channels, and seeds must be integers")
        if self.random_kernel_seed < 0:
            raise ValueError("random_kernel_seed cannot be negative")
        numeric = (self.gate_bias, self.fire_rate)
        if self.max_abs_state is not None:
            numeric += (self.max_abs_state,)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("model scalar hyperparameters must be finite")
        if self.channels < 2:
            raise ValueError("channels must be at least 2")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be at least 1")
        if not 0 <= self.program_channel < self.channels:
            raise ValueError("program_channel must refer to an existing channel")
        if not 0 <= self.io_channel < self.channels:
            raise ValueError("io_channel must refer to an existing channel")
        if self.program_channel == self.io_channel:
            raise ValueError("program and I/O channels must be distinct")
        if (
            not self.fixed_kernels
            and not self.fixed_laplacian
            and self.learnable_kernels == 0
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
        if self.padding not in {"zeros", "reflect", "replicate", "circular"}:
            raise ValueError("unsupported padding mode")
        if self.max_abs_state is not None and self.max_abs_state <= 0:
            raise ValueError("max_abs_state must be positive or None")


@dataclass(frozen=True)
class TrainingConfig:
    updates: int = 3000
    batch_size: int = 256
    free_steps: int = 60
    supervision_steps: int = 140
    learning_rate: float = 2e-3
    final_learning_rate: float = 1e-4
    warmup_updates: int = 0
    weight_decay: float = 2e-5
    grad_clip: float | None = 0.8
    terminator_weight: float = 0.0
    tail_weight: float = 0.0
    seed: int = 0
    validation_every: int = 50
    checkpoint_every: int = 50
    device: str = "auto"

    @property
    def rollout_steps(self) -> int:
        return self.free_steps + self.supervision_steps

    @property
    def supervision_start(self) -> int:
        return self.free_steps + 1

    @property
    def supervision_end(self) -> int:
        return self.rollout_steps

    def validate(self) -> None:
        integers = (
            self.updates,
            self.batch_size,
            self.free_steps,
            self.supervision_steps,
            self.warmup_updates,
            self.seed,
            self.validation_every,
            self.checkpoint_every,
        )
        if any(type(value) is not int for value in integers):
            raise TypeError("training counts and seed must be integers")
        if self.seed < 0:
            raise ValueError("seed cannot be negative")
        numeric = (
            self.learning_rate,
            self.final_learning_rate,
            self.weight_decay,
            self.terminator_weight,
            self.tail_weight,
        )
        if self.grad_clip is not None:
            numeric += (self.grad_clip,)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("training scalar hyperparameters must be finite")
        if self.updates < 1 or self.batch_size < 1:
            raise ValueError("updates and batch_size must be positive")
        if self.free_steps < 0 or self.supervision_steps < 1:
            raise ValueError(
                "free_steps must be non-negative and supervision_steps positive"
            )
        if self.learning_rate <= 0 or self.final_learning_rate <= 0:
            raise ValueError("learning rates must be positive")
        if not 0 <= self.warmup_updates < self.updates:
            raise ValueError("warmup_updates must be in [0, updates)")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.grad_clip is not None and self.grad_clip <= 0:
            raise ValueError("grad_clip must be positive or None")
        if self.terminator_weight < 0 or self.tail_weight < 0:
            raise ValueError("auxiliary loss weights cannot be negative")
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

    def validate(self) -> None:
        self.geometry.validate()
        self.model.validate()
        self.training.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentConfig":
        geometry = GeometryConfig(**data["geometry"])
        model_data = dict(data["model"])
        model_data["fixed_kernels"] = tuple(model_data["fixed_kernels"])
        model = ModelConfig(**model_data)
        training = TrainingConfig(**data["training"])
        config = cls(geometry=geometry, model=model, training=training)
        config.validate()
        return config

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .config import GeometryConfig


TERNARY_THRESHOLD = 0.333
SYMBOL_TO_VALUE = {"0": -1.0, "B": 0.0, "1": 1.0}
VALUE_TO_SYMBOL = {-1: "0", 0: "B", 1: "1"}


def validate_symbols(symbols: str, *, allow_empty: bool = False) -> None:
    if not isinstance(symbols, str):
        raise TypeError("symbols must be a string")
    if not symbols and not allow_empty:
        raise ValueError("symbol strings cannot be empty")
    unknown = set(symbols) - set(SYMBOL_TO_VALUE)
    if unknown:
        raise ValueError(f"unsupported symbols: {sorted(unknown)}")


def encode_strings(strings: Sequence[str], tape_slots: int) -> torch.Tensor:
    if tape_slots < 1:
        raise ValueError("tape_slots must be positive")
    if not strings:
        raise ValueError("at least one string is required")
    encoded = torch.zeros(len(strings), tape_slots, dtype=torch.float32)
    for row, symbols in enumerate(strings):
        validate_symbols(symbols)
        if len(symbols) > tape_slots:
            raise ValueError(
                f"string of length {len(symbols)} exceeds {tape_slots} tape slots"
            )
        encoded[row, : len(symbols)] = torch.tensor(
            [SYMBOL_TO_VALUE[symbol] for symbol in symbols]
        )
    return encoded


def quantize(values: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(values)
    if not torch.is_floating_point(values):
        raise ValueError("values must be floating point")
    if not torch.isfinite(values).all():
        raise ValueError("values must be finite")
    positive = values > TERNARY_THRESHOLD
    negative = values < -TERNARY_THRESHOLD
    return positive.to(torch.int8) - negative.to(torch.int8)


def tensor_to_symbols(values: torch.Tensor) -> str:
    values = torch.as_tensor(values)
    if values.ndim != 1:
        raise ValueError("single-example inference expects a one-dimensional tape")
    discrete = quantize(values).cpu().tolist()
    return "".join(VALUE_TO_SYMBOL[value] for value in discrete)


def integer_to_binary(value: int) -> str:
    if type(value) is not int or value < 0:
        raise ValueError("value must be a non-negative integer")
    return format(value, "b")


def binary_to_integer(symbols: str) -> int:
    if not symbols or set(symbols) - {"0", "1"}:
        raise ValueError("a binary string must contain one or more 0/1 symbols")
    if len(symbols) > 1 and symbols.startswith("0"):
        raise ValueError("integer strings must use minimal binary representation")
    return int(symbols, 2)


@dataclass(frozen=True)
class InterpretedTape:
    raw: str
    binary_strings: tuple[str, ...]
    integers: tuple[int, ...]
    terminated: bool
    valid: bool
    integer_valid: bool


def interpret_tape(values: torch.Tensor, mode: str = "single") -> InterpretedTape:
    if mode not in {"single", "multiple"}:
        raise ValueError("mode must be 'single' or 'multiple'")
    raw = tensor_to_symbols(values)
    if mode == "single":
        end = raw.find("B")
        terminated = end >= 0
        binary_strings = (raw[:end],) if terminated and end > 0 else ()
    else:
        end = raw.find("BB")
        terminated = end >= 0
        prefix = raw[:end] if terminated else raw
        parts = prefix.split("B") if prefix else []
        binary_strings = tuple(parts) if all(parts) else ()
    valid = terminated and bool(binary_strings)
    integer_valid = valid and all(
        len(value) == 1 or value.startswith("1") for value in binary_strings
    )
    integers = (
        tuple(binary_to_integer(value) for value in binary_strings)
        if integer_valid
        else ()
    )
    return InterpretedTape(
        raw=raw,
        binary_strings=binary_strings,
        integers=integers,
        terminated=terminated,
        valid=valid,
        integer_valid=integer_valid,
    )


@dataclass(frozen=True)
class TapeLayout:
    config: GeometryConfig

    def __post_init__(self) -> None:
        self.config.validate()

    @property
    def height(self) -> int:
        return self.config.border_top + 1 + self.config.border_bottom

    @property
    def width(self) -> int:
        return (
            self.config.border_left
            + (self.config.tape_slots - 1) * self.config.stride
            + 1
            + self.config.border_right
        )

    @property
    def tape_row(self) -> int:
        return self.config.border_top

    @property
    def tape_slice(self) -> slice:
        start = self.config.border_left
        stop = start + self.config.tape_slots * self.config.stride
        return slice(start, stop, self.config.stride)

    @property
    def tape_coordinates(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (
                self.tape_row,
                self.config.border_left + index * self.config.stride,
            )
            for index in range(self.config.tape_slots)
        )

    def render_tape(self, values: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(values)
        if values.ndim < 1 or values.shape[-1] != self.config.tape_slots:
            raise ValueError(
                f"values must end with a dimension of {self.config.tape_slots}"
            )
        grid = torch.zeros(
            *values.shape[:-1],
            self.height,
            self.width,
            device=values.device,
            dtype=values.dtype,
        )
        grid[..., self.tape_row, self.tape_slice] = values
        return grid

    def extract_tape(self, grid: torch.Tensor) -> torch.Tensor:
        grid = torch.as_tensor(grid)
        if grid.ndim < 2 or grid.shape[-2:] != (self.height, self.width):
            raise ValueError(
                f"grid must end with dimensions ({self.height}, {self.width})"
            )
        return grid[..., self.tape_row, self.tape_slice]

    def schema(self) -> str:
        cells = [["." for _ in range(self.width)] for _ in range(self.height)]
        for row, column in self.tape_coordinates:
            cells[row][column] = "T"
        return "\n".join(" ".join(row) for row in cells) + ("\nT: logical tape cell")

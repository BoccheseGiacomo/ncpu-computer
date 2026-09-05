from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import product

import torch

from .tape import encode_strings, integer_to_binary, validate_symbols


@dataclass(frozen=True)
class StringExample:
    input: str
    target: str


@dataclass(frozen=True)
class StringTask:
    name: str
    examples: tuple[StringExample, ...]
    output_mode: str = "single"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("task name cannot be empty")
        if not self.examples:
            raise ValueError("a task must contain at least one example")
        if self.output_mode not in {"single", "multiple"}:
            raise ValueError("output_mode must be 'single' or 'multiple'")
        for example in self.examples:
            validate_symbols(example.input)
            validate_symbols(example.target)
            if example.target.startswith("B") or example.target.endswith("B"):
                raise ValueError("targets cannot begin or end with B")
            if self.output_mode == "single" and "B" in example.target:
                raise ValueError("single-output targets cannot contain B")
            if self.output_mode == "multiple":
                if "BB" in example.target:
                    raise ValueError("multiple-output values use single-B separators")
                if any(not part for part in example.target.split("B")):
                    raise ValueError("multiple-output targets contain an empty value")


def addition_task(operand_bits: int) -> StringTask:
    if operand_bits < 1:
        raise ValueError("operand_bits must be positive")
    limit = 1 << operand_bits
    examples = tuple(
        StringExample(
            input=f"{integer_to_binary(a)}B{integer_to_binary(b)}",
            target=integer_to_binary(a + b),
        )
        for a in range(limit)
        for b in range(limit)
    )
    return StringTask(name=f"addition-{operand_bits}-bit", examples=examples)


def binary_strings(max_length: int, *, include_shorter: bool = True) -> tuple[str, ...]:
    if max_length < 1:
        raise ValueError("max_length must be positive")
    lengths = range(1, max_length + 1) if include_shorter else (max_length,)
    return tuple(
        "".join(bits) for length in lengths for bits in product("01", repeat=length)
    )


def reverse_task(max_length: int, *, include_shorter: bool = True) -> StringTask:
    strings = binary_strings(max_length, include_shorter=include_shorter)
    return StringTask(
        name=f"reverse-up-to-{max_length}",
        examples=tuple(StringExample(value, value[::-1]) for value in strings),
    )


def parity_task(max_length: int, *, include_shorter: bool = True) -> StringTask:
    strings = binary_strings(max_length, include_shorter=include_shorter)
    return StringTask(
        name=f"parity-up-to-{max_length}",
        examples=tuple(
            StringExample(value, "1" if value.count("1") % 2 else "0")
            for value in strings
        ),
    )


@dataclass(frozen=True)
class TaskDataset:
    name: str
    output_mode: str
    input_strings: tuple[str, ...]
    target_strings: tuple[str, ...]
    inputs: torch.Tensor
    targets: torch.Tensor
    target_lengths: torch.Tensor
    terminator_mask: torch.Tensor
    tail_mask: torch.Tensor

    @classmethod
    def from_task(cls, task: StringTask, tape_slots: int) -> "TaskDataset":
        inputs = tuple(example.input for example in task.examples)
        targets = tuple(example.target for example in task.examples)
        terminator_width = 1 if task.output_mode == "single" else 2
        if max(map(len, inputs)) > tape_slots:
            raise ValueError("an input exceeds the configured tape capacity")
        if max(map(len, targets)) + terminator_width > tape_slots:
            raise ValueError("the tape must leave room for the output terminator")
        encoded_inputs = encode_strings(inputs, tape_slots)
        encoded_targets = encode_strings(targets, tape_slots)
        lengths = torch.tensor([len(target) for target in targets], dtype=torch.int64)
        positions = torch.arange(tape_slots).unsqueeze(0)
        terminator_mask = (positions >= lengths.unsqueeze(1)) & (
            positions < lengths.unsqueeze(1) + terminator_width
        )
        tail_mask = positions >= lengths.unsqueeze(1) + terminator_width
        return cls(
            name=task.name,
            output_mode=task.output_mode,
            input_strings=inputs,
            target_strings=targets,
            inputs=encoded_inputs,
            targets=encoded_targets,
            target_lengths=lengths,
            terminator_mask=terminator_mask,
            tail_mask=tail_mask,
        )

    def __post_init__(self) -> None:
        count = len(self.input_strings)
        if count < 1 or len(self.target_strings) != count:
            raise ValueError("dataset strings have inconsistent lengths")
        if self.inputs.ndim != 2 or self.targets.shape != self.inputs.shape:
            raise ValueError("inputs and targets must be matching rank-two tensors")
        if self.inputs.shape[0] != count:
            raise ValueError("tensor and string example counts differ")
        expected_vector = (count,)
        expected_matrix = self.inputs.shape
        if self.target_lengths.shape != expected_vector:
            raise ValueError("target_lengths has the wrong shape")
        if self.terminator_mask.shape != expected_matrix:
            raise ValueError("terminator_mask has the wrong shape")
        if self.tail_mask.shape != expected_matrix:
            raise ValueError("tail_mask has the wrong shape")

    def __len__(self) -> int:
        return len(self.input_strings)

    @property
    def tape_slots(self) -> int:
        return self.inputs.shape[1]

    @property
    def signature(self) -> str:
        data = {
            "name": self.name,
            "output_mode": self.output_mode,
            "inputs": self.input_strings,
            "targets": self.target_strings,
        }
        encoded = json.dumps(data, separators=(",", ":"), ensure_ascii=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def take(self, indices: torch.Tensor) -> tuple[torch.Tensor, ...]:
        indices = torch.as_tensor(indices, dtype=torch.int64, device="cpu")
        return (
            self.inputs[indices],
            self.targets[indices],
            self.target_lengths[indices],
            self.terminator_mask[indices],
            self.tail_mask[indices],
        )

    def sample(
        self, batch_size: int, generator: torch.Generator
    ) -> tuple[torch.Tensor, ...]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        indices = torch.randint(len(self), (batch_size,), generator=generator)
        return self.take(indices)


def semantic_correct(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_lengths: torch.Tensor,
    output_mode: str,
) -> torch.Tensor:
    """Compare a quantized batch through the required output terminator."""
    prediction = torch.as_tensor(prediction)
    target = torch.as_tensor(target, device=prediction.device)
    target_lengths = torch.as_tensor(
        target_lengths, dtype=torch.int64, device=prediction.device
    )
    squeeze_time = prediction.ndim == 2
    if squeeze_time:
        prediction = prediction.unsqueeze(1)
    if prediction.ndim != 3:
        raise ValueError(
            "prediction must have shape (batch, time, tape) or (batch, tape)"
        )
    batch, _, tape_slots = prediction.shape
    if target.shape != (batch, tape_slots):
        raise ValueError("target shape does not match prediction")
    if target_lengths.shape != (batch,):
        raise ValueError("target_lengths shape does not match prediction")
    if output_mode not in {"single", "multiple"}:
        raise ValueError("output_mode must be 'single' or 'multiple'")
    terminator_width = 1 if output_mode == "single" else 2
    required_end = target_lengths + terminator_width
    if bool((required_end > tape_slots).any()):
        raise ValueError("target and terminator exceed the tape")
    positions = torch.arange(tape_slots, device=prediction.device).view(1, 1, -1)
    required = positions < required_end.view(batch, 1, 1)
    correct = ((prediction == target.unsqueeze(1)) | ~required).all(dim=-1)
    return correct.squeeze(1) if squeeze_time else correct

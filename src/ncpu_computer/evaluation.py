from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import GeometryConfig
from .model import NeuralCellularAutomaton
from .tape import InterpretedTape, TapeLayout, encode_strings, interpret_tape, quantize
from .tasks import TaskDataset, semantic_correct


@dataclass(frozen=True)
class EvaluationResult:
    task: str
    examples: int
    total_examples: int
    tape_slots: int
    batch_size: int
    seed: int
    step_start: int
    step_end: int
    mean_mse: float
    mean_semantic_accuracy: float
    mean_raw_accuracy: float
    mean_symbol_accuracy: float
    stable_semantic_accuracy: float
    stable_raw_accuracy: float
    best_semantic_step: int
    best_semantic_accuracy: float
    best_mse_step: int
    best_mse: float
    mse_by_step: tuple[float, ...]
    semantic_by_step: tuple[float, ...]
    raw_by_step: tuple[float, ...]
    symbol_by_step: tuple[float, ...]

    def summary(self) -> dict[str, float | int | str]:
        return {
            "task": self.task,
            "examples": self.examples,
            "total_examples": self.total_examples,
            "tape_slots": self.tape_slots,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "steps": f"{self.step_start}-{self.step_end}",
            "mse": self.mean_mse,
            "semantic": self.mean_semantic_accuracy,
            "raw": self.mean_raw_accuracy,
            "symbol": self.mean_symbol_accuracy,
            "stable_semantic": self.stable_semantic_accuracy,
            "stable_raw": self.stable_raw_accuracy,
            "best_semantic_step": self.best_semantic_step,
            "best_semantic": self.best_semantic_accuracy,
            "best_mse_step": self.best_mse_step,
            "best_mse": self.best_mse,
        }


@torch.no_grad()
def evaluate(
    model: NeuralCellularAutomaton,
    geometry: GeometryConfig,
    dataset: TaskDataset,
    *,
    steps: int,
    step_start: int,
    step_end: int,
    batch_size: int = 256,
    max_examples: int | None = None,
    seed: int = 0,
) -> EvaluationResult:
    if dataset.tape_slots != geometry.tape_slots:
        raise ValueError("dataset and geometry have different tape capacities")
    counts = (steps, step_start, step_end, batch_size, seed)
    if any(type(value) is not int for value in counts):
        raise TypeError("steps, batch_size, and seed must be integers")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if seed < 0:
        raise ValueError("seed cannot be negative")
    if not 0 <= step_start <= step_end <= steps:
        raise ValueError("evaluation steps must satisfy 0 <= start <= end <= steps")
    if max_examples is not None:
        if type(max_examples) is not int:
            raise TypeError("max_examples must be an integer or None")
        if max_examples < 1:
            raise ValueError("max_examples must be positive or None")

    generator = torch.Generator().manual_seed(seed)
    if max_examples is not None and max_examples < len(dataset):
        indices = torch.randperm(len(dataset), generator=generator)[:max_examples]
    else:
        indices = torch.arange(len(dataset))
    count = len(indices)
    step_count = steps + 1
    mse_sums = torch.zeros(step_count, dtype=torch.float64)
    semantic_sums = torch.zeros(step_count, dtype=torch.float64)
    raw_sums = torch.zeros(step_count, dtype=torch.float64)
    symbol_sums = torch.zeros(step_count, dtype=torch.float64)
    stable_semantic_sum = 0.0
    stable_raw_sum = 0.0
    layout = TapeLayout(geometry)
    device = model.device
    was_training = model.training
    model.eval()
    devices = [device.index or 0] if device.type == "cuda" else []
    window = slice(step_start, step_end + 1)

    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            for offset in range(0, count, batch_size):
                batch_indices = indices[offset : offset + batch_size]
                inputs, targets, lengths, _, _ = (
                    value.to(device) for value in dataset.take(batch_indices)
                )
                io_grid = layout.render_tape(inputs)
                rollout = model(model.initial_state(io_grid), steps)
                values = layout.extract_tape(rollout[:, :, model.config.io_channel])
                expected = targets.unsqueeze(1)
                discrete = quantize(values)
                discrete_targets = targets.to(torch.int8)
                correct_symbols = discrete == discrete_targets.unsqueeze(1)
                raw = correct_symbols.all(dim=-1)
                semantic = semantic_correct(
                    discrete, discrete_targets, lengths, dataset.output_mode
                )
                mse_sums += (values - expected).square().sum(dim=(0, 2)).cpu()
                symbol_sums += correct_symbols.sum(dim=(0, 2)).cpu()
                raw_sums += raw.sum(dim=0).cpu()
                semantic_sums += semantic.sum(dim=0).cpu()
                stable_semantic_sum += float(semantic[:, window].all(dim=1).sum())
                stable_raw_sum += float(raw[:, window].all(dim=1).sum())
    finally:
        model.train(was_training)

    mse_curve = mse_sums / (count * geometry.tape_slots)
    semantic_curve = semantic_sums / count
    raw_curve = raw_sums / count
    symbol_curve = symbol_sums / (count * geometry.tape_slots)
    mse_window = mse_curve[window]
    semantic_window = semantic_curve[window]
    best_semantic_offset = int(semantic_window.argmax())
    best_mse_offset = int(mse_window.argmin())
    return EvaluationResult(
        task=dataset.name,
        examples=count,
        total_examples=len(dataset),
        tape_slots=geometry.tape_slots,
        batch_size=batch_size,
        seed=seed,
        step_start=step_start,
        step_end=step_end,
        mean_mse=float(mse_window.mean()),
        mean_semantic_accuracy=float(semantic_window.mean()),
        mean_raw_accuracy=float(raw_curve[window].mean()),
        mean_symbol_accuracy=float(symbol_curve[window].mean()),
        stable_semantic_accuracy=stable_semantic_sum / count,
        stable_raw_accuracy=stable_raw_sum / count,
        best_semantic_step=step_start + best_semantic_offset,
        best_semantic_accuracy=float(semantic_window[best_semantic_offset]),
        best_mse_step=step_start + best_mse_offset,
        best_mse=float(mse_window[best_mse_offset]),
        mse_by_step=tuple(map(float, mse_curve)),
        semantic_by_step=tuple(map(float, semantic_curve)),
        raw_by_step=tuple(map(float, raw_curve)),
        symbol_by_step=tuple(map(float, symbol_curve)),
    )


@dataclass(frozen=True)
class InferenceResult:
    input: str
    steps: int
    values: torch.Tensor
    interpreted: InterpretedTape


@torch.no_grad()
def infer(
    model: NeuralCellularAutomaton,
    geometry: GeometryConfig,
    input_symbols: str,
    *,
    steps: int,
    output_mode: str = "single",
) -> InferenceResult:
    if type(steps) is not int:
        raise TypeError("steps must be an integer")
    if steps < 0:
        raise ValueError("steps cannot be negative")
    if output_mode not in {"single", "multiple"}:
        raise ValueError("output_mode must be 'single' or 'multiple'")
    layout = TapeLayout(geometry)
    device = model.device
    encoded = encode_strings((input_symbols,), geometry.tape_slots).to(device)
    initial = model.initial_state(layout.render_tape(encoded))
    was_training = model.training
    model.eval()
    try:
        final = model(initial, steps)[:, -1, model.config.io_channel]
        values = layout.extract_tape(final)[0].cpu()
    finally:
        model.train(was_training)
    return InferenceResult(
        input=input_symbols,
        steps=steps,
        values=values,
        interpreted=interpret_tape(values, output_mode),
    )


def format_results(results: list[EvaluationResult]) -> str:
    header = (
        "task             n  tape   steps      mse    semantic      raw   "
        "symbol   stable"
    )
    lines = [header]
    for result in results:
        lines.append(
            f"{result.task[:16]:<16} {result.examples:>5} "
            f"{result.tape_slots:>5}  {result.step_start:>3}-{result.step_end:<3} "
            f"{result.mean_mse:>9.6f}  {result.mean_semantic_accuracy:>8.2%} "
            f"{result.mean_raw_accuracy:>8.2%} {result.mean_symbol_accuracy:>8.2%} "
            f"{result.stable_semantic_accuracy:>8.2%}"
        )
    return "\n".join(lines)

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import ExperimentConfig
from .model import NeuralCellularAutomaton
from .tape import TERNARY_THRESHOLD, TapeLayout, quantize
from .tasks import TaskDataset, addition_task
from .training import supervised_loss


@dataclass(frozen=True)
class ValidationReport:
    checks: tuple[str, ...]
    parameter_count: int
    grid_shape: tuple[int, int]
    examples: int

    def __str__(self) -> str:
        checks = ", ".join(self.checks)
        return (
            f"validated {checks}\n"
            f"parameters: {self.parameter_count:,}\n"
            f"grid: {self.grid_shape[0]} x {self.grid_shape[1]}\n"
            f"task examples: {self.examples:,}"
        )


def validate_experiment(
    config: ExperimentConfig,
    dataset: TaskDataset | None = None,
) -> ValidationReport:
    config.validate()
    if dataset is None:
        dataset = TaskDataset.from_task(addition_task(4), config.geometry.tape_slots)
    if dataset.tape_slots != config.geometry.tape_slots:
        raise ValueError("dataset and geometry have different tape capacities")
    layout = TapeLayout(config.geometry)
    checks = []

    sample = dataset.inputs[: min(3, len(dataset))]
    rendered = layout.render_tape(sample)
    if not torch.equal(layout.extract_tape(rendered), sample):
        raise AssertionError("tape render/extract round trip failed")
    occupied = torch.zeros(layout.height, layout.width, dtype=torch.bool)
    occupied[layout.tape_row, layout.tape_slice] = True
    if bool((rendered[:, ~occupied] != 0).any()):
        raise AssertionError("tape rendering wrote outside logical positions")
    checks.append("tape geometry")

    values = torch.tensor(
        [
            -TERNARY_THRESHOLD - 1e-6,
            -TERNARY_THRESHOLD,
            0.0,
            TERNARY_THRESHOLD,
            TERNARY_THRESHOLD + 1e-6,
        ]
    )
    if quantize(values).tolist() != [-1, 0, 0, 0, 1]:
        raise AssertionError("ternary thresholds are not strict at 0.333")
    checks.append("ternary codec")

    model = NeuralCellularAutomaton(config.model)
    initial = model.initial_state(rendered)
    if not torch.equal(model(initial, 1)[:, 1], initial):
        raise AssertionError("zero-initialized update rule is not identity")
    checks.append("identity initialization")

    probe = NeuralCellularAutomaton(config.model)
    with torch.no_grad():
        probe.rule.output.weight.fill_(0.05)
        if probe.rule.output.bias is not None:
            probe.rule.output.bias.fill_(0.05)
    state = probe.initial_state(rendered)
    updated = probe.step(state)
    channel = config.model.program_channel
    if not torch.equal(updated[:, channel], state[:, channel]):
        raise AssertionError("frozen program channel changed during an update")
    checks.append("zero frozen program")

    batch = dataset.take(torch.arange(min(2, len(dataset))))
    inputs, targets, _, terminator, tail = batch
    probe_targets = targets.clone()
    probe_targets[:, 0] = torch.where(inputs[:, 0] <= 0, 1.0, -1.0)
    train_model = NeuralCellularAutomaton(config.model)
    rollout = train_model(train_model.initial_state(layout.render_tape(inputs)), 1)
    losses = supervised_loss(
        rollout,
        probe_targets,
        layout,
        config.model.io_channel,
        free_steps=0,
        supervision_steps=1,
        terminator_mask=terminator,
        tail_mask=tail,
        terminator_weight=config.training.terminator_weight,
        tail_weight=config.training.tail_weight,
    )
    losses.total.backward()
    gradients = [
        parameter.grad
        for parameter in train_model.parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(
        torch.isfinite(gradient).all() for gradient in gradients
    ):
        raise AssertionError("finite gradients did not reach the local rule")
    if not any(bool((gradient != 0).any()) for gradient in gradients):
        raise AssertionError("loss produced no learning signal")
    checks.append("loss and gradients")

    return ValidationReport(
        checks=tuple(checks),
        parameter_count=model.parameter_count,
        grid_shape=(layout.height, layout.width),
        examples=len(dataset),
    )

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import ExperimentConfig
from .model import NeuralCellularAutomaton
from .tape import TERNARY_THRESHOLD, TapeLayout, quantize
from .tasks import MultiTaskDataset, binary_tasks
from .training import supervised_loss


DEFAULT_TASKS = ("copy", "bit_not", "reverse", "parity", "append_0", "append_1")


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
    datasets: MultiTaskDataset | None = None,
) -> ValidationReport:
    config.validate()
    if datasets is None:
        max_length = min(2, config.geometry.tape_slots - 1)
        if max_length < 1:
            raise ValueError("default programmed tasks require at least two tape slots")
        datasets = MultiTaskDataset.from_tasks(
            binary_tasks(DEFAULT_TASKS, max_length), config.geometry.tape_slots
        )
    if datasets.tape_slots != config.geometry.tape_slots:
        raise ValueError("datasets and geometry have different tape capacities")
    layout = TapeLayout(config.geometry)
    checks = []

    sample_count = min(3, len(datasets.datasets[0]), len(datasets.datasets))
    sample = datasets.datasets[0].inputs[:sample_count]
    rendered = layout.render_tape(sample)
    if not torch.equal(layout.extract_tape(rendered), sample):
        raise AssertionError("tape render/extract round trip failed")
    occupied = torch.zeros(layout.height, layout.width, dtype=torch.bool)
    occupied[layout.tape_row, layout.tape_slice] = True
    if bool((rendered[:, ~occupied] != 0).any()):
        raise AssertionError("tape rendering wrote outside logical positions")
    checks.append("strided tape geometry")

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
    checks.append("direct ternary codec")

    torch.manual_seed(config.training.seed)
    model = NeuralCellularAutomaton(config.model, config.geometry, datasets.task_names)
    task_indices = torch.arange(sample_count)
    initial = model.initial_state(rendered, task_indices)
    expected_programs = model.program_grid(task_indices)
    if not torch.equal(initial[:, : config.model.program_channels], expected_programs):
        raise AssertionError("selected programs were not injected")
    if config.model.program_placement == "tape" and bool(
        (expected_programs[:, :, ~occupied] != 0).any()
    ):
        raise AssertionError("tape programs wrote outside logical positions")
    if not torch.equal(initial[:, config.model.input_channel], rendered):
        raise AssertionError("input tape was not injected directly")
    if config.model.io_mode == "separate" and torch.count_nonzero(
        initial[:, config.model.output_channel]
    ):
        raise AssertionError("separate output channel is not zero initially")
    if not torch.equal(model(initial, 1)[:, 1], initial):
        raise AssertionError("zero-initialized update rule is not identity")
    checks.append("task-indexed program and I/O initialization")

    probe = NeuralCellularAutomaton(config.model, config.geometry, datasets.task_names)
    with torch.no_grad():
        probe.rule.output.weight.fill_(0.05)
        if probe.rule.output.bias is not None:
            probe.rule.output.bias.fill_(0.05)
    state = probe.initial_state(rendered, task_indices)
    updated = probe.step(state)
    program = slice(0, config.model.program_channels)
    if not config.model.program_mutable and not torch.equal(
        updated[:, program], state[:, program]
    ):
        raise AssertionError("read-only program changed during an update")
    if config.model.io_mode == "separate" and config.model.input_mode == "frozen":
        if not torch.equal(
            updated[:, config.model.input_channel], state[:, config.model.input_channel]
        ):
            raise AssertionError("frozen input changed during an update")
    if float(probe.update_mask[0, config.model.output_channel, 0, 0]) != 1.0:
        raise AssertionError("readout channel is not mutable")
    checks.append("channel mutability")

    train_model = NeuralCellularAutomaton(
        config.model, config.geometry, datasets.task_names
    )
    with torch.no_grad():
        train_model.rule.hidden.weight.fill_(0.1)
        train_model.rule.hidden.bias.fill_(0.1)
        train_model.rule.output.weight.fill_(0.01)
        if train_model.rule.output.bias is not None:
            train_model.rule.output.bias.zero_()
    inputs = torch.stack(
        [dataset.inputs[min(2, len(dataset) - 1)] for dataset in datasets.datasets]
    )
    targets = torch.stack(
        [dataset.targets[min(2, len(dataset) - 1)] for dataset in datasets.datasets]
    )
    all_tasks = torch.arange(len(datasets.datasets))
    rollout = train_model(
        train_model.initial_state(layout.render_tape(inputs), all_tasks), 1
    )
    loss = supervised_loss(
        rollout,
        targets,
        layout,
        config.model.output_channel,
        free_steps=0,
        supervision_steps=1,
        task_indices=all_tasks,
        task_count=len(datasets.datasets),
    )
    loss.total.backward()
    if train_model.rule.output.weight.grad is None or not bool(
        (train_model.rule.output.weight.grad != 0).any()
    ):
        raise AssertionError("loss produced no learning signal for the shared rule")
    program_gradient = train_model.programs.grad
    if program_gradient is None or not bool(
        (program_gradient.flatten(1).abs().sum(dim=1) > 0).all()
    ):
        raise AssertionError("loss did not reach every task program")
    checks.append("balanced full-tape MSE and gradients")

    return ValidationReport(
        checks=tuple(checks),
        parameter_count=model.parameter_count,
        grid_shape=(layout.height, layout.width),
        examples=len(datasets),
    )

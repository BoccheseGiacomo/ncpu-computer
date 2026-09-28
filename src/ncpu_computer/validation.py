from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import ExperimentConfig
from .model import NeuralCellularAutomaton
from .tape import TERNARY_THRESHOLD, TapeLayout, quantize
from .tasks import MultiTaskDataset, binary_tasks
from .training import TrialSpec, sample_trials, supervised_loss


DEFAULT_TASKS = ("copy", "bit_not", "reverse", "parity", "append_0", "append_1")


@dataclass(frozen=True)
class ValidationReport:
    checks: tuple[str, ...]
    parameter_count: int
    sample_trials: tuple[TrialSpec, ...]
    task_names: tuple[str, ...]

    def __str__(self) -> str:
        trials = ", ".join(
            f"{trial.tape_slots} slots / L={trial.input_max_length} / "
            f"{trial.free_steps}+{trial.supervision_steps} steps"
            for trial in self.sample_trials
        )
        return (
            f"validated {', '.join(self.checks)}\n"
            f"parameters: {self.parameter_count:,}\n"
            f"tasks: {', '.join(self.task_names)}\n"
            f"sample trials: {trials}"
        )


def validate_experiment(
    config: ExperimentConfig,
    task_names: tuple[str, ...] = DEFAULT_TASKS,
) -> ValidationReport:
    config.validate()
    binary_tasks(task_names, 0)
    trials = sample_trials(
        config.training, torch.Generator().manual_seed(config.training.seed)
    )
    if len({trial.tape_slots for trial in trials}) != len(trials):
        raise AssertionError("sampled tape capacities are not unique")
    if len({trial.input_max_length for trial in trials}) != len(trials):
        raise AssertionError("sampled input lengths are not unique")
    if len({trial.free_steps for trial in trials}) != len(trials):
        raise AssertionError("sampled computation times are not unique")
    checks = ["stratified variable trials"]

    trial = trials[0]
    layout = TapeLayout(config.geometry, trial.tape_slots)
    datasets = MultiTaskDataset.from_tasks(
        binary_tasks(task_names, min(2, trial.input_max_length)), trial.tape_slots
    )
    sample_count = min(3, len(datasets.datasets[0]), len(task_names))
    sample = datasets.datasets[0].inputs[:sample_count]
    rendered = layout.render_tape(sample)
    if not torch.equal(layout.extract_tape(rendered), sample):
        raise AssertionError("tape render/extract round trip failed")
    occupied = torch.zeros(layout.height, layout.width, dtype=torch.bool)
    occupied[layout.tape_row, layout.tape_slice] = True
    if bool((rendered[:, ~occupied] != 0).any()):
        raise AssertionError("tape rendering wrote outside logical positions")
    checks.append("shared strided tape")

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
    model = NeuralCellularAutomaton(config.model, config.geometry, task_names)
    task_indices = torch.arange(sample_count)
    initial = model.initial_state(rendered, task_indices)
    programs = model.program_grid(task_indices, layout.width)
    if not torch.equal(initial[:, : config.model.program_channels], programs):
        raise AssertionError("task programs were not injected")
    tile = model.programs[task_indices]
    for offset in range(0, layout.width, config.geometry.stride):
        if not torch.equal(
            programs[..., offset : offset + config.geometry.stride], tile
        ):
            raise AssertionError("program tile does not repeat from absolute x=0")
    if config.model.program_mode == "zero" and torch.count_nonzero(programs):
        raise AssertionError("zero program is not zero")
    if not torch.equal(initial[:, config.model.io_channel], rendered):
        raise AssertionError("input was not injected into the shared I/O channel")
    if not torch.equal(model(initial, 1)[:, 1], initial):
        raise AssertionError("zero-initialized update rule is not identity")
    checks.append("periodic task program")

    probe = NeuralCellularAutomaton(config.model, config.geometry, task_names)
    with torch.no_grad():
        probe.rule.hidden.weight.fill_(0.1)
        probe.rule.hidden.bias.fill_(0.1)
        probe.rule.output.weight.fill_(0.05)
        if probe.rule.output.bias is not None:
            probe.rule.output.bias.fill_(0.05)
    state = probe.initial_state(rendered, task_indices)
    updated = probe.step(state)
    program_slice = slice(0, config.model.program_channels)
    if config.model.program_mutable:
        if torch.equal(updated[:, program_slice], state[:, program_slice]):
            raise AssertionError("mutable program did not update")
    elif not torch.equal(updated[:, program_slice], state[:, program_slice]):
        raise AssertionError("read-only program changed")
    if torch.equal(
        updated[:, config.model.io_channel], state[:, config.model.io_channel]
    ):
        raise AssertionError("shared I/O channel is not mutable")
    checks.append("channel mutability")

    targets = torch.stack(
        [
            dataset.targets[min(2, len(dataset) - 1)]
            for dataset in datasets.datasets[:sample_count]
        ]
    )
    rollout = probe(state, 1)
    loss = supervised_loss(
        rollout,
        targets,
        layout,
        config.model.io_channel,
        free_steps=0,
        supervision_steps=1,
        task_indices=task_indices,
        task_count=sample_count,
    )
    loss.total.backward()
    if probe.rule.output.weight.grad is None or not bool(
        (probe.rule.output.weight.grad != 0).any()
    ):
        raise AssertionError("loss produced no learning signal for the rule")
    if config.model.program_mode != "zero":
        gradient = probe.programs.grad
        if gradient is None or not bool((gradient[task_indices] != 0).any()):
            raise AssertionError("loss did not reach the learned program")
    checks.append("full-tape MSE gradients")

    return ValidationReport(tuple(checks), model.parameter_count, trials, task_names)

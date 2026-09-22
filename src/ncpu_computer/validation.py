from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import ExperimentConfig
from .model import NeuralCellularAutomaton
from .tape import TERNARY_THRESHOLD, TapeLayout, quantize
from .tasks import TaskDataset, bitwise_not_task
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
        dataset = TaskDataset.from_task(
            bitwise_not_task(config.geometry.tape_slots),
            config.geometry.tape_slots,
        )
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
    model = NeuralCellularAutomaton(config.model)
    initial = model.initial_state(rendered)
    if torch.count_nonzero(initial[:, config.model.output_channel]) != 0:
        raise AssertionError("output channel is not zero at initialization")
    if not torch.equal(initial[:, config.model.input_channel], rendered):
        raise AssertionError("input tape was not injected directly")
    if not torch.equal(model(initial, 1)[:, 1], initial):
        raise AssertionError("zero-initialized update rule is not identity")
    checks.append("direct input and zero output initialization")

    probe = NeuralCellularAutomaton(config.model)
    with torch.no_grad():
        probe.rule.output.weight.fill_(0.05)
        if probe.rule.output.bias is not None:
            probe.rule.output.bias.fill_(0.05)
    state = probe.initial_state(rendered)
    updated = probe.step(state)
    program = slice(0, config.model.program_channels)
    if not torch.equal(updated[:, program], state[:, program]):
        raise AssertionError("frozen program channels changed during an update")
    if config.model.input_mode == "frozen" and not torch.equal(
        updated[:, config.model.input_channel], state[:, config.model.input_channel]
    ):
        raise AssertionError("frozen input channel changed during an update")
    if float(probe.update_mask[0, config.model.output_channel, 0, 0]) != 1.0:
        raise AssertionError("output channel is not mutable")
    checks.append("channel mutability")

    inputs = torch.zeros(1, config.geometry.tape_slots)
    inputs[0, 0] = 1.0
    targets = torch.zeros_like(inputs)
    targets[0, 0] = -1.0
    train_model = NeuralCellularAutomaton(config.model)
    with torch.no_grad():
        train_model.rule.hidden.weight.fill_(0.1)
        train_model.rule.hidden.bias.fill_(0.1)
    rollout = train_model(train_model.initial_state(layout.render_tape(inputs)), 1)
    loss = supervised_loss(
        rollout,
        targets,
        layout,
        config.model.output_channel,
        free_steps=0,
        supervision_steps=1,
    )
    loss.total.backward()
    gradients = [
        parameter.grad
        for parameter in train_model.parameters()
        if parameter.grad is not None
    ]
    if not gradients or not all(
        torch.isfinite(gradient).all() for gradient in gradients
    ):
        raise AssertionError("finite gradients did not reach the model")
    rule_gradient = train_model.rule.output.weight.grad
    if rule_gradient is None or not bool((rule_gradient != 0).any()):
        raise AssertionError("loss produced no learning signal for the local rule")
    checks.append("full-tape MSE and gradients")

    return ValidationReport(
        checks=tuple(checks),
        parameter_count=model.parameter_count,
        grid_shape=(layout.height, layout.width),
        examples=len(dataset),
    )

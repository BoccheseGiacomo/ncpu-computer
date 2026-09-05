from dataclasses import replace

import pytest
import torch

from ncpu_computer.config import (
    ExperimentConfig,
    GeometryConfig,
    ModelConfig,
    TrainingConfig,
)
from ncpu_computer.evaluation import evaluate, infer
from ncpu_computer.model import NeuralCellularAutomaton
from ncpu_computer.tape import TapeLayout
from ncpu_computer.tasks import (
    StringExample,
    StringTask,
    TaskDataset,
    reverse_task,
)
from ncpu_computer.training import Trainer, load_model, supervised_loss
from ncpu_computer.validation import validate_experiment


def tiny_setup(*, updates=2, fire_rate=1.0):
    geometry = GeometryConfig(
        tape_slots=4,
        stride=2,
        border_left=1,
        border_right=1,
        border_top=1,
        border_bottom=1,
    )
    config = ExperimentConfig(
        geometry=geometry,
        model=ModelConfig(
            hidden_size=4,
            fixed_kernels=("identity",),
            learnable_kernels=0,
            fire_rate=fire_rate,
            max_abs_state=None,
        ),
        training=TrainingConfig(
            updates=updates,
            batch_size=2,
            free_steps=0,
            supervision_steps=1,
            validation_every=1,
            checkpoint_every=1,
            device="cpu",
        ),
    )
    task = StringTask(
        "binary-not",
        (StringExample("0", "1"), StringExample("1", "0")),
    )
    dataset = TaskDataset.from_task(task, geometry.tape_slots)
    return config, dataset


def test_supervised_loss_uses_only_requested_tape_window():
    geometry = GeometryConfig(
        tape_slots=4,
        stride=2,
        border_left=1,
        border_right=1,
        border_top=1,
        border_bottom=1,
    )
    layout = TapeLayout(geometry)
    rollout = torch.zeros(1, 4, 2, layout.height, layout.width)
    predicted = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rollout[:, 2:4, 1, layout.tape_row, layout.tape_slice] = predicted
    rollout[:, 2:4, 1, 0, 0] = 1000.0
    target = torch.zeros(1, 4)
    terminator = torch.tensor([[False, True, False, False]])
    tail = torch.tensor([[False, False, True, True]])
    losses = supervised_loss(
        rollout,
        target,
        layout,
        io_channel=1,
        free_steps=1,
        supervision_steps=2,
        terminator_mask=terminator,
        tail_mask=tail,
        terminator_weight=0.5,
        tail_weight=0.2,
    )
    assert float(losses.base) == pytest.approx(7.5)
    assert float(losses.terminator) == pytest.approx(4.0)
    assert float(losses.tail) == pytest.approx(12.5)
    assert float(losses.total) == pytest.approx(12.0)


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    config, dataset = tiny_setup(fire_rate=0.5)
    uninterrupted = Trainer(config, dataset)
    uninterrupted.train_step()
    uninterrupted.train_step()

    interrupted = Trainer(config, dataset)
    interrupted.train_step()
    checkpoint = tmp_path / "resume.pt"
    interrupted.save(checkpoint)
    resumed = Trainer.from_checkpoint(checkpoint, dataset, device="cpu")
    resumed.train_step()

    assert resumed.current_update == uninterrupted.current_update == 2
    for expected, actual in zip(
        uninterrupted.model.parameters(), resumed.model.parameters()
    ):
        assert torch.equal(expected, actual)


def test_training_step_changes_the_zero_initialized_rule():
    config, dataset = tiny_setup()
    trainer = Trainer(config, dataset)
    before = trainer.model.rule.output.weight.detach().clone()
    metrics = trainer.train_step()
    assert metrics.loss > 0
    assert not torch.equal(trainer.model.rule.output.weight, before)


def test_exhaustive_validation_loss_is_independent_of_batch_partition():
    config, dataset = tiny_setup()
    weighted = replace(
        config.training,
        batch_size=1,
        terminator_weight=0.7,
        tail_weight=0.4,
    )
    first = Trainer(replace(config, training=weighted), dataset)
    second = Trainer(
        replace(config, training=replace(weighted, batch_size=len(dataset))),
        dataset,
    )
    second.model.load_state_dict(first.model.state_dict())
    assert first.validation_loss() == pytest.approx(second.validation_loss())
    assert first.model.training


def test_short_fit_writes_loadable_best_and_latest_checkpoints(tmp_path):
    config, dataset = tiny_setup(updates=1)
    trainer = Trainer(config, dataset)
    history = trainer.fit(tmp_path, progress_every=1)
    assert len(history) == 1
    assert (tmp_path / "best.pt").is_file()
    assert (tmp_path / "latest.pt").is_file()
    model, loaded_config, checkpoint = load_model(tmp_path / "best.pt", "cpu")
    assert loaded_config == config
    assert checkpoint["dataset_signature"] == dataset.signature
    assert model.training is False

    other_data = TaskDataset.from_task(reverse_task(2), config.geometry.tape_slots)
    with pytest.raises(ValueError, match="dataset"):
        Trainer.from_checkpoint(tmp_path / "best.pt", other_data, device="cpu")


def test_identity_model_evaluates_copy_task_exactly():
    config, dataset = tiny_setup()
    dataset = TaskDataset.from_task(reverse_task(1), config.geometry.tape_slots)
    model = NeuralCellularAutomaton(config.model)
    result = evaluate(
        model,
        config.geometry,
        dataset,
        steps=2,
        step_start=0,
        step_end=2,
        batch_size=1,
    )
    assert result.mean_mse == 0.0
    assert result.mean_semantic_accuracy == 1.0
    assert result.mean_raw_accuracy == 1.0
    assert result.stable_semantic_accuracy == 1.0


def test_inference_reads_full_tape_then_applies_multiple_decoder():
    geometry = GeometryConfig(tape_slots=9)
    model = NeuralCellularAutomaton(ModelConfig())
    result = infer(
        model,
        geometry,
        "101B11BB0",
        steps=0,
        output_mode="multiple",
    )
    assert result.interpreted.raw == "101B11BB0"
    assert result.interpreted.binary_strings == ("101", "11")
    assert result.interpreted.integers == (5, 3)


def test_validation_covers_core_invariants():
    config, dataset = tiny_setup()
    report = validate_experiment(config, dataset)
    assert "tape geometry" in report.checks
    assert "loss and gradients" in report.checks
    assert report.examples == len(dataset)


def test_validation_does_not_require_task_outputs_to_differ_from_inputs():
    config, _ = tiny_setup()
    copy_data = TaskDataset.from_task(reverse_task(1), config.geometry.tape_slots)
    assert validate_experiment(config, copy_data).examples == len(copy_data)

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
from ncpu_computer.tasks import StringExample, StringTask, TaskDataset, reverse_task
from ncpu_computer.training import Trainer, load_model, supervised_loss
from ncpu_computer.validation import validate_experiment
from ncpu_computer.tape import TapeLayout


def tiny_setup(*, updates=2, fire_rate=1.0, input_mode="mutable"):
    geometry = GeometryConfig(
        tape_slots=3,
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
            input_mode=input_mode,
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
        (StringExample("", ""), StringExample("0", "1"), StringExample("1", "0")),
    )
    return config, TaskDataset.from_task(task, geometry.tape_slots)


def test_supervised_loss_uses_only_output_tape_and_requested_window():
    geometry = GeometryConfig(
        tape_slots=4,
        stride=2,
        border_left=1,
        border_right=1,
        border_top=1,
        border_bottom=1,
    )
    layout = TapeLayout(geometry)
    rollout = torch.zeros(1, 4, 4, layout.height, layout.width)
    predicted = torch.tensor([1.0, 2.0, 3.0, 4.0])
    rollout[:, 2:4, 2, layout.tape_row, layout.tape_slice] = predicted
    rollout[:, 2:4, 2, 0, 0] = 1000.0
    rollout[:, 2:4, 1, layout.tape_row, layout.tape_slice] = 1000.0
    losses = supervised_loss(
        rollout,
        torch.zeros(1, 4),
        layout,
        output_channel=2,
        free_steps=1,
        supervision_steps=2,
    )
    assert float(losses.base) == pytest.approx(7.5)
    assert float(losses.total) == pytest.approx(7.5)


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


def test_training_step_changes_zero_initialized_rule():
    config, dataset = tiny_setup()
    trainer = Trainer(config, dataset)
    before = trainer.model.rule.output.weight.detach().clone()
    metrics = trainer.train_step()
    assert metrics.loss > 0
    assert not torch.equal(trainer.model.rule.output.weight, before)


def test_exhaustive_validation_loss_is_independent_of_batch_partition():
    config, dataset = tiny_setup()
    first = Trainer(
        replace(config, training=replace(config.training, batch_size=1)), dataset
    )
    second = Trainer(
        replace(config, training=replace(config.training, batch_size=len(dataset))),
        dataset,
    )
    second.model.load_state_dict(first.model.state_dict())
    assert first.validation_loss() == pytest.approx(second.validation_loss())
    assert first.model.training


def test_short_fit_writes_loadable_version_three_checkpoints(tmp_path):
    config, dataset = tiny_setup(updates=1)
    trainer = Trainer(config, dataset)
    history = trainer.fit(tmp_path, progress_every=1)
    assert len(history) == 1
    model, loaded_config, checkpoint = load_model(tmp_path / "best.pt", "cpu")
    assert loaded_config == config
    assert checkpoint["format_version"] == 3
    assert checkpoint["dataset_signature"] == dataset.signature
    assert not model.training
    other_data = TaskDataset.from_task(reverse_task(2), config.geometry.tape_slots)
    with pytest.raises(ValueError, match="dataset"):
        Trainer.from_checkpoint(tmp_path / "best.pt", other_data, device="cpu")


def test_zero_output_model_evaluates_empty_target_exactly():
    config, _ = tiny_setup()
    dataset = TaskDataset.from_task(
        StringTask("empty", (StringExample("101", ""),)),
        config.geometry.tape_slots,
    )
    model = Trainer(config, dataset).model
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


def test_inference_reads_the_separate_zero_output_channel():
    config, _ = tiny_setup()
    model = Trainer(config, TaskDataset.from_task(reverse_task(1), 3)).model
    result = infer(model, config.geometry, "101", steps=0)
    assert result.values.tolist() == [0.0, 0.0, 0.0]
    assert result.interpreted.raw == "BBB"
    assert result.interpreted.binary_strings == ("",)


def test_validation_covers_core_invariants_for_both_input_modes():
    for mode in ("mutable", "frozen"):
        config, dataset = tiny_setup(input_mode=mode)
        report = validate_experiment(config, dataset)
        assert "direct input and zero output initialization" in report.checks
        assert "full-tape MSE and gradients" in report.checks
        assert report.examples == len(dataset)

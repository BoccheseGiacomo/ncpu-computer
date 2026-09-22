from dataclasses import replace

import pytest
import torch

from ncpu_computer.config import (
    ExperimentConfig,
    GeometryConfig,
    ModelConfig,
    TrainingConfig,
)
from ncpu_computer.evaluation import evaluate, evaluate_tasks, infer
from ncpu_computer.tasks import (
    MultiTaskDataset,
    StringExample,
    StringTask,
    TaskDataset,
    binary_tasks,
)
from ncpu_computer.training import Trainer, load_model, supervised_loss
from ncpu_computer.validation import validate_experiment
from ncpu_computer.tape import TapeLayout


def tiny_setup(
    *,
    updates=2,
    fire_rate=1.0,
    input_mode="mutable",
    io_mode="separate",
    program_placement="grid",
    program_mutable=False,
):
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
            io_mode=io_mode,
            program_placement=program_placement,
            program_mutable=program_mutable,
            max_abs_state=None,
        ),
        training=TrainingConfig(
            updates=updates,
            batch_size_per_task=2,
            free_steps=0,
            supervision_steps=1,
            validation_every=1,
            checkpoint_every=1,
            device="cpu",
        ),
    )
    datasets = MultiTaskDataset.from_tasks(
        binary_tasks(("copy", "bit_not"), 1), geometry.tape_slots
    )
    return config, datasets


def test_supervised_loss_uses_full_output_tape_and_equal_task_means():
    geometry = GeometryConfig(
        tape_slots=2,
        stride=2,
        border_left=1,
        border_right=1,
        border_top=1,
        border_bottom=1,
    )
    layout = TapeLayout(geometry)
    rollout = torch.zeros(3, 2, 4, layout.height, layout.width)
    rollout[0, 1, 2, layout.tape_row, layout.tape_slice] = 1.0
    rollout[1:, 1, 2, layout.tape_row, layout.tape_slice] = 3.0
    rollout[:, 1, 2, 0, 0] = 1000.0
    rollout[:, 1, 1, layout.tape_row, layout.tape_slice] = 1000.0
    losses = supervised_loss(
        rollout,
        torch.zeros(3, 2),
        layout,
        output_channel=2,
        free_steps=0,
        supervision_steps=1,
        task_indices=torch.tensor([0, 1, 1]),
        task_count=2,
    )
    assert losses.per_task.tolist() == pytest.approx([1.0, 9.0])
    assert float(losses.base) == pytest.approx(5.0)


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    config, datasets = tiny_setup(fire_rate=0.5)
    uninterrupted = Trainer(config, datasets)
    uninterrupted.train_step()
    uninterrupted.train_step()
    interrupted = Trainer(config, datasets)
    interrupted.train_step()
    checkpoint = tmp_path / "resume.pt"
    interrupted.save(checkpoint)
    resumed = Trainer.from_checkpoint(checkpoint, datasets, device="cpu")
    resumed.train_step()
    assert resumed.current_update == uninterrupted.current_update == 2
    for expected, actual in zip(
        uninterrupted.model.parameters(), resumed.model.parameters()
    ):
        assert torch.equal(expected, actual)


def test_training_step_is_balanced_and_optimizer_separates_program_decay():
    config, datasets = tiny_setup()
    trainer = Trainer(config, datasets)
    assert trainer.optimizer.param_groups[0]["weight_decay"] == pytest.approx(
        config.training.weight_decay
    )
    assert trainer.optimizer.param_groups[1]["weight_decay"] == pytest.approx(
        config.training.program_weight_decay
    )
    assert trainer.optimizer.param_groups[1]["params"] == [trainer.model.programs]
    before = trainer.model.rule.output.weight.detach().clone()
    metrics = trainer.train_step()
    assert metrics.loss > 0
    assert not torch.equal(trainer.model.rule.output.weight, before)


def test_exhaustive_validation_loss_is_independent_of_batch_partition():
    config, datasets = tiny_setup()
    first = Trainer(
        replace(config, training=replace(config.training, batch_size_per_task=1)),
        datasets,
    )
    second = Trainer(
        replace(config, training=replace(config.training, batch_size_per_task=3)),
        datasets,
    )
    second.model.load_state_dict(first.model.state_dict())
    assert first.validation_loss() == pytest.approx(second.validation_loss())
    assert first.model.training


def test_short_fit_writes_loadable_version_four_checkpoint(tmp_path):
    config, datasets = tiny_setup(updates=1)
    trainer = Trainer(config, datasets)
    history = trainer.fit(tmp_path, progress_every=1)
    assert len(history) == 1
    model, loaded_config, checkpoint = load_model(tmp_path / "best.pt", "cpu", datasets)
    assert loaded_config == config
    assert checkpoint["format_version"] == 4
    assert tuple(checkpoint["task_names"]) == datasets.task_names
    assert tuple(checkpoint["dataset_signatures"]) == datasets.signatures
    assert not model.training
    reversed_order = MultiTaskDataset(tuple(reversed(datasets.datasets)))
    with pytest.raises(ValueError, match="task order"):
        Trainer.from_checkpoint(tmp_path / "best.pt", reversed_order, device="cpu")
    different_examples = MultiTaskDataset.from_tasks(
        binary_tasks(datasets.task_names, 2), config.geometry.tape_slots
    )
    with pytest.raises(ValueError, match="datasets"):
        Trainer.from_checkpoint(tmp_path / "best.pt", different_examples, device="cpu")
    incompatible = dict(checkpoint)
    incompatible["format_version"] = 3
    incompatible_path = tmp_path / "old.pt"
    torch.save(incompatible, incompatible_path)
    with pytest.raises(ValueError, match="incompatible"):
        load_model(incompatible_path, "cpu")


def test_zero_output_model_evaluates_empty_target_exactly():
    config, _ = tiny_setup()
    dataset = TaskDataset.from_task(
        StringTask("empty", (StringExample("101", ""),)),
        config.geometry.tape_slots,
    )
    datasets = MultiTaskDataset((dataset,))
    model = Trainer(config, datasets).model
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


def test_inference_requires_explicit_task_and_reads_configured_channel():
    config, datasets = tiny_setup()
    model = Trainer(config, datasets).model
    result = infer(model, config.geometry, "101", task_name="copy", steps=0)
    assert result.task == "copy"
    assert result.values.tolist() == [0.0, 0.0, 0.0]
    assert result.interpreted.raw == "BBB"
    shared_config, shared_datasets = tiny_setup(io_mode="shared")
    shared = Trainer(shared_config, shared_datasets).model
    shared_result = infer(
        shared, shared_config.geometry, "101", task_name="copy", steps=0
    )
    assert shared_result.values.tolist() == [1.0, -1.0, 1.0]


def test_evaluate_tasks_returns_each_task_and_equal_aggregate():
    config, datasets = tiny_setup()
    model = Trainer(config, datasets).model
    results = evaluate_tasks(
        model,
        config.geometry,
        datasets,
        steps=1,
        step_start=0,
        step_end=1,
        batch_size=2,
    )
    assert [result.task for result in results] == ["copy", "bit_not", "aggregate"]
    assert results[-1].mean_mse == pytest.approx(
        sum(result.mean_mse for result in results[:-1]) / 2
    )
    with pytest.raises(ValueError, match="fixed model geometry"):
        evaluate(
            model,
            replace(config.geometry, tape_slots=4),
            datasets.datasets[0],
            steps=1,
            step_start=0,
            step_end=1,
        )


@pytest.mark.parametrize("program_placement", ["grid", "tape"])
@pytest.mark.parametrize(
    "io_mode,input_mode",
    [("separate", "mutable"), ("separate", "frozen"), ("shared", "mutable")],
)
def test_validation_covers_program_and_io_modes(program_placement, io_mode, input_mode):
    config, datasets = tiny_setup(
        input_mode=input_mode,
        io_mode=io_mode,
        program_placement=program_placement,
    )
    report = validate_experiment(config, datasets)
    assert "task-indexed program and I/O initialization" in report.checks
    assert "balanced full-tape MSE and gradients" in report.checks
    assert report.examples == len(datasets)

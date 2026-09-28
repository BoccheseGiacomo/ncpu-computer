from dataclasses import replace

import pytest
import torch

from ncpu_computer import (
    ExperimentConfig,
    GeometryConfig,
    ModelConfig,
    TapeLayout,
    TestCase,
    Trainer,
    TrainingConfig,
    evaluate_cases,
    infer,
    load_model,
    perception_noise,
    sample_trials,
    supervised_loss,
)


def tiny_config(
    *, updates=2, mode="zero", train_rule=True, train_program=False, fire_rate=1.0
):
    return ExperimentConfig(
        geometry=GeometryConfig(stride=2, vertical_space=1, horizontal_space=2),
        model=ModelConfig(
            hidden_size=4,
            fixed_kernels=("identity",),
            learnable_kernels=0,
            program_mode=mode,
            fire_rate=fire_rate,
            max_abs_state=None,
        ),
        training=TrainingConfig(
            updates=updates,
            batch_size_per_task=2,
            n_trials=2,
            tape_slots_min=3,
            tape_slots_max=4,
            input_max_length_min=1,
            input_max_length_max=2,
            free_steps_min=1,
            free_steps_max=2,
            supervision_ratio=1.0,
            train_rule=train_rule,
            train_program=train_program,
            validation_every=1,
            checkpoint_every=1,
            device="cpu",
        ),
        test_cases=(TestCase("tiny", 3, 1, 1, 1),),
    )


def test_stratified_trials_are_unique_coupled_and_reproducible():
    config = tiny_config().training
    first = sample_trials(config, torch.Generator().manual_seed(9))
    second = sample_trials(config, torch.Generator().manual_seed(9))
    assert first == second
    assert len({trial.tape_slots for trial in first}) == 2
    assert len({trial.input_max_length for trial in first}) == 2
    assert len({trial.free_steps for trial in first}) == 2
    assert all(trial.input_max_length <= trial.tape_slots - 1 for trial in first)
    ranked = sorted(first, key=lambda trial: trial.tape_slots)
    assert ranked[0].input_max_length < ranked[1].input_max_length
    assert ranked[0].free_steps < ranked[1].free_steps


def test_supervised_loss_uses_every_tape_cell_and_equal_task_means():
    layout = TapeLayout(GeometryConfig(), 2)
    rollout = torch.zeros(3, 2, 3, layout.height, layout.width)
    rollout[0, 1, 1, layout.tape_row, layout.tape_slice] = 1.0
    rollout[1:, 1, 1, layout.tape_row, layout.tape_slice] = 3.0
    rollout[:, 1, 1, 0, 0] = 1000.0
    losses = supervised_loss(
        rollout,
        torch.zeros(3, 2),
        layout,
        1,
        0,
        1,
        torch.tensor([0, 1, 1]),
        2,
    )
    assert losses.per_task.tolist() == pytest.approx([1.0, 9.0])
    assert float(losses.total) == pytest.approx(5.0)


def test_training_accumulates_trials_and_optimizer_respects_modes():
    rule_only = Trainer(tiny_config(), ("copy", "bit_not"))
    metrics = rule_only.train_step()
    assert len(metrics.trials) == 2
    assert [group["name"] for group in rule_only.optimizer.param_groups] == ["rule"]
    assert all(
        not parameter.requires_grad for parameter in rule_only.model.program_parameters
    )

    program_config = tiny_config(
        mode="learned_read_only", train_rule=False, train_program=True
    )
    program_only = Trainer(program_config, ("copy", "bit_not"))
    with torch.no_grad():
        program_only.model.rule.hidden.weight.fill_(0.1)
        program_only.model.rule.hidden.bias.fill_(0.1)
        program_only.model.rule.output.weight.fill_(0.05)
    before = program_only.model.programs.detach().clone()
    program_only.train_step()
    assert [group["name"] for group in program_only.optimizer.param_groups] == [
        "program"
    ]
    assert not torch.equal(before, program_only.model.programs)

    joint = Trainer(
        tiny_config(mode="learned_mutable", train_rule=True, train_program=True),
        ("copy", "bit_not"),
    )
    assert [group["name"] for group in joint.optimizer.param_groups] == [
        "rule",
        "program",
    ]
    assert joint.optimizer.param_groups[0]["weight_decay"] == pytest.approx(
        joint.config.training.weight_decay
    )
    assert joint.optimizer.param_groups[1]["weight_decay"] == pytest.approx(
        joint.config.training.program_weight_decay
    )


def test_noise_anneals_linearly():
    training = replace(
        tiny_config(updates=3).training,
        perception_noise_start=0.1,
        perception_noise_end=0.0,
    )
    assert [perception_noise(training, i) for i in range(3)] == pytest.approx(
        [0.1, 0.05, 0.0]
    )


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path):
    config = replace(
        tiny_config(fire_rate=0.5),
        training=replace(
            tiny_config(fire_rate=0.5).training,
            perception_noise_start=0.02,
            perception_noise_end=0.01,
        ),
    )
    names = ("copy", "bit_not")
    uninterrupted = Trainer(config, names)
    uninterrupted.train_step()
    expected_second = uninterrupted.train_step()
    interrupted = Trainer(config, names)
    interrupted.train_step()
    path = tmp_path / "resume.pt"
    interrupted.save(path)
    resumed = Trainer.from_checkpoint(path, names, device="cpu")
    actual_second = resumed.train_step()
    assert actual_second.trials == expected_second.trials
    assert actual_second.loss == pytest.approx(expected_second.loss)
    for expected, actual in zip(
        uninterrupted.model.parameters(), resumed.model.parameters()
    ):
        assert torch.equal(expected, actual)


def test_checkpoint_and_variable_case_evaluation(tmp_path):
    config = tiny_config(updates=1)
    trainer = Trainer(config, ("copy", "bit_not"))
    trainer.fit(tmp_path, progress_every=1)
    model, loaded, checkpoint = load_model(tmp_path / "best.pt", "cpu")
    assert loaded == config
    assert checkpoint["format_version"] == 5
    assert len(checkpoint["task_signatures"]) == 2
    results = evaluate_cases(
        model,
        config.geometry,
        ("copy", "bit_not"),
        config.test_cases,
        batch_size=2,
    )
    assert [result.task for result in results] == ["copy", "bit_not", "aggregate"]
    inferred = infer(
        model, config.geometry, "10", tape_slots=5, task_name="copy", steps=0
    )
    assert inferred.values.tolist() == [1.0, -1.0, 0.0, 0.0, 0.0]

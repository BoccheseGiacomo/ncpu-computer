from __future__ import annotations

import math
import random
import shutil
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import torch

from .config import ExperimentConfig, TrainingConfig, integer_strata
from .model import NeuralCellularAutomaton
from .tape import TapeLayout, quantize
from .tasks import MultiTaskDataset, binary_tasks, semantic_correct


CHECKPOINT_FORMAT = 5


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _task_signatures(
    config: ExperimentConfig, task_names: tuple[str, ...]
) -> tuple[str, ...]:
    datasets = MultiTaskDataset.from_tasks(
        binary_tasks(task_names, config.training.input_max_length_max),
        config.training.tape_slots_max,
    )
    return datasets.signatures


@dataclass(frozen=True)
class TrialSpec:
    tape_slots: int
    input_max_length: int
    free_steps: int
    supervision_steps: int

    @property
    def steps(self) -> int:
        return self.free_steps + self.supervision_steps


def _sample_integer(lower: int, upper: int, generator: torch.Generator) -> int:
    return int(torch.randint(lower, upper + 1, (), generator=generator))


def sample_trials(
    training: TrainingConfig, generator: torch.Generator
) -> tuple[TrialSpec, ...]:
    training.validate()
    tape_strata = integer_strata(
        training.tape_slots_min, training.tape_slots_max, training.n_trials
    )
    input_strata = integer_strata(
        training.input_max_length_min,
        training.input_max_length_max,
        training.n_trials,
    )
    time_strata = integer_strata(
        training.free_steps_min, training.free_steps_max, training.n_trials
    )
    trials = []
    for tape_range, input_range, time_range in zip(
        tape_strata, input_strata, time_strata
    ):
        if tape_range[0] < input_range[0] + 1:
            raise ValueError("coupled tape/input strata have no feasible sample")
        tape_slots = _sample_integer(*tape_range, generator)
        input_high = min(input_range[1], tape_slots - 1)
        input_max = _sample_integer(input_range[0], input_high, generator)
        free_steps = _sample_integer(*time_range, generator)
        supervision_steps = round(training.supervision_ratio * free_steps)
        trials.append(TrialSpec(tape_slots, input_max, free_steps, supervision_steps))
    order = torch.randperm(training.n_trials, generator=generator).tolist()
    return tuple(trials[index] for index in order)


def validation_trials(training: TrainingConfig) -> tuple[TrialSpec, ...]:
    training.validate()
    tape_strata = integer_strata(
        training.tape_slots_min, training.tape_slots_max, training.n_trials
    )
    input_strata = integer_strata(
        training.input_max_length_min,
        training.input_max_length_max,
        training.n_trials,
    )
    time_strata = integer_strata(
        training.free_steps_min, training.free_steps_max, training.n_trials
    )
    trials = []
    for tape_range, input_range, time_range in zip(
        tape_strata, input_strata, time_strata
    ):
        tape_slots = sum(tape_range) // 2
        input_max = min(sum(input_range) // 2, tape_slots - 1)
        free_steps = sum(time_range) // 2
        trials.append(
            TrialSpec(
                tape_slots,
                input_max,
                free_steps,
                round(training.supervision_ratio * free_steps),
            )
        )
    return tuple(trials)


@dataclass(frozen=True)
class LossComponents:
    total: torch.Tensor
    base: torch.Tensor
    per_task: torch.Tensor


def supervised_loss(
    rollout: torch.Tensor,
    target: torch.Tensor,
    layout: TapeLayout,
    io_channel: int,
    free_steps: int,
    supervision_steps: int,
    task_indices: torch.Tensor | None = None,
    task_count: int | None = None,
) -> LossComponents:
    start = free_steps + 1
    end = start + supervision_steps
    if rollout.ndim != 5:
        raise ValueError(
            "rollout must have shape (batch, time, channels, height, width)"
        )
    if not 0 <= io_channel < rollout.shape[2]:
        raise ValueError("io_channel does not exist in rollout")
    if start < 1 or end > rollout.shape[1]:
        raise ValueError("rollout does not cover the supervision window")
    if target.shape != (rollout.shape[0], layout.tape_slots):
        raise ValueError("target shape does not match rollout and tape layout")
    prediction = layout.extract_tape(rollout[:, start:end, io_channel])
    per_example = (
        (prediction - target.to(prediction.device).unsqueeze(1))
        .square()
        .mean(dim=(1, 2))
    )
    if task_indices is None:
        if task_count is not None:
            raise ValueError("task_count requires task_indices")
        per_task = per_example.mean().reshape(1)
    else:
        task_indices = torch.as_tensor(
            task_indices, dtype=torch.int64, device=prediction.device
        )
        if task_indices.shape != (rollout.shape[0],):
            raise ValueError("task_indices must contain one index per example")
        if type(task_count) is not int or task_count < 1:
            raise ValueError("task_count must be a positive integer")
        if bool(((task_indices < 0) | (task_indices >= task_count)).any()):
            raise ValueError("task index is out of range")
        values = []
        for task_index in range(task_count):
            selected = per_example[task_indices == task_index]
            if not selected.numel():
                raise ValueError("every task must occur in a balanced batch")
            values.append(selected.mean())
        per_task = torch.stack(values)
    base = per_task.mean()
    return LossComponents(base, base, per_task)


def cosine_learning_rate(config: ExperimentConfig, update: int) -> float:
    training = config.training
    if not 0 <= update < training.updates:
        raise ValueError("update must be in [0, updates)")
    if training.warmup_updates and update < training.warmup_updates:
        return training.learning_rate * (update + 1) / training.warmup_updates
    decay_updates = training.updates - training.warmup_updates
    decay_index = update - training.warmup_updates
    progress = 1.0 if decay_updates == 1 else decay_index / (decay_updates - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (
        training.final_learning_rate
        + (training.learning_rate - training.final_learning_rate) * cosine
    )


def perception_noise(config: TrainingConfig, update: int) -> float:
    progress = 0.0 if config.updates == 1 else update / (config.updates - 1)
    return config.perception_noise_start + progress * (
        config.perception_noise_end - config.perception_noise_start
    )


@dataclass(frozen=True)
class StepMetrics:
    update: int
    loss: float
    learning_rate: float
    perception_noise: float
    gradient_norm: float
    semantic_accuracy: float
    raw_accuracy: float
    trials: tuple[TrialSpec, ...]
    validation_loss: float | None = None

    def to_dict(self) -> dict:
        data = self.__dict__.copy()
        data["trials"] = [asdict(trial) for trial in self.trials]
        return data


@dataclass(frozen=True)
class SeedResult:
    seed: int
    checkpoint: Path
    best_validation_loss: float
    history: list[dict]


class Trainer:
    def __init__(
        self,
        config: ExperimentConfig,
        task_names: tuple[str, ...],
        model: NeuralCellularAutomaton | None = None,
    ):
        config.validate()
        binary_tasks(task_names, 0)
        self.config = config
        self.task_names = tuple(task_names)
        self.task_signatures = _task_signatures(config, self.task_names)
        self.device = resolve_device(config.training.device)
        random.seed(config.training.seed)
        torch.manual_seed(config.training.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.training.seed)
        self.model = (
            NeuralCellularAutomaton(config.model, config.geometry, self.task_names)
            if model is None
            else model
        ).to(self.device)
        if self.model.config != config.model or self.model.geometry != config.geometry:
            raise ValueError("model and experiment configurations differ")
        if self.model.task_names != self.task_names:
            raise ValueError("model and trainer task order differs")
        for parameter in self.model.shared_parameters:
            parameter.requires_grad_(config.training.train_rule)
        for parameter in self.model.program_parameters:
            parameter.requires_grad_(config.training.train_program)
        groups = []
        if config.training.train_rule:
            groups.append(
                {
                    "params": self.model.shared_parameters,
                    "weight_decay": config.training.weight_decay,
                    "name": "rule",
                }
            )
        if config.training.train_program:
            groups.append(
                {
                    "params": self.model.program_parameters,
                    "weight_decay": config.training.program_weight_decay,
                    "name": "program",
                }
            )
        self.optimizer = torch.optim.Adam(groups, lr=config.training.learning_rate)
        self.trial_generator = torch.Generator().manual_seed(config.training.seed)
        self.data_generator = torch.Generator().manual_seed(config.training.seed + 1)
        self.current_update = 0
        self.best_validation_loss = math.inf
        self.history: list[dict] = []

    def _datasets(
        self, trial: TrialSpec, include_shorter: bool = True
    ) -> MultiTaskDataset:
        return MultiTaskDataset.from_tasks(
            binary_tasks(
                self.task_names,
                trial.input_max_length,
                include_shorter=include_shorter,
            ),
            trial.tape_slots,
        )

    def _prepare(self, datasets: MultiTaskDataset) -> tuple[torch.Tensor, ...]:
        batch = datasets.balanced_sample(
            self.config.training.batch_size_per_task, self.data_generator
        )
        return tuple(value.to(self.device) for value in batch)

    def train_step(self) -> StepMetrics:
        self.model.train()
        learning_rate = cosine_learning_rate(self.config, self.current_update)
        noise = perception_noise(self.config.training, self.current_update)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        trials = sample_trials(self.config.training, self.trial_generator)
        self.optimizer.zero_grad(set_to_none=True)
        loss_total = semantic_total = raw_total = 0.0
        for trial in trials:
            datasets = self._datasets(trial)
            inputs, targets, lengths, task_indices = self._prepare(datasets)
            layout = TapeLayout(self.config.geometry, trial.tape_slots)
            initial = self.model.initial_state(layout.render_tape(inputs), task_indices)
            rollout = self.model(initial, trial.steps, noise)
            losses = supervised_loss(
                rollout,
                targets,
                layout,
                self.config.model.io_channel,
                trial.free_steps,
                trial.supervision_steps,
                task_indices,
                len(self.task_names),
            )
            if not torch.isfinite(losses.total):
                raise FloatingPointError(
                    f"non-finite loss at update {self.current_update + 1}"
                )
            (losses.total / len(trials)).backward()
            loss_total += float(losses.total.detach()) / len(trials)
            with torch.no_grad():
                final = quantize(
                    layout.extract_tape(rollout[:, -1, self.config.model.io_channel])
                )
                expected = targets.to(torch.int8)
                raw_total += float((final == expected).all(dim=1).float().mean()) / len(
                    trials
                )
                semantic = torch.empty(
                    final.shape[0], dtype=torch.bool, device=self.device
                )
                for task_index, dataset in enumerate(datasets.datasets):
                    selected = task_indices == task_index
                    semantic[selected] = semantic_correct(
                        final[selected],
                        expected[selected],
                        lengths[selected],
                        dataset.output_mode,
                    )
                semantic_total += float(semantic.float().mean()) / len(trials)
            del rollout, losses
        parameters = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if self.config.training.grad_clip is None:
            gradient_norm = float(
                torch.stack(
                    [parameter.grad.detach().square().sum() for parameter in parameters]
                )
                .sum()
                .sqrt()
            )
        else:
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    parameters, self.config.training.grad_clip
                )
            )
        if not math.isfinite(gradient_norm):
            raise FloatingPointError(
                f"non-finite gradient norm at update {self.current_update + 1}"
            )
        self.optimizer.step()
        self.current_update += 1
        return StepMetrics(
            self.current_update,
            loss_total,
            learning_rate,
            noise,
            gradient_norm,
            semantic_total,
            raw_total,
            trials,
        )

    @torch.no_grad()
    def validation_loss(self) -> float:
        was_training = self.model.training
        self.model.eval()
        values = []
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.config.training.seed)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(self.config.training.seed)
                for trial in validation_trials(self.config.training):
                    datasets = self._datasets(trial)
                    layout = TapeLayout(self.config.geometry, trial.tape_slots)
                    for task_index, dataset in enumerate(datasets.datasets):
                        weighted_loss = 0.0
                        for offset in range(
                            0, len(dataset), self.config.training.batch_size_per_task
                        ):
                            indices = torch.arange(
                                offset,
                                min(
                                    offset + self.config.training.batch_size_per_task,
                                    len(dataset),
                                ),
                            )
                            inputs, targets, _ = (
                                value.to(self.device) for value in dataset.take(indices)
                            )
                            task_indices = torch.full(
                                (len(indices),),
                                task_index,
                                dtype=torch.int64,
                                device=self.device,
                            )
                            rollout = self.model(
                                self.model.initial_state(
                                    layout.render_tape(inputs), task_indices
                                ),
                                trial.steps,
                            )
                            loss = supervised_loss(
                                rollout,
                                targets,
                                layout,
                                self.config.model.io_channel,
                                trial.free_steps,
                                trial.supervision_steps,
                            ).base
                            weighted_loss += float(loss) * len(indices)
                        values.append(weighted_loss / len(dataset))
        finally:
            self.model.train(was_training)
        result = sum(values) / len(values)
        if not math.isfinite(result):
            raise FloatingPointError("validation produced a non-finite loss")
        return result

    def fit(
        self,
        checkpoint_dir: str | Path = "checkpoints",
        progress_every: int = 25,
        callback: Callable[[StepMetrics], None] | None = None,
    ) -> list[dict]:
        if progress_every < 1:
            raise ValueError("progress_every must be positive")
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        while self.current_update < self.config.training.updates:
            metrics = self.train_step()
            should_validate = (
                metrics.update % self.config.training.validation_every == 0
                or metrics.update == self.config.training.updates
            )
            validation_loss = self.validation_loss() if should_validate else None
            is_best = (
                validation_loss is not None
                and validation_loss < self.best_validation_loss
            )
            if is_best:
                self.best_validation_loss = validation_loss
            metrics = replace(metrics, validation_loss=validation_loss)
            self.history.append(metrics.to_dict())
            if is_best:
                self.save(checkpoint_dir / "best.pt")
            if callback is not None:
                callback(metrics)
            if metrics.update % progress_every == 0 or should_validate:
                validation = (
                    "" if validation_loss is None else f" val={validation_loss:.6g}"
                )
                print(
                    f"update {metrics.update:5d}/{self.config.training.updates} "
                    f"loss={metrics.loss:.6g} semantic={metrics.semantic_accuracy:.2%} "
                    f"raw={metrics.raw_accuracy:.2%} lr={metrics.learning_rate:.3g}"
                    f" noise={metrics.perception_noise:.3g}{validation}"
                )
            if (
                metrics.update % self.config.training.checkpoint_every == 0
                or metrics.update == self.config.training.updates
            ):
                self.save(checkpoint_dir / "latest.pt")
        return self.history

    def checkpoint(self) -> dict:
        state = {
            "format_version": CHECKPOINT_FORMAT,
            "config": self.config.to_dict(),
            "task_names": self.task_names,
            "task_signatures": self.task_signatures,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update": self.current_update,
            "best_validation_loss": self.best_validation_loss,
            "history": self.history,
            "trial_rng_state": self.trial_generator.get_state(),
            "data_rng_state": self.data_generator.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
        }
        if torch.cuda.is_available():
            state["cuda_rng_state"] = torch.cuda.get_rng_state_all()
        return state

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.checkpoint(), temporary)
        temporary.replace(path)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        task_names: tuple[str, ...] | None = None,
        device: str | None = None,
    ) -> "Trainer":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        _validate_checkpoint(checkpoint, task_names)
        config_data = checkpoint["config"]
        if device is not None:
            config_data = {
                **config_data,
                "training": {**config_data["training"], "device": device},
            }
        names = tuple(checkpoint["task_names"])
        trainer = cls(ExperimentConfig.from_dict(config_data), names)
        trainer.model.load_state_dict(checkpoint["model"], strict=True)
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        trainer.current_update = int(checkpoint["update"])
        trainer.best_validation_loss = float(checkpoint["best_validation_loss"])
        trainer.history = list(checkpoint["history"])
        trainer.trial_generator.set_state(checkpoint["trial_rng_state"])
        trainer.data_generator.set_state(checkpoint["data_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        if trainer.device.type == "cuda" and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        return trainer


def _validate_checkpoint(checkpoint: dict, task_names: tuple[str, ...] | None) -> None:
    if checkpoint.get("format_version") != CHECKPOINT_FORMAT:
        raise ValueError("checkpoint uses an incompatible model representation")
    stored = tuple(checkpoint.get("task_names", ()))
    if not stored:
        raise ValueError("checkpoint does not define its ordered tasks")
    if task_names is not None and tuple(task_names) != stored:
        raise ValueError("checkpoint task order does not match")
    config = ExperimentConfig.from_dict(checkpoint["config"])
    if tuple(checkpoint.get("task_signatures", ())) != _task_signatures(config, stored):
        raise ValueError("checkpoint task definitions do not match")


def load_model(
    path: str | Path,
    device: str = "auto",
) -> tuple[NeuralCellularAutomaton, ExperimentConfig, dict]:
    resolved = resolve_device(device)
    checkpoint = torch.load(path, map_location=resolved, weights_only=False)
    _validate_checkpoint(checkpoint, None)
    config = ExperimentConfig.from_dict(checkpoint["config"])
    model = NeuralCellularAutomaton(
        config.model, config.geometry, tuple(checkpoint["task_names"])
    ).to(resolved)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, config, checkpoint


def train_seeds(
    config: ExperimentConfig,
    task_names: tuple[str, ...],
    seeds: tuple[int, ...],
    checkpoint_dir: str | Path = "checkpoints",
    *,
    resume: bool = False,
    progress_every: int = 25,
) -> list[SeedResult]:
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must contain non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in seeds:
        model_config = config.model
        if model_config.learnable_kernel_init == "random":
            model_config = replace(model_config, random_kernel_seed=seed)
        seed_config = replace(
            config,
            model=model_config,
            training=replace(config.training, seed=seed),
        )
        seed_dir = checkpoint_dir / f"seed_{seed}"
        latest = seed_dir / "latest.pt"
        if resume and latest.is_file():
            trainer = Trainer.from_checkpoint(
                latest, task_names, seed_config.training.device
            )
            if trainer.config != seed_config:
                raise ValueError(f"seed {seed} checkpoint configuration does not match")
        else:
            trainer = Trainer(seed_config, task_names)
        history = trainer.fit(seed_dir, progress_every=progress_every)
        results.append(
            SeedResult(
                seed, seed_dir / "best.pt", trainer.best_validation_loss, history
            )
        )
    best = min(results, key=lambda result: result.best_validation_loss)
    shutil.copy2(best.checkpoint, checkpoint_dir / "best.pt")
    return results

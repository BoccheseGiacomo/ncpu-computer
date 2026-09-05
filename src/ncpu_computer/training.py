from __future__ import annotations

import math
import random
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import torch

from .config import ExperimentConfig
from .model import NeuralCellularAutomaton
from .tape import TapeLayout, quantize
from .tasks import TaskDataset, semantic_correct


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


@dataclass(frozen=True)
class LossComponents:
    total: torch.Tensor
    base: torch.Tensor
    terminator: torch.Tensor
    tail: torch.Tensor


def _masked_mean(error: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.shape != (error.shape[0], error.shape[2]):
        raise ValueError("loss mask shape does not match prediction")
    mask = mask.to(device=error.device, dtype=error.dtype).unsqueeze(1)
    count = mask.sum() * error.shape[1]
    if float(count) == 0.0:
        return error.sum() * 0.0
    return (error * mask).sum() / count


def supervised_loss(
    rollout: torch.Tensor,
    target: torch.Tensor,
    layout: TapeLayout,
    io_channel: int,
    free_steps: int,
    supervision_steps: int,
    terminator_mask: torch.Tensor,
    tail_mask: torch.Tensor,
    *,
    terminator_weight: float = 0.0,
    tail_weight: float = 0.0,
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
    if target.shape != (rollout.shape[0], layout.config.tape_slots):
        raise ValueError("target shape does not match rollout and tape layout")
    if terminator_weight < 0 or tail_weight < 0:
        raise ValueError("auxiliary loss weights cannot be negative")

    prediction = layout.extract_tape(rollout[:, start:end, io_channel])
    expected = target.to(prediction.device).unsqueeze(1)
    error = (prediction - expected).square()
    base = error.mean()
    terminator = _masked_mean(error, terminator_mask)
    tail = _masked_mean(error, tail_mask)
    total = base + terminator_weight * terminator + tail_weight * tail
    return LossComponents(total=total, base=base, terminator=terminator, tail=tail)


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


@dataclass(frozen=True)
class StepMetrics:
    update: int
    loss: float
    base_loss: float
    terminator_loss: float
    tail_loss: float
    learning_rate: float
    gradient_norm: float
    semantic_accuracy: float
    raw_accuracy: float
    validation_loss: float | None = None

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "update": self.update,
            "loss": self.loss,
            "base_loss": self.base_loss,
            "terminator_loss": self.terminator_loss,
            "tail_loss": self.tail_loss,
            "learning_rate": self.learning_rate,
            "gradient_norm": self.gradient_norm,
            "semantic_accuracy": self.semantic_accuracy,
            "raw_accuracy": self.raw_accuracy,
            "validation_loss": self.validation_loss,
        }


@dataclass(frozen=True)
class SeedResult:
    seed: int
    checkpoint: Path
    best_validation_loss: float
    history: list[dict[str, float | int | None]]


class Trainer:
    def __init__(
        self,
        config: ExperimentConfig,
        dataset: TaskDataset,
        model: NeuralCellularAutomaton | None = None,
    ):
        config.validate()
        if dataset.tape_slots != config.geometry.tape_slots:
            raise ValueError("dataset and geometry have different tape capacities")
        self.config = config
        self.dataset = dataset
        self.device = resolve_device(config.training.device)
        random.seed(config.training.seed)
        torch.manual_seed(config.training.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.training.seed)
        self.model = (
            NeuralCellularAutomaton(config.model) if model is None else model
        ).to(self.device)
        if self.model.config != config.model:
            raise ValueError("model and experiment configurations differ")
        self.layout = TapeLayout(config.geometry)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.data_generator = torch.Generator().manual_seed(config.training.seed)
        self.current_update = 0
        self.best_validation_loss = math.inf
        self.history: list[dict[str, float | int | None]] = []

    def _prepare(self, batch: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        inputs, targets, lengths, terminator, tail = (
            value.to(self.device) for value in batch
        )
        io_grid = self.layout.render_tape(inputs)
        return io_grid, targets, lengths, terminator, tail

    def _loss(
        self,
        rollout: torch.Tensor,
        targets: torch.Tensor,
        terminator: torch.Tensor,
        tail: torch.Tensor,
    ) -> LossComponents:
        training = self.config.training
        return supervised_loss(
            rollout,
            targets,
            self.layout,
            self.config.model.io_channel,
            training.free_steps,
            training.supervision_steps,
            terminator,
            tail,
            terminator_weight=training.terminator_weight,
            tail_weight=training.tail_weight,
        )

    def train_step(self) -> StepMetrics:
        self.model.train()
        learning_rate = cosine_learning_rate(self.config, self.current_update)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        batch = self.dataset.sample(
            self.config.training.batch_size, self.data_generator
        )
        io_grid, targets, lengths, terminator, tail = self._prepare(batch)
        rollout = self.model(
            self.model.initial_state(io_grid), self.config.training.rollout_steps
        )
        losses = self._loss(rollout, targets, terminator, tail)
        if not torch.isfinite(losses.total):
            raise FloatingPointError(
                f"non-finite loss at update {self.current_update + 1}"
            )

        self.optimizer.zero_grad(set_to_none=True)
        losses.total.backward()
        if self.config.training.grad_clip is None:
            squared_norm = sum(
                parameter.grad.detach().square().sum()
                for parameter in self.model.parameters()
                if parameter.grad is not None
            )
            gradient_norm = float(squared_norm.sqrt())
        else:
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.grad_clip
                )
            )
        if not math.isfinite(gradient_norm):
            raise FloatingPointError(
                f"non-finite gradient norm at update {self.current_update + 1}"
            )
        self.optimizer.step()

        with torch.no_grad():
            final = quantize(
                self.layout.extract_tape(rollout[:, -1, self.config.model.io_channel])
            )
            discrete_targets = targets.to(torch.int8)
            raw = (final == discrete_targets).all(dim=1)
            semantic = semantic_correct(
                final, discrete_targets, lengths, self.dataset.output_mode
            )

        self.current_update += 1
        return StepMetrics(
            update=self.current_update,
            loss=float(losses.total.detach()),
            base_loss=float(losses.base.detach()),
            terminator_loss=float(losses.terminator.detach()),
            tail_loss=float(losses.tail.detach()),
            learning_rate=learning_rate,
            gradient_norm=gradient_norm,
            semantic_accuracy=float(semantic.float().mean()),
            raw_accuracy=float(raw.float().mean()),
        )

    @torch.no_grad()
    def validation_loss(self) -> float:
        was_training = self.model.training
        self.model.eval()
        component_sums = {"base": 0.0, "terminator": 0.0, "tail": 0.0}
        component_counts = {"base": 0.0, "terminator": 0.0, "tail": 0.0}
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(self.config.training.seed)
                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(self.config.training.seed)
                for offset in range(
                    0, len(self.dataset), self.config.training.batch_size
                ):
                    end = min(
                        offset + self.config.training.batch_size, len(self.dataset)
                    )
                    indices = torch.arange(offset, end)
                    io_grid, targets, _, terminator, tail = self._prepare(
                        self.dataset.take(indices)
                    )
                    rollout = self.model(
                        self.model.initial_state(io_grid),
                        self.config.training.rollout_steps,
                    )
                    losses = self._loss(rollout, targets, terminator, tail)
                    weights = {
                        "base": float(len(indices) * self.dataset.tape_slots),
                        "terminator": float(terminator.sum()),
                        "tail": float(tail.sum()),
                    }
                    for name, weight in weights.items():
                        component_sums[name] += float(getattr(losses, name)) * weight
                        component_counts[name] += weight
        finally:
            self.model.train(was_training)
        means = {
            name: (component_sums[name] / count if count else 0.0)
            for name, count in component_counts.items()
        }
        return (
            means["base"]
            + self.config.training.terminator_weight * means["terminator"]
            + self.config.training.tail_weight * means["tail"]
        )

    def fit(
        self,
        checkpoint_dir: str | Path = "checkpoints",
        progress_every: int = 25,
        callback: Callable[[StepMetrics], None] | None = None,
    ) -> list[dict[str, float | int | None]]:
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
            is_best = validation_loss is not None and (
                validation_loss < self.best_validation_loss
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
                    f"{validation}"
                )
            if (
                metrics.update % self.config.training.checkpoint_every == 0
                or metrics.update == self.config.training.updates
            ):
                self.save(checkpoint_dir / "latest.pt")
        return self.history

    def checkpoint(self) -> dict:
        state = {
            "format_version": 1,
            "config": self.config.to_dict(),
            "dataset_signature": self.dataset.signature,
            "dataset_name": self.dataset.name,
            "output_mode": self.dataset.output_mode,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "update": self.current_update,
            "best_validation_loss": self.best_validation_loss,
            "history": self.history,
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
        dataset: TaskDataset,
        device: str | None = None,
    ) -> "Trainer":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format")
        if checkpoint.get("dataset_signature") != dataset.signature:
            raise ValueError("checkpoint dataset does not match the supplied dataset")
        config_data = checkpoint["config"]
        if device is not None:
            config_data = {
                **config_data,
                "training": {**config_data["training"], "device": device},
            }
        trainer = cls(ExperimentConfig.from_dict(config_data), dataset)
        trainer.model.load_state_dict(checkpoint["model"], strict=True)
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        trainer.current_update = int(checkpoint["update"])
        trainer.best_validation_loss = float(checkpoint["best_validation_loss"])
        trainer.history = list(checkpoint["history"])
        trainer.data_generator.set_state(checkpoint["data_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        if trainer.device.type == "cuda" and "cuda_rng_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        return trainer


def load_model(
    path: str | Path, device: str = "auto"
) -> tuple[NeuralCellularAutomaton, ExperimentConfig, dict]:
    resolved = resolve_device(device)
    checkpoint = torch.load(path, map_location=resolved, weights_only=False)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    config = ExperimentConfig.from_dict(checkpoint["config"])
    model = NeuralCellularAutomaton(config.model).to(resolved)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, config, checkpoint


def train_seeds(
    config: ExperimentConfig,
    dataset: TaskDataset,
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
                latest, dataset, device=seed_config.training.device
            )
            if trainer.config != seed_config:
                raise ValueError(f"seed {seed} checkpoint configuration does not match")
        else:
            trainer = Trainer(seed_config, dataset)
        history = trainer.fit(seed_dir, progress_every=progress_every)
        results.append(
            SeedResult(
                seed=seed,
                checkpoint=seed_dir / "best.pt",
                best_validation_loss=trainer.best_validation_loss,
                history=history,
            )
        )
    best = min(results, key=lambda result: result.best_validation_loss)
    shutil.copy2(best.checkpoint, checkpoint_dir / "best.pt")
    return results

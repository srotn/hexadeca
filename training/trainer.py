"""Config-driven AlphaZero optimizer loop with AMP, metrics, and resume."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from math import cos, isfinite, pi
from pathlib import Path
from time import perf_counter
from typing import Protocol

import torch
from torch.amp.grad_scaler import GradScaler
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR

from config.schema import AppConfig, TrainingConfig
from network import (
    AlphaZeroLoss,
    LossOutput,
    LossWeights,
    NetworkSpecification,
    PolicyValueNetwork,
)
from training.checkpoint import (
    CheckpointManager,
    CheckpointMetadata,
    MetricValue,
)
from training.data import ReplayBatch, build_replay_data_loader
from training.errors import (
    TrainerConfigurationError,
    TrainerStateError,
    TrainerStepError,
)
from training.replay import ReplayBuffer, ReplaySnapshot


class MetricSink(Protocol):
    """Minimal scalar metric destination implemented by TensorBoard."""

    def add_scalar(self, tag: str, value: float, step: int) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


class TensorBoardMetricSink:
    """Lazy TensorBoard adapter that keeps the dependency at the app boundary."""

    __slots__ = ("_writer",)

    def __init__(
        self, directory: Path, *, flush_seconds: int, maximum_queue: int
    ) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise TrainerConfigurationError(
                "TensorBoard logging is enabled but tensorboard is not installed"
            ) from error
        directory.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(
            log_dir=str(directory),
            flush_secs=flush_seconds,
            max_queue=maximum_queue,
        )

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self._writer.add_scalar(tag, value, step)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()


class _NullMetricSink:
    __slots__ = ()

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class TrainingBatchMetrics:
    """One completed optimizer update for callbacks and live monitoring."""

    iteration: int
    batch_index: int
    global_step: int
    positions: int
    total_loss: float
    policy_loss: float
    win_loss: float
    black_score_loss: float
    white_score_loss: float
    policy_entropy: float
    learning_rate: float
    gradient_norm: float
    amp_retries: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class TrainingIterationMetrics:
    """Position-weighted metrics for one fixed-budget training iteration."""

    iteration: int
    batches: int
    positions: int
    starting_global_step: int
    completed_global_step: int
    total_loss: float
    policy_loss: float
    win_loss: float
    black_score_loss: float
    white_score_loss: float
    policy_entropy: float
    starting_learning_rate: float
    ending_learning_rate: float
    mean_gradient_norm: float
    maximum_gradient_norm: float
    amp_retries: int
    elapsed_seconds: float
    amp_enabled: bool
    device: str
    checkpoint_id: str | None = None

    def checkpoint_metrics(self) -> dict[str, MetricValue]:
        """Return the stable scalar subset persisted in checkpoint metadata."""

        return {
            "training_steps_completed": self.completed_global_step,
            "training_batches": self.batches,
            "training_positions": self.positions,
            "total_loss": self.total_loss,
            "policy_loss": self.policy_loss,
            "win_loss": self.win_loss,
            "black_score_loss": self.black_score_loss,
            "white_score_loss": self.white_score_loss,
            "policy_entropy": self.policy_entropy,
            "learning_rate": self.ending_learning_rate,
            "mean_gradient_norm": self.mean_gradient_norm,
            "maximum_gradient_norm": self.maximum_gradient_norm,
            "amp_retries": self.amp_retries,
            "training_elapsed_seconds": self.elapsed_seconds,
            "amp_enabled": self.amp_enabled,
            "device": self.device,
        }


BatchCallback = Callable[[TrainingBatchMetrics], None]


def build_optimizer(model: PolicyValueNetwork, config: TrainingConfig) -> Optimizer:
    """Construct the configured optimizer without duplicating L2 in the loss."""

    if config.optimizer != "adamw":
        raise TrainerConfigurationError(
            f"Unsupported training optimizer: {config.optimizer}"
        )
    return AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(config.adamw_beta1, config.adamw_beta2),
        eps=config.adamw_epsilon,
        amsgrad=config.adamw_amsgrad,
    )


def build_scheduler(optimizer: Optimizer, config: TrainingConfig) -> LambdaLR:
    """Construct the confirmed step-based warmup and cosine schedule."""

    if config.scheduler != "warmup-cosine":
        raise TrainerConfigurationError(
            f"Unsupported training scheduler: {config.scheduler}"
        )

    def schedule(step: int) -> float:
        return warmup_cosine_multiplier(step, config)

    return LambdaLR(optimizer, schedule)


def warmup_cosine_multiplier(step: int, config: TrainingConfig) -> float:
    """Return the LR multiplier for one optimizer-step index."""

    if type(step) is not int or step < 0:
        raise TrainerConfigurationError("Scheduler step must be nonnegative")
    minimum = config.scheduler_minimum_learning_rate / config.learning_rate
    if step < config.scheduler_warmup_steps:
        return (step + 1) / config.scheduler_warmup_steps
    decay_step = step - config.scheduler_warmup_steps
    if decay_step >= config.scheduler_decay_steps:
        return minimum
    progress = decay_step / config.scheduler_decay_steps
    cosine = 0.5 * (1.0 + cos(pi * progress))
    return minimum + (1.0 - minimum) * cosine


class Trainer:
    """Own model updates, schedule state, metrics, and checkpoint boundaries."""

    __slots__ = (
        "_amp_dtype",
        "_amp_enabled",
        "_checkpoint_manager",
        "_closed",
        "_completed_iteration",
        "_config",
        "_device",
        "_loss",
        "_metric_sink",
        "_model",
        "_optimizer",
        "_owns_metric_sink",
        "_parent_checkpoint_id",
        "_scaler",
        "_scheduler",
        "_specification",
        "_updates_completed",
    )

    def __init__(
        self,
        config: AppConfig,
        specification: NetworkSpecification,
        model: PolicyValueNetwork,
        *,
        device: torch.device | str,
        checkpoint_manager: CheckpointManager | None = None,
        metric_sink: MetricSink | None = None,
        tensorboard_directory: Path | None = None,
    ) -> None:
        specification.validate()
        model.specification.ensure_compatible(specification)
        configured = NetworkSpecification.from_config(config.rules, config.network)
        configured.ensure_compatible(specification)
        resolved_device = torch.device(device)
        if resolved_device.type not in {"cpu", "cuda"}:
            raise TrainerConfigurationError(
                "Trainer device must currently be CPU or CUDA"
            )
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise TrainerConfigurationError("CUDA training requested but unavailable")
        if (
            config.training.checkpoint_interval_iterations > 0
            and checkpoint_manager is None
        ):
            raise TrainerConfigurationError(
                "CheckpointManager is required when periodic checkpoints are enabled"
            )

        self._config = config
        self._specification = specification
        self._device = resolved_device
        self._model = model.to(resolved_device)
        self._loss = AlphaZeroLoss(LossWeights.from_config(config.training))
        self._optimizer = build_optimizer(self._model, config.training)
        self._scheduler = build_scheduler(self._optimizer, config.training)
        self._amp_enabled = (
            config.training.amp_enabled and resolved_device.type == "cuda"
        )
        self._amp_dtype = _amp_dtype(config.training.amp_dtype)
        self._scaler = GradScaler(
            "cuda",
            init_scale=config.training.amp_initial_scale,
            growth_factor=config.training.amp_growth_factor,
            backoff_factor=config.training.amp_backoff_factor,
            growth_interval=config.training.amp_growth_interval,
            enabled=self._amp_enabled,
        )
        self._checkpoint_manager = checkpoint_manager
        self._completed_iteration = 0
        self._updates_completed = 0
        self._parent_checkpoint_id: str | None = None
        self._closed = False

        if metric_sink is not None:
            self._metric_sink = metric_sink
            self._owns_metric_sink = False
        elif config.training.tensorboard_enabled:
            directory = tensorboard_directory or (
                config.paths.run_directory / "tensorboard"
            )
            self._metric_sink = TensorBoardMetricSink(
                directory,
                flush_seconds=config.training.tensorboard_flush_seconds,
                maximum_queue=config.training.tensorboard_max_queue,
            )
            self._owns_metric_sink = True
        else:
            self._metric_sink = _NullMetricSink()
            self._owns_metric_sink = True

    @property
    def model(self) -> PolicyValueNetwork:
        """Return the exclusively owned trainable model."""

        return self._model

    @property
    def optimizer(self) -> Optimizer:
        """Return optimizer state for inspection and checkpoint integration."""

        return self._optimizer

    @property
    def scheduler(self) -> LambdaLR:
        """Return the step-based scheduler owned by this Trainer."""

        return self._scheduler

    @property
    def scaler(self) -> GradScaler:
        """Return the enabled or no-op GradScaler used for every update."""

        return self._scaler

    @property
    def completed_iteration(self) -> int:
        """Return the last fully trained iteration, or zero for a fresh Trainer."""

        return self._completed_iteration

    @property
    def updates_completed(self) -> int:
        """Return the number of successful optimizer updates."""

        return self._updates_completed

    @property
    def amp_enabled(self) -> bool:
        """Return whether CUDA FP16 autocast and scaling are active."""

        return self._amp_enabled

    def train_iteration(
        self,
        replay: ReplayBuffer | ReplaySnapshot,
        *,
        iteration: int | None = None,
        batch_callback: BatchCallback | None = None,
    ) -> TrainingIterationMetrics:
        """Train one exact-budget iteration and publish its periodic checkpoint."""

        self._ensure_open()
        resolved_iteration = (
            self._completed_iteration + 1 if iteration is None else iteration
        )
        if (
            type(resolved_iteration) is not int
            or resolved_iteration != self._completed_iteration + 1
        ):
            raise TrainerStateError(
                "Training iteration must be exactly one after the completed iteration"
            )
        replay_size = len(replay)
        if replay_size < self._config.training.minimum_replay_positions:
            raise TrainerStateError(
                "Replay does not contain the configured minimum training positions"
            )

        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            _iteration_seed(self._config.training.random_seed, resolved_iteration)
        )
        loader = build_replay_data_loader(
            replay,
            self._specification,
            self._config.training,
            generator=generator,
        )
        iterator = iter(loader)
        batch_metrics: list[TrainingBatchMetrics] = []
        starting_step = self._updates_completed
        started_at = perf_counter()
        self._model.train()

        for batch_index in range(1, self._config.training.batches_per_iteration + 1):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                try:
                    batch = next(iterator)
                except StopIteration as error:
                    raise TrainerStateError(
                        "Replay DataLoader produced no training batch"
                    ) from error
            metrics = self._train_batch(
                batch,
                iteration=resolved_iteration,
                batch_index=batch_index,
            )
            batch_metrics.append(metrics)
            self._write_batch_metrics(metrics)
            if batch_callback is not None:
                batch_callback(metrics)

        result = _aggregate_iteration(
            resolved_iteration,
            starting_step,
            batch_metrics,
            elapsed_seconds=perf_counter() - started_at,
            ending_learning_rate=self._current_learning_rate(),
            amp_enabled=self._amp_enabled,
            device=str(self._device),
        )
        self._completed_iteration = resolved_iteration
        self._write_iteration_metrics(result)
        self._metric_sink.flush()

        interval = self._config.training.checkpoint_interval_iterations
        if interval > 0 and resolved_iteration % interval == 0:
            result = replace(result, checkpoint_id=self._save_checkpoint(result))
        return result

    def run_iterations(
        self,
        replay: ReplayBuffer | ReplaySnapshot,
        iteration_count: int,
        *,
        batch_callback: BatchCallback | None = None,
    ) -> tuple[TrainingIterationMetrics, ...]:
        """Run a caller-bounded sequence over the current Replay source."""

        if type(iteration_count) is not int or iteration_count <= 0:
            raise TrainerStateError("Training iteration count must be positive")
        return tuple(
            self.train_iteration(replay, batch_callback=batch_callback)
            for _ in range(iteration_count)
        )

    def resume(self, identifier: str = "latest") -> CheckpointMetadata:
        """Restore the last complete iteration and all optimizer/AMP state."""

        self._ensure_open()
        if self._completed_iteration != 0 or self._updates_completed != 0:
            raise TrainerStateError("Only a fresh Trainer can resume a checkpoint")
        if self._checkpoint_manager is None:
            raise TrainerStateError("CheckpointManager is required for resume")
        metadata = self._checkpoint_manager.read_metadata(identifier)
        _validate_resume_configuration(metadata, self._config.training)
        completed_steps = _required_nonnegative_metric(
            metadata.metrics, "training_steps_completed"
        )
        restored = self._checkpoint_manager.load(
            identifier,
            model=self._model,
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            scaler=self._scaler if metadata.has_scaler else None,
            restore_rng=True,
            map_location=self._device,
        )
        if self._scheduler.last_epoch != completed_steps:
            raise TrainerStateError(
                "Checkpoint scheduler step disagrees with training metadata"
            )
        self._completed_iteration = restored.iteration
        self._updates_completed = completed_steps
        self._parent_checkpoint_id = restored.checkpoint_id
        return restored

    def close(self) -> None:
        """Flush metrics and release an internally owned writer."""

        if self._closed:
            return
        self._metric_sink.flush()
        if self._owns_metric_sink:
            self._metric_sink.close()
        self._closed = True

    def __enter__(self) -> Trainer:
        self._ensure_open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _train_batch(
        self,
        batch: ReplayBatch,
        *,
        iteration: int,
        batch_index: int,
    ) -> TrainingBatchMetrics:
        started_at = perf_counter()
        moved = batch.to(
            self._device,
            non_blocking=(
                self._device.type == "cuda"
                and self._config.training.data_loader_pin_memory
            ),
        )
        positions = int(moved.features.shape[0])
        learning_rate = self._current_learning_rate()
        maximum_attempts = 1 + (
            self._config.training.amp_max_step_retries if self._amp_enabled else 0
        )
        for attempt in range(maximum_attempts):
            self._optimizer.zero_grad(set_to_none=True)
            try:
                with torch.autocast(
                    device_type=self._device.type,
                    dtype=self._amp_dtype,
                    enabled=self._amp_enabled,
                ):
                    output = self._model(moved.features)
                    loss = self._loss(output, moved.targets)
                if not torch.isfinite(loss.total).item():
                    raise TrainerStepError("Training loss is non-finite")
                self._scaler.scale(loss.total).backward()
                self._scaler.unscale_(self._optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(),
                    self._config.training.gradient_clip_norm,
                    error_if_nonfinite=False,
                )
                scale_before = self._scaler.get_scale()
                self._scaler.step(self._optimizer)
                self._scaler.update()
                overflow = (
                    not torch.isfinite(gradient_norm).item()
                    or self._scaler.get_scale() < scale_before
                )
                if overflow:
                    if attempt + 1 == maximum_attempts:
                        raise TrainerStepError(
                            "AMP gradients remained non-finite after configured retries"
                        )
                    continue
                self._scheduler.step()
                break
            except TrainerStepError:
                self._optimizer.zero_grad(set_to_none=True)
                raise
            except Exception as error:
                self._optimizer.zero_grad(set_to_none=True)
                raise TrainerStepError("Training optimizer update failed") from error
        else:
            raise TrainerStepError("Training optimizer update made no progress")

        self._updates_completed += 1
        return _batch_metrics(
            iteration=iteration,
            batch_index=batch_index,
            global_step=self._updates_completed,
            positions=positions,
            loss=loss,
            learning_rate=learning_rate,
            gradient_norm=float(gradient_norm.detach().item()),
            amp_retries=attempt,
            elapsed_seconds=perf_counter() - started_at,
        )

    def _save_checkpoint(self, metrics: TrainingIterationMetrics) -> str:
        if self._checkpoint_manager is None:
            raise TrainerStateError("CheckpointManager is unavailable")
        checkpoint_id = f"iteration-{metrics.iteration:06d}"
        metadata = self._checkpoint_manager.save(
            checkpoint_id,
            model=self._model,
            optimizer=self._optimizer,
            scheduler=self._scheduler,
            scaler=self._scaler,
            iteration=metrics.iteration,
            config=self._config,
            metrics=metrics.checkpoint_metrics(),
            parent_checkpoint_id=self._parent_checkpoint_id,
            aliases=("latest",),
        )
        self._parent_checkpoint_id = metadata.checkpoint_id
        return metadata.checkpoint_id

    def _write_batch_metrics(self, metrics: TrainingBatchMetrics) -> None:
        interval = self._config.training.tensorboard_log_interval_batches
        if metrics.global_step % interval != 0:
            return
        values = {
            "train/batch_total_loss": metrics.total_loss,
            "train/batch_policy_loss": metrics.policy_loss,
            "train/batch_win_loss": metrics.win_loss,
            "train/batch_black_score_loss": metrics.black_score_loss,
            "train/batch_white_score_loss": metrics.white_score_loss,
            "train/batch_policy_entropy": metrics.policy_entropy,
            "train/learning_rate": metrics.learning_rate,
            "train/gradient_norm": metrics.gradient_norm,
            "train/amp_retries": float(metrics.amp_retries),
            "train/batch_seconds": metrics.elapsed_seconds,
        }
        for tag, value in values.items():
            self._metric_sink.add_scalar(tag, value, metrics.global_step)

    def _write_iteration_metrics(self, metrics: TrainingIterationMetrics) -> None:
        values = {
            "train/iteration_total_loss": metrics.total_loss,
            "train/iteration_policy_loss": metrics.policy_loss,
            "train/iteration_win_loss": metrics.win_loss,
            "train/iteration_black_score_loss": metrics.black_score_loss,
            "train/iteration_white_score_loss": metrics.white_score_loss,
            "train/iteration_policy_entropy": metrics.policy_entropy,
            "train/iteration_positions": float(metrics.positions),
            "train/iteration_seconds": metrics.elapsed_seconds,
        }
        for tag, value in values.items():
            self._metric_sink.add_scalar(tag, value, metrics.iteration)

    def _current_learning_rate(self) -> float:
        learning_rates = {float(group["lr"]) for group in self._optimizer.param_groups}
        if len(learning_rates) != 1:
            raise TrainerStateError(
                "Trainer requires one consistent learning rate across parameter groups"
            )
        return learning_rates.pop()

    def _ensure_open(self) -> None:
        if self._closed:
            raise TrainerStateError("Trainer is closed")


def _batch_metrics(
    *,
    iteration: int,
    batch_index: int,
    global_step: int,
    positions: int,
    loss: LossOutput,
    learning_rate: float,
    gradient_norm: float,
    amp_retries: int,
    elapsed_seconds: float,
) -> TrainingBatchMetrics:
    values = tuple(float(item.detach().item()) for item in loss)
    if any(not isfinite(value) for value in (*values, gradient_norm, elapsed_seconds)):
        raise TrainerStepError("Training metrics must be finite")
    return TrainingBatchMetrics(
        iteration=iteration,
        batch_index=batch_index,
        global_step=global_step,
        positions=positions,
        total_loss=values[0],
        policy_loss=values[1],
        win_loss=values[2],
        black_score_loss=values[3],
        white_score_loss=values[4],
        policy_entropy=values[5],
        learning_rate=learning_rate,
        gradient_norm=gradient_norm,
        amp_retries=amp_retries,
        elapsed_seconds=elapsed_seconds,
    )


def _aggregate_iteration(
    iteration: int,
    starting_step: int,
    batches: list[TrainingBatchMetrics],
    *,
    elapsed_seconds: float,
    ending_learning_rate: float,
    amp_enabled: bool,
    device: str,
) -> TrainingIterationMetrics:
    if not batches:
        raise TrainerStateError("Training iteration produced no batches")
    positions = sum(batch.positions for batch in batches)

    def weighted(attribute: str) -> float:
        return (
            sum(float(getattr(batch, attribute)) * batch.positions for batch in batches)
            / positions
        )

    gradient_norms = tuple(batch.gradient_norm for batch in batches)
    return TrainingIterationMetrics(
        iteration=iteration,
        batches=len(batches),
        positions=positions,
        starting_global_step=starting_step,
        completed_global_step=batches[-1].global_step,
        total_loss=weighted("total_loss"),
        policy_loss=weighted("policy_loss"),
        win_loss=weighted("win_loss"),
        black_score_loss=weighted("black_score_loss"),
        white_score_loss=weighted("white_score_loss"),
        policy_entropy=weighted("policy_entropy"),
        starting_learning_rate=batches[0].learning_rate,
        ending_learning_rate=ending_learning_rate,
        mean_gradient_norm=sum(gradient_norms) / len(gradient_norms),
        maximum_gradient_norm=max(gradient_norms),
        amp_retries=sum(batch.amp_retries for batch in batches),
        elapsed_seconds=elapsed_seconds,
        amp_enabled=amp_enabled,
        device=device,
    )


def _validate_resume_configuration(
    metadata: CheckpointMetadata, training: TrainingConfig
) -> None:
    stored = metadata.configuration.get("training")
    if not isinstance(stored, Mapping) or dict(stored) != asdict(training):
        raise TrainerStateError(
            "Checkpoint training configuration does not match the active Trainer"
        )


def _required_nonnegative_metric(metrics: Mapping[str, MetricValue], name: str) -> int:
    value = metrics.get(name)
    if type(value) is not int or value < 0:
        raise TrainerStateError(
            f"Checkpoint metric {name} must be a nonnegative integer"
        )
    return value


def _iteration_seed(random_seed: int, iteration: int) -> int:
    return (random_seed + iteration * 0x9E3779B97F4A7C15) & ((1 << 63) - 1)


def _amp_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    raise TrainerConfigurationError(f"Unsupported AMP dtype: {name}")

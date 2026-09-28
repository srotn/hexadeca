"""Optimizer-loop, scheduler, TensorBoard, AMP, and resume tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from config import load_config
from config.schema import AppConfig
from game import GameEnvironment, Player
from network import NetworkSpecification, PolicyValueNetwork
from training import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointManager,
    ReplayBuffer,
    ReplaySample,
    Trainer,
    TrainerStateError,
    TrainingBatchMetrics,
    restart_cosine_multiplier,
    warmup_cosine_multiplier,
)


class RecordingMetricSink:
    """In-memory scalar sink used to verify Trainer metric routing."""

    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.flush_count = 0
        self.closed = False

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, value, step))

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.closed = True


def _small_config(
    *,
    checkpoint_interval: int,
    tensorboard_enabled: bool = False,
    batches_per_iteration: int = 2,
) -> AppConfig:
    config = load_config()
    network = replace(
        config.network,
        residual_blocks=1,
        channels=4,
        value_hidden_features=8,
    )
    training = replace(
        config.training,
        learning_rate=0.001,
        scheduler_minimum_learning_rate=0.0001,
        scheduler_warmup_steps=2,
        scheduler_decay_steps=4,
        scheduler_restart=replace(
            config.training.scheduler_restart,
            initial_learning_rate=0.0002,
            peak_learning_rate=0.0005,
            warmup_steps=2,
            decay_steps=4,
            minimum_learning_rate=0.0001,
        ),
        batch_size=2,
        batches_per_iteration=batches_per_iteration,
        minimum_replay_positions=2,
        data_loader_workers=0,
        data_loader_pin_memory=False,
        data_loader_drop_last=True,
        checkpoint_interval_iterations=checkpoint_interval,
        tensorboard_enabled=tensorboard_enabled,
        random_seed=73,
    )
    replay = replace(config.replay, capacity_positions=16)
    return replace(config, network=network, training=training, replay=replay)


def _buffer(config: AppConfig, count: int = 4) -> ReplayBuffer:
    specification = NetworkSpecification.from_config(config.rules, config.network)
    buffer = ReplayBuffer(config.rules, specification, config.replay.capacity_positions)
    environment = GameEnvironment(config.rules)
    samples: list[ReplaySample] = []
    for _ in range(count):
        state = environment.state
        legal_count = sum(state.legal_mask)
        samples.append(
            ReplaySample.create(
                state,
                tuple(
                    1.0 / legal_count if legal else 0.0 for legal in state.legal_mask
                ),
                win=1.0 if state.to_play is Player.BLACK else 0.0,
                black_score=100,
                white_score=10,
            )
        )
        environment.step(environment.legal_moves()[0])
    buffer.extend(samples)
    return buffer


def _model(config: AppConfig) -> tuple[NetworkSpecification, PolicyValueNetwork]:
    specification = NetworkSpecification.from_config(config.rules, config.network)
    return specification, PolicyValueNetwork(specification)


def test_warmup_cosine_schedule_has_exact_boundaries() -> None:
    """Warmup reaches the base LR before cosine decays and then holds minimum."""

    config = _small_config(checkpoint_interval=0).training

    assert warmup_cosine_multiplier(0, config) == pytest.approx(0.5)
    assert warmup_cosine_multiplier(1, config) == pytest.approx(1.0)
    assert warmup_cosine_multiplier(2, config) == pytest.approx(1.0)
    assert warmup_cosine_multiplier(4, config) == pytest.approx(0.55)
    assert warmup_cosine_multiplier(6, config) == pytest.approx(0.1)
    assert warmup_cosine_multiplier(100, config) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="nonnegative"):
        warmup_cosine_multiplier(-1, config)


def test_scheduler_restart_schedule_has_exact_boundaries() -> None:
    """Recovery warmup starts at the floor and decays back to it exactly."""

    restart = replace(
        _small_config(checkpoint_interval=0).training.scheduler_restart,
        enabled=True,
        checkpoint_id="iteration-000001",
        initial_learning_rate=0.0001,
        peak_learning_rate=0.001,
        warmup_steps=2,
        decay_steps=4,
        minimum_learning_rate=0.0001,
    )

    assert restart_cosine_multiplier(0, restart) == pytest.approx(0.1)
    assert restart_cosine_multiplier(1, restart) == pytest.approx(0.55)
    assert restart_cosine_multiplier(2, restart) == pytest.approx(1.0)
    assert restart_cosine_multiplier(4, restart) == pytest.approx(0.55)
    assert restart_cosine_multiplier(6, restart) == pytest.approx(0.1)
    assert restart_cosine_multiplier(100, restart) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="nonnegative"):
        restart_cosine_multiplier(-1, restart)


def test_training_iteration_updates_model_metrics_and_checkpoint(
    tmp_path: Path,
) -> None:
    """A fixed-budget CPU iteration updates parameters and publishes full state."""

    config = _small_config(checkpoint_interval=1, batches_per_iteration=3)
    specification, model = _model(config)
    manager = CheckpointManager(tmp_path / "checkpoints", specification)
    sink = RecordingMetricSink()
    callbacks: list[TrainingBatchMetrics] = []
    before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    trainer = Trainer(
        config,
        specification,
        model,
        device="cpu",
        checkpoint_manager=manager,
        metric_sink=sink,
    )

    metrics = trainer.train_iteration(_buffer(config), batch_callback=callbacks.append)

    assert metrics.iteration == 1
    assert metrics.batches == 3
    assert metrics.positions == 6
    assert metrics.starting_global_step == 0
    assert metrics.completed_global_step == 3
    assert metrics.total_loss == pytest.approx(
        metrics.policy_loss
        + metrics.win_loss
        + metrics.black_score_loss
        + metrics.white_score_loss
    )
    assert metrics.maximum_gradient_norm >= metrics.mean_gradient_norm > 0.0
    assert metrics.checkpoint_id == "iteration-000001"
    assert len(callbacks) == 3
    assert [item.global_step for item in callbacks] == [1, 2, 3]
    assert any(tag == "train/batch_total_loss" for tag, _, _ in sink.scalars)
    assert any(
        tag == "train/iteration_policy_kl_divergence" for tag, _, _ in sink.scalars
    )
    assert any(
        not torch.equal(parameter, before[name])
        for name, parameter in model.named_parameters()
    )

    metadata = manager.read_metadata("latest")
    assert metadata.schema_version == CHECKPOINT_SCHEMA_VERSION == 2
    assert metadata.has_optimizer is True
    assert metadata.has_scheduler is True
    assert metadata.has_scaler is True
    assert metadata.metrics["training_steps_completed"] == 3
    trainer.close()
    assert sink.flush_count >= 2
    assert sink.closed is False


def test_checkpoint_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    """Iteration-boundary resume reproduces uninterrupted CPU model updates."""

    config = _small_config(checkpoint_interval=1)
    replay = _buffer(config)

    torch.manual_seed(101)
    continuous_specification, continuous_model = _model(config)
    continuous = Trainer(
        config,
        continuous_specification,
        continuous_model,
        device="cpu",
        checkpoint_manager=CheckpointManager(
            tmp_path / "continuous", continuous_specification
        ),
    )
    continuous.run_iterations(replay, 2)

    torch.manual_seed(101)
    interrupted_specification, interrupted_model = _model(config)
    interrupted_manager = CheckpointManager(
        tmp_path / "interrupted", interrupted_specification
    )
    interrupted = Trainer(
        config,
        interrupted_specification,
        interrupted_model,
        device="cpu",
        checkpoint_manager=interrupted_manager,
    )
    first = interrupted.train_iteration(replay)
    assert first.checkpoint_id == "iteration-000001"

    _, resumed_model = _model(config)
    resumed = Trainer(
        config,
        interrupted_specification,
        resumed_model,
        device="cpu",
        checkpoint_manager=interrupted_manager,
    )
    metadata = resumed.resume()
    assert metadata.iteration == 1
    assert resumed.completed_iteration == 1
    assert resumed.updates_completed == 2
    second = resumed.train_iteration(replay)
    assert second.iteration == 2
    assert second.starting_global_step == 2
    assert second.completed_global_step == 4

    for name, parameter in continuous.model.state_dict().items():
        assert torch.equal(parameter, resumed.model.state_dict()[name]), name
    torch.testing.assert_close(
        continuous.optimizer.state_dict(),
        resumed.optimizer.state_dict(),
        rtol=0.0,
        atol=0.0,
    )
    assert continuous.scheduler.state_dict() == resumed.scheduler.state_dict()


def test_scheduler_restart_preserves_adamw_state_and_restarts_only_lr_cycle(
    tmp_path: Path,
) -> None:
    """A targeted restart restores AdamW moments but replaces scheduler progress."""

    original = _small_config(checkpoint_interval=1, batches_per_iteration=1)
    replay = _buffer(original)
    specification, model = _model(original)
    manager = CheckpointManager(tmp_path / "restart", specification)
    source = Trainer(
        original,
        specification,
        model,
        device="cpu",
        checkpoint_manager=manager,
    )
    first = source.train_iteration(replay)
    assert first.checkpoint_id == "iteration-000001"
    stored_optimizer = source.optimizer.state_dict()
    stored_scheduler = source.scheduler.state_dict()

    restart_config = replace(
        original,
        training=replace(
            original.training,
            scheduler_restart=replace(
                original.training.scheduler_restart,
                enabled=True,
                checkpoint_id="iteration-000001",
            ),
        ),
    )
    _, resumed_model = _model(restart_config)
    resumed = Trainer(
        restart_config,
        specification,
        resumed_model,
        device="cpu",
        checkpoint_manager=manager,
    )

    metadata = resumed.resume("iteration-000001")

    assert metadata.iteration == 1
    assert resumed.completed_iteration == 1
    assert resumed.updates_completed == 1
    torch.testing.assert_close(
        resumed.optimizer.state_dict()["state"],
        stored_optimizer["state"],
        rtol=0.0,
        atol=0.0,
    )
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(0.0002)
    assert resumed.scheduler.last_epoch == 0
    assert resumed.scheduler.state_dict() != stored_scheduler

    second = resumed.train_iteration(replay)
    assert second.iteration == 2
    assert second.starting_global_step == 1
    assert second.ending_learning_rate == pytest.approx(0.00035)
    assert manager.read_metadata("latest").metrics["scheduler_steps_completed"] == 1

    _, twice_resumed_model = _model(restart_config)
    twice_resumed = Trainer(
        restart_config,
        specification,
        twice_resumed_model,
        device="cpu",
        checkpoint_manager=manager,
    )
    twice_resumed.resume("latest")
    assert twice_resumed.scheduler.last_epoch == 1
    assert twice_resumed.optimizer.param_groups[0]["lr"] == pytest.approx(0.00035)

    third = twice_resumed.train_iteration(replay)
    assert third.iteration == 3
    assert third.starting_global_step == 2
    assert third.ending_learning_rate == pytest.approx(0.0005)


def test_scheduler_restart_accepts_a_legacy_checkpoint_without_restart_config(
    tmp_path: Path,
) -> None:
    """Pre-feature checkpoints can opt into the configured recovery curve."""

    config = _small_config(checkpoint_interval=1, batches_per_iteration=1)
    replay = _buffer(config)
    specification, model = _model(config)
    manager = CheckpointManager(tmp_path / "legacy", specification)
    source = Trainer(
        config,
        specification,
        model,
        device="cpu",
        checkpoint_manager=manager,
    )
    source.train_iteration(replay)
    manifest_path = (
        tmp_path / "legacy" / "bundles" / "iteration-000001" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["configuration"]["training"].pop("scheduler_restart")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    restart_config = replace(
        config,
        training=replace(
            config.training,
            scheduler_restart=replace(
                config.training.scheduler_restart,
                enabled=True,
                checkpoint_id="iteration-000001",
            ),
        ),
    )
    _, resumed_model = _model(restart_config)
    resumed = Trainer(
        restart_config,
        specification,
        resumed_model,
        device="cpu",
        checkpoint_manager=manager,
    )

    resumed.resume("iteration-000001")

    assert resumed.completed_iteration == 1
    assert resumed.scheduler.last_epoch == 0
    assert resumed.optimizer.param_groups[0]["lr"] == pytest.approx(0.0002)


def test_resume_accepts_a_legacy_checkpoint_without_symmetry_augmentation(
    tmp_path: Path,
) -> None:
    """Pre-D4 checkpoints can resume with the active augmentation policy."""

    config = _small_config(checkpoint_interval=1, batches_per_iteration=1)
    replay = _buffer(config)
    specification, model = _model(config)
    manager = CheckpointManager(tmp_path / "legacy-augmentation", specification)
    source = Trainer(
        config,
        specification,
        model,
        device="cpu",
        checkpoint_manager=manager,
    )
    source.train_iteration(replay)

    manifest_path = (
        tmp_path
        / "legacy-augmentation"
        / "bundles"
        / "iteration-000001"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["configuration"]["training"].pop("symmetry_augmentation")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _, resumed_model = _model(config)
    resumed = Trainer(
        config,
        specification,
        resumed_model,
        device="cpu",
        checkpoint_manager=manager,
    )

    metadata = resumed.resume("iteration-000001")

    assert metadata.iteration == 1
    assert resumed.completed_iteration == 1
    assert resumed.updates_completed == 1


def test_scheduler_restart_rejects_a_different_checkpoint(tmp_path: Path) -> None:
    """A one-shot restart cannot silently apply to a later checkpoint."""

    config = _small_config(checkpoint_interval=1, batches_per_iteration=1)
    replay = _buffer(config)
    specification, model = _model(config)
    manager = CheckpointManager(tmp_path / "mismatch", specification)
    source = Trainer(
        config,
        specification,
        model,
        device="cpu",
        checkpoint_manager=manager,
    )
    source.run_iterations(replay, 2)
    restart_config = replace(
        config,
        training=replace(
            config.training,
            scheduler_restart=replace(
                config.training.scheduler_restart,
                enabled=True,
                checkpoint_id="iteration-000001",
            ),
        ),
    )
    _, resumed_model = _model(restart_config)
    resumed = Trainer(
        restart_config,
        specification,
        resumed_model,
        device="cpu",
        checkpoint_manager=manager,
    )

    with pytest.raises(TrainerStateError, match="does not match"):
        resumed.resume("iteration-000002")


def test_scheduler_restart_rejects_changed_recovery_curve(tmp_path: Path) -> None:
    """The checkpoint identity toggle cannot smuggle in different LR parameters."""

    config = _small_config(checkpoint_interval=1, batches_per_iteration=1)
    replay = _buffer(config)
    specification, model = _model(config)
    manager = CheckpointManager(tmp_path / "curve", specification)
    source = Trainer(
        config,
        specification,
        model,
        device="cpu",
        checkpoint_manager=manager,
    )
    source.train_iteration(replay)
    changed_curve = replace(
        config,
        training=replace(
            config.training,
            scheduler_restart=replace(
                config.training.scheduler_restart,
                enabled=True,
                checkpoint_id="iteration-000001",
                peak_learning_rate=0.0006,
            ),
        ),
    )
    _, resumed_model = _model(changed_curve)
    resumed = Trainer(
        changed_curve,
        specification,
        resumed_model,
        device="cpu",
        checkpoint_manager=manager,
    )

    with pytest.raises(TrainerStateError, match="may change only"):
        resumed.resume("iteration-000001")


def test_trainer_rejects_insufficient_replay_and_invalid_iteration(
    tmp_path: Path,
) -> None:
    """Training cannot start with undersized data or skip iteration boundaries."""

    config = _small_config(checkpoint_interval=1)
    specification, model = _model(config)
    trainer = Trainer(
        config,
        specification,
        model,
        device="cpu",
        checkpoint_manager=CheckpointManager(tmp_path, specification),
    )

    with pytest.raises(TrainerStateError, match="minimum"):
        trainer.train_iteration(_buffer(config, count=1))
    with pytest.raises(TrainerStateError, match="exactly one"):
        trainer.train_iteration(_buffer(config), iteration=2)
    trainer.close()
    with pytest.raises(TrainerStateError, match="closed"):
        trainer.train_iteration(_buffer(config))


def test_tensorboard_sink_writes_real_event_file(tmp_path: Path) -> None:
    """Default TensorBoard integration writes and flushes standard event data."""

    config = _small_config(
        checkpoint_interval=0,
        tensorboard_enabled=True,
        batches_per_iteration=1,
    )
    specification, model = _model(config)
    directory = tmp_path / "tensorboard"
    trainer = Trainer(
        config,
        specification,
        model,
        device="cpu",
        tensorboard_directory=directory,
    )

    trainer.train_iteration(_buffer(config))
    trainer.close()

    event_files = tuple(directory.glob("events.out.tfevents.*"))
    assert len(event_files) == 1
    assert event_files[0].stat().st_size > 0
    events = EventAccumulator(str(directory)).Reload()
    assert "train/batch_total_loss" in events.Tags()["scalars"]
    assert "train/iteration_total_loss" in events.Tags()["scalars"]
    assert "train/iteration_policy_kl_divergence" in events.Tags()["scalars"]
    assert len(events.Scalars("train/batch_total_loss")) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_amp_iteration_saves_scaler_state(tmp_path: Path) -> None:
    """CUDA training activates FP16 autocast and checkpoints GradScaler state."""

    config = _small_config(checkpoint_interval=1, batches_per_iteration=1)
    specification, model = _model(config)
    manager = CheckpointManager(tmp_path, specification)
    trainer = Trainer(
        config,
        specification,
        model,
        device="cuda",
        checkpoint_manager=manager,
    )

    metrics = trainer.train_iteration(_buffer(config))

    assert trainer.amp_enabled is True
    assert trainer.scaler.is_enabled() is True
    assert metrics.amp_enabled is True
    assert metrics.device.startswith("cuda")
    assert manager.read_metadata("latest").has_scaler is True

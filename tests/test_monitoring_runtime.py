"""Focused orchestration tests for the monitoring training runtime."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

import monitoring.runtime as runtime_module
from config import load_config
from config.schema import AppConfig
from game import GameEnvironment
from monitoring.runtime import TrainingRuntime
from network import NetworkSpecification, PolicyValueNetwork
from training import TrainingBatchMetrics, TrainingIterationMetrics


class _Replay:
    """Minimal replay double for one controller iteration."""

    total_positions_added = 3

    def __len__(self) -> int:
        return 3


class _Trainer:
    """Minimal trainer double preserving callback and metric contracts."""

    completed_iteration = 0
    updates_completed = 0
    model = object()

    def close(self) -> None:
        return None

    def train_iteration(
        self,
        replay: _Replay,
        *,
        batch_callback: Callable[[TrainingBatchMetrics], None],
    ) -> TrainingIterationMetrics:
        del replay
        batch_callback(
            TrainingBatchMetrics(
                iteration=1,
                batch_index=1,
                global_step=1,
                positions=3,
                total_loss=1.0,
                policy_loss=0.4,
                win_loss=0.2,
                black_score_loss=0.2,
                white_score_loss=0.2,
                policy_entropy=1.5,
                target_policy_entropy=1.2,
                policy_kl_divergence=0.3,
                learning_rate=0.001,
                gradient_norm=0.5,
                amp_retries=0,
                elapsed_seconds=0.01,
            )
        )
        self.completed_iteration = 1
        self.updates_completed = 1
        return TrainingIterationMetrics(
            iteration=1,
            batches=1,
            positions=3,
            starting_global_step=0,
            completed_global_step=1,
            total_loss=1.0,
            policy_loss=0.4,
            win_loss=0.2,
            black_score_loss=0.2,
            white_score_loss=0.2,
            policy_entropy=1.5,
            target_policy_entropy=1.2,
            policy_kl_divergence=0.3,
            starting_learning_rate=0.001,
            ending_learning_rate=0.001,
            mean_gradient_norm=0.5,
            maximum_gradient_norm=0.5,
            amp_retries=0,
            elapsed_seconds=0.02,
            amp_enabled=False,
            device="cpu",
        )


class _SelfPlayBatch:
    """Self-play result double containing monitoring-compatible statistics."""

    model_identifier = "fresh"

    class _Game:
        plies = 4

    games = (_Game(), _Game())
    total_simulations = 16
    inference_batches = 2
    inference_positions = 8
    maximum_inference_batch_size = 4
    elapsed_seconds = 0.03
    worker_processes = 2

    def commit(self, replay: _Replay) -> int:
        del replay
        return 3


class _SelfPlayCoordinator:
    """Coordinator double avoiding worker processes in unit tests."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def run(self, **kwargs: object) -> _SelfPlayBatch:
        del kwargs
        return _SelfPlayBatch()


class _Evaluator:
    """Inference double preventing construction of a real neural evaluator."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs


def _config(tmp_path: Path) -> AppConfig:
    config = load_config()
    paths = replace(
        config.paths,
        run_directory=tmp_path / "runs",
        checkpoint_directory=tmp_path / "checkpoints",
        log_directory=tmp_path / "logs",
        evaluation_directory=tmp_path / "evaluation-results",
    )
    replay = replace(config.replay, persistence_enabled=False)
    monitoring = replace(
        config.monitoring,
        training_device="cpu",
        auto_resume=False,
        evaluation_enabled=False,
        telemetry_interval_seconds=60.0,
    )
    return replace(config, paths=paths, replay=replay, monitoring=monitoring)


def test_runtime_coordinates_self_play_replay_and_training_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One controller iteration preserves the ordered monitoring event stream."""

    monkeypatch.setattr(runtime_module, "TorchBatchEvaluator", _Evaluator)
    monkeypatch.setattr(runtime_module, "SelfPlayCoordinator", _SelfPlayCoordinator)
    runtime = TrainingRuntime(_config(tmp_path), project_root=tmp_path)
    runtime._trainer = _Trainer()  # type: ignore[assignment]
    runtime._replay = _Replay()  # type: ignore[assignment]

    try:
        runtime._run_iteration()
        events = runtime.events.read_after(0, 20).events
        status = runtime.status()
    finally:
        runtime.close()

    assert [event.event_type for event in events] == [
        "runtime_state",
        "runtime_state",
        "self_play_batch",
        "runtime_state",
        "training_batch",
        "training_iteration",
        "runtime_state",
    ]
    replay_status = cast(dict[str, object], status["replay"])
    self_play_status = cast(dict[str, object], status["self_play"])
    configuration = cast(dict[str, object], status["configuration"])
    scheduler_restart = cast(dict[str, object], configuration["scheduler_restart"])
    assert replay_status["size"] == 3
    assert self_play_status["games_completed"] == 2
    assert scheduler_restart["enabled"] is False
    assert scheduler_restart["checkpoint_id"] == "none"


def test_runtime_can_serve_an_interactive_move_when_training_is_inactive(
    tmp_path: Path,
) -> None:
    """An idle runtime serially owns a real configured evaluation search."""

    config = _config(tmp_path)
    network = replace(
        config.network,
        residual_blocks=1,
        channels=8,
        value_hidden_features=8,
    )
    mcts = replace(
        config.mcts,
        engine="reference",
        evaluation_simulations=2,
        max_inference_batch_size=2,
    )
    monitoring = replace(
        config.monitoring,
        interactive_move_simulations_default=2,
        interactive_move_simulations_minimum=2,
        interactive_move_simulations_maximum=2,
        interactive_move_simulations_step=1,
    )
    training = replace(config.training, tensorboard_enabled=False)
    runtime = TrainingRuntime(
        replace(
            config,
            network=network,
            mcts=mcts,
            training=training,
            monitoring=monitoring,
        ),
        project_root=tmp_path,
    )

    try:
        result = runtime.select_interactive_move(GameEnvironment(config.rules).state)
        status = runtime.status()
    finally:
        runtime.close()

    assert result.action in range(config.rules.action_size)
    assert result.simulations == 2
    assert result.inference_positions >= 1
    runtime_status = cast(dict[str, object], status["runtime"])
    assert runtime_status["interactive_search_in_progress"] is False


def test_manual_checkpoint_load_ignores_training_profile_for_play(
    tmp_path: Path,
) -> None:
    """Interactive loading restores model weights without resuming training state."""

    config = _config(tmp_path)
    network = replace(
        config.network,
        residual_blocks=1,
        channels=8,
        value_hidden_features=8,
    )
    mcts = replace(
        config.mcts,
        engine="reference",
        evaluation_simulations=2,
        max_inference_batch_size=2,
    )
    monitoring = replace(
        config.monitoring,
        interactive_move_simulations_default=2,
        interactive_move_simulations_minimum=2,
        interactive_move_simulations_maximum=2,
        interactive_move_simulations_step=1,
    )
    active = replace(
        config,
        network=network,
        mcts=mcts,
        monitoring=monitoring,
        training=replace(config.training, tensorboard_enabled=False),
    )
    runtime = TrainingRuntime(active, project_root=tmp_path)
    specification = NetworkSpecification.from_config(active.rules, active.network)
    checkpoint_config = replace(
        active,
        training=replace(active.training, symmetry_augmentation="none"),
    )
    runtime.checkpoint_manager.save(
        "iteration-000042",
        model=PolicyValueNetwork(specification),
        optimizer=None,
        scheduler=None,
        iteration=42,
        config=checkpoint_config,
        metrics={},
    )

    try:
        loaded = runtime.load_checkpoint("iteration-000042")
        status = runtime.status()
        move = runtime.select_interactive_move(GameEnvironment(active.rules).state)
        start = runtime.start()
    finally:
        runtime.close()

    assert loaded.accepted is True
    assert loaded.message == "Checkpoint loaded for interactive play"
    assert status["model"]["active_identifier"] == "iteration-000042"  # type: ignore[index]
    assert status["model"]["inference_only"] is True  # type: ignore[index]
    assert status["runtime"]["completed_iteration"] == 42  # type: ignore[index]
    assert move.model_identifier == "iteration-000042"
    assert start.accepted is False
    assert "loaded for play only" in start.message


def test_interactive_move_can_use_a_request_scoped_checkpoint(
    tmp_path: Path,
) -> None:
    """A selected neural participant loads without replacing runtime state."""

    config = _config(tmp_path)
    network = replace(
        config.network,
        residual_blocks=1,
        channels=8,
        value_hidden_features=8,
    )
    mcts = replace(
        config.mcts,
        engine="reference",
        evaluation_simulations=2,
        max_inference_batch_size=2,
    )
    monitoring = replace(
        config.monitoring,
        interactive_move_simulations_default=2,
        interactive_move_simulations_minimum=2,
        interactive_move_simulations_maximum=2,
        interactive_move_simulations_step=1,
    )
    active = replace(
        config,
        network=network,
        mcts=mcts,
        monitoring=monitoring,
        training=replace(config.training, tensorboard_enabled=False),
    )
    runtime = TrainingRuntime(active, project_root=tmp_path)
    specification = NetworkSpecification.from_config(active.rules, active.network)
    runtime.checkpoint_manager.save(
        "iteration-000123",
        model=PolicyValueNetwork(specification),
        optimizer=None,
        scheduler=None,
        iteration=123,
        config=active,
        metrics={},
    )

    try:
        result = runtime.select_interactive_move(
            GameEnvironment(active.rules).state,
            2,
            "iteration-000123",
        )
        status = runtime.status()
    finally:
        runtime.close()

    assert result.action in range(active.rules.action_size)
    assert result.model_identifier == "iteration-000123"
    assert status["model"]["active_identifier"] == "fresh"  # type: ignore[index]
    assert status["model"]["inference_only"] is False  # type: ignore[index]


def test_old_champion_interactive_move_is_legal(tmp_path: Path) -> None:
    """The bundled legacy engine can be selected without neural inference."""

    config = _config(tmp_path)
    monitoring = replace(
        config.monitoring,
        interactive_move_simulations_default=2,
        interactive_move_simulations_minimum=2,
        interactive_move_simulations_maximum=2,
        interactive_move_simulations_step=1,
    )
    runtime = TrainingRuntime(
        replace(config, monitoring=monitoring), project_root=tmp_path
    )
    old_directory = tmp_path / "000 Arena" / "old-champion"
    old_directory.mkdir(parents=True)
    source = (
        Path(__file__).resolve().parents[1]
        / "000 Arena"
        / "old-champion"
        / "hexadeca_with_computer.py"
    )
    if not source.is_file():
        pytest.skip("the legacy old-champion source is excluded from the public release")
    (old_directory / source.name).write_bytes(source.read_bytes())
    state = GameEnvironment(config.rules).state

    try:
        result = runtime.select_interactive_move(state, 2, "old_champion")
    finally:
        runtime.close()

    assert state.legal_mask[result.action]
    assert result.model_identifier == "old_champion"
    assert result.simulations == 0
    assert result.analysis is None

"""Long-running, single-owner AlphaZero training runtime."""

from __future__ import annotations

import importlib.util
import json
import logging
import random
import subprocess
import sys
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from threading import Condition, Event, RLock, Thread
from time import monotonic
from types import ModuleType
from typing import Any

import psutil  # type: ignore[import-untyped]
import torch

from config.schema import AppConfig
from game import GameState, Player
from mcts import (
    MctsSearch,
    ScorePrediction,
    SearchMode,
    SearchResult,
    TorchBatchEvaluator,
)
from monitoring.events import EventBuffer, MonitoringEvent
from network import NetworkSpecification, PolicyValueNetwork
from training import (
    CheckpointEvaluationGate,
    CheckpointManager,
    EvaluationReportStore,
    ReplayBuffer,
    SelfPlayBatch,
    SelfPlayCoordinator,
    SelfPlayProgress,
    SqliteReplayStore,
    Trainer,
    TrainingBatchMetrics,
    TrainingIterationMetrics,
)


class RuntimeState(Enum):
    """Externally visible lifecycle states for cooperative runtime control."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSE_REQUESTED = "pause_requested"
    PAUSED = "paused"
    STOP_REQUESTED = "stop_requested"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RuntimeCommandResult:
    """Result of a state-transition command issued through the web service."""

    accepted: bool
    state: RuntimeState
    message: str

    def to_dict(self) -> dict[str, object]:
        """Return the stable JSON response body for command endpoints."""

        return {
            "accepted": self.accepted,
            "state": self.state.value,
            "message": self.message,
        }


class InteractiveSearchUnavailableError(RuntimeError):
    """Raised when the model is unavailable for an interactive AI move."""


@dataclass(frozen=True, slots=True)
class InteractiveMoveResult:
    """One configured evaluation search result for the manual-play endpoint."""

    action: int
    board_size: int
    simulations: int
    root_value: float
    inference_batches: int
    inference_positions: int
    maximum_inference_batch_size: int
    elapsed_seconds: float
    model_identifier: str
    analysis: InteractiveAnalysisResult | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the stable public response for one selected placement."""

        payload: dict[str, object] = {
            "action": self.action,
            "row": self.action // self.board_size,
            "column": self.action % self.board_size,
            "simulations": self.simulations,
            "root_value": self.root_value,
            "inference_batches": self.inference_batches,
            "inference_positions": self.inference_positions,
            "maximum_inference_batch_size": self.maximum_inference_batch_size,
            "elapsed_seconds": self.elapsed_seconds,
            "model_identifier": self.model_identifier,
        }
        if self.analysis is not None:
            payload["analysis"] = self.analysis.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class AnalysisCandidate:
    """One legal root action ranked by the completed MCTS search."""

    action: int
    row: int
    column: int
    player: int
    visits: int
    prior: float
    probability: float
    q_value: float
    win_probability: float
    solved: bool = False
    solved_outcome: int | None = None

    def to_dict(self) -> dict[str, object]:
        """Return the stable browser analysis representation."""

        return {
            "action": self.action,
            "row": self.row,
            "column": self.column,
            "player": self.player,
            "visits": self.visits,
            "prior": self.prior,
            "probability": self.probability,
            "q_value": self.q_value,
            "win_probability": self.win_probability,
            "solved": self.solved,
            "solved_outcome": self.solved_outcome,
        }


@dataclass(frozen=True, slots=True)
class AnalysisBranchStep:
    """One predicted placement in the principal variation."""

    action: int
    row: int
    column: int
    player: int
    visits: int
    q_value: float
    win_probability: float

    def to_dict(self) -> dict[str, object]:
        """Return the stable browser branch representation."""

        return {
            "action": self.action,
            "row": self.row,
            "column": self.column,
            "player": self.player,
            "visits": self.visits,
            "q_value": self.q_value,
            "win_probability": self.win_probability,
        }


@dataclass(frozen=True, slots=True)
class InteractiveAnalysisResult:
    """Read-only MCTS diagnostics for one interactive game position."""

    current_player: int
    simulations: int
    root_value: float
    win_probability: float
    black_win_probability: float
    predicted_black_score: float
    predicted_white_score: float
    predicted_score_margin: float
    candidates: tuple[AnalysisCandidate, ...]
    principal_variation: tuple[AnalysisBranchStep, ...]
    elapsed_seconds: float
    model_identifier: str
    solved: bool = False
    solved_outcome: int | None = None
    exact_states_evaluated: int = 0

    def to_dict(self) -> dict[str, object]:
        """Return one versioned JSON-compatible analysis payload."""

        return {
            "schema_version": 1,
            "current_player": self.current_player,
            "simulations": self.simulations,
            "root_value": self.root_value,
            "win_probability": self.win_probability,
            # Keep legacy keys for clients that have not migrated.
            "blue_win_probability": self.black_win_probability,
            "predicted_blue_score": self.predicted_black_score,
            "predicted_orange_score": self.predicted_white_score,
            "black_win_probability": self.black_win_probability,
            "predicted_black_score": self.predicted_black_score,
            "predicted_white_score": self.predicted_white_score,
            "predicted_score_margin": self.predicted_score_margin,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "principal_variation": [
                step.to_dict() for step in self.principal_variation
            ],
            "elapsed_seconds": self.elapsed_seconds,
            "model_identifier": self.model_identifier,
            "solved": self.solved,
            "solved_outcome": self.solved_outcome,
            "exact_states_evaluated": self.exact_states_evaluated,
        }


class TrainingRuntime:
    """Coordinate self-play, replay, training, evaluation, and monitoring.

    The runtime owns one trainable model and invokes all model operations from
    its dedicated training thread. Self-play workers remain CPU-only and route
    inference through ``SelfPlayCoordinator``'s centralized evaluator, so model
    updates never overlap with search inference.
    """

    __slots__ = (
        "_active_model_identifier",
        "_checkpoint_manager",
        "_condition",
        "_config",
        "_current_game",
        "_current_phase",
        "_device",
        "_events",
        "_inference_only_checkpoint",
        "_interactive_models",
        "_interactive_search_in_progress",
        "_iteration_metrics_path",
        "_last_batch_metrics",
        "_last_error",
        "_last_iteration_metrics",
        "_last_self_play",
        "_last_telemetry",
        "_loaded_checkpoint_iteration",
        "_logger",
        "_model_identifier",
        "_old_champion_computer",
        "_old_champion_module",
        "_pause_requested",
        "_project_root",
        "_replay",
        "_replay_path",
        "_replay_store",
        "_shutdown_event",
        "_specification",
        "_started_at",
        "_state",
        "_stop_requested",
        "_telemetry_thread",
        "_thread",
        "_total_self_play_games",
        "_trainer",
        "_training_thread_lock",
    )

    def __init__(
        self,
        config: AppConfig,
        *,
        project_root: Path,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._project_root = project_root.resolve()
        self._logger = logger or logging.getLogger("hexadeca.monitoring")
        self._device = _resolve_device(config.monitoring.training_device)
        specification = NetworkSpecification.from_config(config.rules, config.network)
        self._specification = specification
        checkpoint_directory = self._resolve_path(config.paths.checkpoint_directory)
        self._checkpoint_manager = CheckpointManager(
            checkpoint_directory, specification
        )
        self._replay_path = (
            self._resolve_path(config.paths.run_directory)
            / config.replay.persistence_file_name
        )
        self._iteration_metrics_path = (
            self._resolve_path(config.paths.run_directory) / "training_iterations.jsonl"
        )
        self._replay_store = (
            SqliteReplayStore.from_config(config.replay)
            if config.replay.persistence_enabled
            else None
        )
        self._events = EventBuffer(config.monitoring.event_buffer_capacity)
        self._condition = Condition(RLock())
        self._training_thread_lock = RLock()
        self._interactive_search_in_progress = False
        self._interactive_models: OrderedDict[str, PolicyValueNetwork] = OrderedDict()
        self._old_champion_module: ModuleType | None = None
        self._old_champion_computer: Any | None = None
        self._inference_only_checkpoint = False
        self._loaded_checkpoint_iteration: int | None = None
        self._state = RuntimeState.IDLE
        self._current_phase = "idle"
        self._pause_requested = False
        self._stop_requested = False
        self._shutdown_event = Event()
        self._thread: Thread | None = None
        self._telemetry_thread: Thread | None = None
        self._trainer: Trainer | None = None
        self._replay: ReplayBuffer | None = None
        self._active_model_identifier = "fresh"
        self._model_identifier = "fresh"
        self._last_batch_metrics: TrainingBatchMetrics | None = None
        self._last_iteration_metrics: TrainingIterationMetrics | None = None
        self._last_self_play: SelfPlayBatch | None = None
        self._current_game: dict[str, object] | None = None
        self._last_telemetry: dict[str, object] = {}
        self._last_error: str | None = None
        self._total_self_play_games = 0
        self._started_at = monotonic()
        self._publish_state("initialized")
        self._start_telemetry()

    @property
    def events(self) -> EventBuffer:
        """Return the recovery-capable event history owned by this runtime."""

        return self._events

    @property
    def checkpoint_manager(self) -> CheckpointManager:
        """Return the immutable-checkpoint manager for read-only web views."""

        return self._checkpoint_manager

    def validate_agent_identifier(self, identifier: str) -> None:
        """Validate one externally selected automatic participant."""

        self._validate_agent_identifier(identifier)

    def validate_neural_identifier(self, identifier: str) -> None:
        """Validate one externally selected neural evaluator."""

        self._validate_neural_identifier(identifier)

    def start(self) -> RuntimeCommandResult:
        """Start the background loop or resume it after a cooperative pause."""

        with self._condition:
            if self._interactive_search_in_progress:
                result = RuntimeCommandResult(
                    False,
                    self._state,
                    "An interactive AI search is still using the model",
                )
            elif self._inference_only_checkpoint:
                result = RuntimeCommandResult(
                    False,
                    self._state,
                    (
                        "The selected checkpoint was loaded for play only; "
                        "restart the service with its matching training profile "
                        "to continue training"
                    ),
                )
            elif self._thread is not None and self._thread.is_alive():
                if self._state is RuntimeState.PAUSED:
                    self._pause_requested = False
                    self._state = RuntimeState.RUNNING
                    self._current_phase = "resuming"
                    self._condition.notify_all()
                    result = RuntimeCommandResult(True, self._state, "Training resumed")
                else:
                    result = RuntimeCommandResult(
                        False, self._state, "Training runtime is already active"
                    )
            elif self._state is RuntimeState.COMPLETED:
                result = RuntimeCommandResult(
                    False,
                    self._state,
                    "Configured iteration limit has already been reached",
                )
            else:
                self._stop_requested = False
                self._pause_requested = False
                self._last_error = None
                self._state = RuntimeState.RUNNING
                self._current_phase = "initializing"
                self._thread = Thread(
                    target=self._run_loop,
                    name="hexadeca-training-runtime",
                    daemon=False,
                )
                self._thread.start()
                result = RuntimeCommandResult(True, self._state, "Training started")
        self._publish_state("command")
        return result

    def request_pause(self) -> RuntimeCommandResult:
        """Request a pause that takes effect after the current iteration boundary."""

        with self._condition:
            if self._state is not RuntimeState.RUNNING:
                return RuntimeCommandResult(
                    False, self._state, "Training is not running"
                )
            self._pause_requested = True
            self._state = RuntimeState.PAUSE_REQUESTED
            self._current_phase = "finishing_iteration"
            result = RuntimeCommandResult(True, self._state, "Pause requested")
        self._publish_state("command")
        return result

    def resume(self) -> RuntimeCommandResult:
        """Resume a paused loop without rebuilding its model or replay state."""

        with self._condition:
            if self._state is not RuntimeState.PAUSED:
                return RuntimeCommandResult(
                    False, self._state, "Training is not paused"
                )
            self._pause_requested = False
            self._state = RuntimeState.RUNNING
            self._current_phase = "resuming"
            self._condition.notify_all()
            result = RuntimeCommandResult(True, self._state, "Training resumed")
        self._publish_state("command")
        return result

    def request_stop(self) -> RuntimeCommandResult:
        """Request a graceful stop after the current self-play/train boundary."""

        with self._condition:
            if self._thread is None or not self._thread.is_alive():
                return RuntimeCommandResult(
                    False, self._state, "Training is not active"
                )
            self._stop_requested = True
            self._pause_requested = False
            self._state = RuntimeState.STOP_REQUESTED
            self._current_phase = "finishing_iteration"
            self._condition.notify_all()
            result = RuntimeCommandResult(True, self._state, "Stop requested")
        self._publish_state("command")
        return result

    def load_checkpoint(self, identifier: str) -> RuntimeCommandResult:
        """Load checkpoint model weights for interactive inference.

        Manual selection is intentionally independent of the active training
        profile.  Optimizer, scheduler, scaler, and RNG state remain untouched;
        automatic training resume continues to use ``Trainer.resume`` and its
        strict configuration validation.
        """

        with self._condition:
            if self._interactive_search_in_progress:
                return RuntimeCommandResult(
                    False,
                    self._state,
                    "Wait for the interactive AI search before loading a checkpoint",
                )
            if self._thread is not None and self._thread.is_alive():
                return RuntimeCommandResult(
                    False,
                    self._state,
                    "Pause and stop the runtime before loading a checkpoint",
                )
        try:
            self._close_trainer()
            self._initialize_stack(identifier, inference_only=True)
        except Exception as error:
            message = f"Checkpoint load failed: {error}"
            with self._condition:
                self._state = RuntimeState.FAILED
                self._current_phase = "failed"
                self._last_error = message
            self._publish_error(message)
            return RuntimeCommandResult(False, RuntimeState.FAILED, message)
        with self._condition:
            self._state = RuntimeState.IDLE
            self._current_phase = "idle"
            self._last_error = None
        self._publish_state("checkpoint_loaded")
        return RuntimeCommandResult(
            True, RuntimeState.IDLE, "Checkpoint loaded for interactive play"
        )

    def status(self) -> dict[str, object]:
        """Return one JSON-compatible, lock-consistent runtime snapshot."""

        with self._condition:
            trainer = self._trainer
            replay = self._replay
            current_game = (
                dict(self._current_game) if self._current_game is not None else None
            )
            last_telemetry = dict(self._last_telemetry)
            state = self._state
            phase = self._current_phase
            last_error = self._last_error
            total_self_play_games = self._total_self_play_games
            last_batch = self._last_batch_metrics
            last_iteration = self._last_iteration_metrics
            last_self_play = self._last_self_play
            model_identifier = self._model_identifier
            inference_only_checkpoint = self._inference_only_checkpoint
            loaded_checkpoint_iteration = self._loaded_checkpoint_iteration
            interactive_search_in_progress = self._interactive_search_in_progress

        replay_size = len(replay) if replay is not None else 0
        replay_capacity = self._config.replay.capacity_positions
        checkpoint_ids = self._checkpoint_manager.list_checkpoint_ids()
        latest_checkpoint = _alias_checkpoint(self._checkpoint_manager, "latest")
        best_checkpoint = _alias_checkpoint(self._checkpoint_manager, "best")
        completed_iteration = (
            loaded_checkpoint_iteration
            if inference_only_checkpoint and loaded_checkpoint_iteration is not None
            else trainer.completed_iteration
            if trainer is not None
            else 0
        )
        updates_completed = trainer.updates_completed if trainer is not None else 0
        estimated_iteration_seconds = _estimated_iteration_seconds(
            last_self_play, last_iteration
        )
        iteration_limit = self._config.monitoring.iteration_limit
        return {
            "schema_version": 1,
            "runtime": {
                "state": state.value,
                "phase": phase,
                "active_iteration": (
                    completed_iteration + 1
                    if state
                    in {
                        RuntimeState.RUNNING,
                        RuntimeState.PAUSE_REQUESTED,
                        RuntimeState.STOP_REQUESTED,
                    }
                    else None
                ),
                "completed_iteration": completed_iteration,
                "iteration_limit": iteration_limit,
                "estimated_iteration_seconds": estimated_iteration_seconds,
                "eta_seconds": (
                    (iteration_limit - completed_iteration)
                    * estimated_iteration_seconds
                    if iteration_limit > completed_iteration
                    and estimated_iteration_seconds is not None
                    else None
                ),
                "updates_completed": updates_completed,
                "uptime_seconds": monotonic() - self._started_at,
                "last_error": last_error,
                "latest_event_sequence": self._events.latest_sequence,
                "interactive_search_in_progress": interactive_search_in_progress,
            },
            "configuration": {
                "self_play_games_per_iteration": (
                    self._config.self_play.games_per_iteration
                ),
                "training_simulations": self._config.mcts.training_simulations,
                "evaluation_simulations": self._config.mcts.evaluation_simulations,
                "interactive_move_simulations_default": (
                    self._config.monitoring.interactive_move_simulations_default
                ),
                "interactive_analysis_simulations": (
                    self._config.monitoring.interactive_analysis_simulations
                ),
                "c_puct": self._config.mcts.c_puct,
                "virtual_loss": self._config.mcts.virtual_loss,
                "exact_endgame_enabled": self._config.mcts.exact_endgame_enabled,
                "exact_endgame_max_legal_moves": (
                    self._config.mcts.exact_endgame_max_legal_moves
                ),
                "evaluation_enabled": self._config.monitoring.evaluation_enabled,
                "evaluation_interval_iterations": (
                    self._config.monitoring.evaluation_interval_iterations
                ),
                "residual_blocks": self._config.network.residual_blocks,
                "channels": self._config.network.channels,
                "training_batches_per_iteration": (
                    self._config.training.batches_per_iteration
                ),
                "learning_rate": self._config.training.learning_rate,
                "scheduler": self._config.training.scheduler,
                "scheduler_warmup_steps": self._config.training.scheduler_warmup_steps,
                "scheduler_decay_steps": self._config.training.scheduler_decay_steps,
                "scheduler_minimum_learning_rate": (
                    self._config.training.scheduler_minimum_learning_rate
                ),
                "scheduler_restart": asdict(self._config.training.scheduler_restart),
            },
            "model": {
                "active_identifier": model_identifier,
                "inference_only": inference_only_checkpoint,
                "latest_checkpoint": latest_checkpoint,
                "best_checkpoint": best_checkpoint,
                "available_checkpoints": list(checkpoint_ids),
            },
            "replay": {
                "size": replay_size,
                "capacity": replay_capacity,
                "utilization": replay_size / replay_capacity,
                "total_positions_added": (
                    replay.total_positions_added if replay is not None else 0
                ),
            },
            "self_play": {
                "games_completed": total_self_play_games,
                "current_game": current_game,
                "last_batch": _self_play_payload(last_self_play)
                if last_self_play is not None
                else None,
            },
            "training": {
                "last_batch": asdict(last_batch) if last_batch is not None else None,
                "last_iteration": (
                    asdict(last_iteration) if last_iteration is not None else None
                ),
                "iteration_metrics_file": str(self._iteration_metrics_path),
            },
            "telemetry": last_telemetry,
        }

    def close(self) -> None:
        """Stop background helpers and release trainer-owned writer resources."""

        with self._condition:
            while self._interactive_search_in_progress:
                self._condition.wait()
        self.request_stop()
        self._shutdown_event.set()
        telemetry_thread = self._telemetry_thread
        if telemetry_thread is not None:
            telemetry_thread.join(self._config.monitoring.telemetry_interval_seconds)
        training_thread = self._thread
        if training_thread is not None:
            training_thread.join(self._config.self_play.worker_shutdown_timeout_seconds)
        self._close_trainer()
        for model in self._interactive_models.values():
            model.to("cpu")
        self._interactive_models.clear()
        self._old_champion_computer = None
        self._old_champion_module = None

    def select_interactive_move(
        self,
        state: GameState,
        simulations: int | None = None,
        agent_identifier: str | None = None,
    ) -> InteractiveMoveResult:
        """Run one config-driven AI move while the training loop is inactive.

        The runtime owns the model lifecycle. Reserving it here prevents a
        manual browser game from overlapping a trainer update, checkpoint load,
        or centralized self-play inference call.
        """

        if state.terminal:
            raise ValueError("Cannot select a move from a terminal position")
        move_simulations = (
            self._config.monitoring.interactive_move_simulations_default
            if simulations is None
            else simulations
        )
        self._validate_interactive_move_simulations(move_simulations)
        if agent_identifier == "human":
            raise ValueError("Human participants do not have an automatic move")

        with self._condition:
            if self._interactive_search_in_progress:
                raise InteractiveSearchUnavailableError(
                    "Another interactive AI search is already in progress"
                )
            if (
                self._thread is not None and self._thread.is_alive()
            ) or self._state not in {
                RuntimeState.IDLE,
                RuntimeState.STOPPED,
                RuntimeState.COMPLETED,
            }:
                raise InteractiveSearchUnavailableError(
                    "Interactive AI is available only while training is inactive"
                )
            self._interactive_search_in_progress = True

        try:
            if agent_identifier is not None:
                self._validate_agent_identifier(agent_identifier)
            if agent_identifier == "old_champion":
                return self._select_old_champion_move(state)
            model, model_identifier = self._interactive_model(agent_identifier)
            evaluator = TorchBatchEvaluator(
                model,
                self._specification,
                device=self._device,
                use_amp=(
                    self._config.training.amp_enabled and self._device.type == "cuda"
                ),
            )
            search_started_at = monotonic()
            score_prediction = evaluator.predict_scores((state,))[0]
            search_config = replace(
                self._config.mcts, evaluation_simulations=move_simulations
            )
            search_result = MctsSearch(
                self._config.rules, search_config, evaluator
            ).run(
                state,
                SearchMode.EVALUATION,
                random_source=random.Random(self._config.self_play.random_seed),
            )
            action = search_result.select_action(
                random.Random(self._config.self_play.random_seed)
            )
            elapsed_seconds = monotonic() - search_started_at
            analysis = _analysis_payload(
                state,
                search_result,
                score_prediction,
                elapsed_seconds,
                model_identifier,
            )
            return InteractiveMoveResult(
                action=action,
                board_size=state.board_size,
                simulations=search_result.simulations,
                root_value=search_result.root_value,
                inference_batches=search_result.inference_batches,
                inference_positions=search_result.inference_positions,
                maximum_inference_batch_size=search_result.maximum_batch_size,
                elapsed_seconds=elapsed_seconds,
                model_identifier=model_identifier,
                analysis=analysis,
            )
        finally:
            with self._condition:
                self._interactive_search_in_progress = False
                self._condition.notify_all()

    def analyze_interactive_position(
        self, state: GameState, model_identifier: str | None = None
    ) -> InteractiveAnalysisResult:
        """Run one evaluation search and return diagnostics without playing."""

        if state.terminal:
            raise ValueError("Cannot analyze a terminal position")
        if model_identifier in {"human", "old_champion"}:
            raise ValueError("Position analysis requires a neural checkpoint")
        with self._condition:
            if self._interactive_search_in_progress:
                raise InteractiveSearchUnavailableError(
                    "Another interactive AI search is already in progress"
                )
            if (
                self._thread is not None and self._thread.is_alive()
            ) or self._state not in {
                RuntimeState.IDLE,
                RuntimeState.STOPPED,
                RuntimeState.COMPLETED,
            }:
                raise InteractiveSearchUnavailableError(
                    "Interactive AI is available only while training is inactive"
                )
            self._interactive_search_in_progress = True
        try:
            model, resolved_model_identifier = self._interactive_model(model_identifier)
            evaluator = TorchBatchEvaluator(
                model,
                self._specification,
                device=self._device,
                use_amp=(
                    self._config.training.amp_enabled and self._device.type == "cuda"
                ),
            )
            started_at = monotonic()
            score_prediction = evaluator.predict_scores((state,))[0]
            search_config = replace(
                self._config.mcts,
                evaluation_simulations=(
                    self._config.monitoring.interactive_analysis_simulations
                ),
            )
            result = MctsSearch(self._config.rules, search_config, evaluator).run(
                state, SearchMode.EVALUATION
            )
            return _analysis_payload(
                state,
                result,
                score_prediction,
                monotonic() - started_at,
                resolved_model_identifier,
            )
        finally:
            with self._condition:
                self._interactive_search_in_progress = False
                self._condition.notify_all()

    def _validate_interactive_move_simulations(self, simulations: int) -> None:
        """Reject browser move budgets outside the configured slider lattice."""

        monitoring = self._config.monitoring
        minimum = monitoring.interactive_move_simulations_minimum
        maximum = monitoring.interactive_move_simulations_maximum
        step = monitoring.interactive_move_simulations_step
        if type(simulations) is not int or not minimum <= simulations <= maximum:
            raise ValueError(
                f"Interactive move simulations must be between {minimum} and {maximum}"
            )
        if (simulations - minimum) % step:
            raise ValueError(
                f"Interactive move simulations must use increments of {step}"
            )

    def _interactive_model(
        self, identifier: str | None
    ) -> tuple[PolicyValueNetwork, str]:
        """Return the active model or an isolated cached checkpoint model."""

        if identifier is None:
            self._initialize_stack(None)
            trainer = self._required_trainer()
            with self._condition:
                return trainer.model, self._model_identifier

        self._validate_neural_identifier(identifier)
        cached = self._interactive_models.get(identifier)
        if cached is None:
            cached = PolicyValueNetwork(self._specification)
            metadata = self._checkpoint_manager.load(
                identifier,
                model=cached,
                restore_rng=False,
                map_location="cpu",
            )
            cached.eval()
            identifier = metadata.checkpoint_id
            self._interactive_models[identifier] = cached
            while len(self._interactive_models) > 3:
                _, evicted = self._interactive_models.popitem(last=False)
                evicted.to("cpu")
        else:
            self._interactive_models.move_to_end(identifier)

        if self._device.type == "cuda":
            for cached_identifier, cached_model in self._interactive_models.items():
                if cached_identifier != identifier:
                    cached_model.to("cpu")
        return cached, identifier

    def _validate_agent_identifier(self, identifier: str) -> None:
        """Reject non-playable participant identifiers before doing any search."""

        if identifier == "old_champion":
            return
        if identifier == "human":
            raise ValueError("Human participants do not have an automatic move")
        self._validate_neural_identifier(identifier)

    def _validate_neural_identifier(self, identifier: str) -> None:
        """Require an immutable locally available neural checkpoint ID."""

        if identifier not in self._checkpoint_manager.list_checkpoint_ids():
            raise ValueError(f"Unknown neural checkpoint: {identifier}")

    def _select_old_champion_move(self, state: GameState) -> InteractiveMoveResult:
        """Replay a canonical state into and query the legacy champion engine."""

        started_at = monotonic()
        old_module, old_computer = self._old_champion()
        old_state = old_module.GameState()
        for move in state.history:
            color = (
                old_module.BLACK if move.player is Player.BLACK else old_module.WHITE
            )
            old_state.place(move.action, color)
        canonical_legal = tuple(
            action for action, legal in enumerate(state.legal_mask) if legal
        )
        if tuple(old_state.legal_moves()) != canonical_legal:
            raise RuntimeError("Old champion and canonical legal moves disagree")
        color = old_module.BLACK if state.to_play is Player.BLACK else old_module.WHITE
        seed = self._config.self_play.random_seed ^ state.zobrist_hash
        evaluation = old_computer.choose_move(old_state, color, random.Random(seed))
        if evaluation is None:
            raise RuntimeError("Old champion returned no move in a live position")
        action = int(evaluation.move)
        if action not in canonical_legal:
            raise RuntimeError("Old champion returned an illegal move")
        return InteractiveMoveResult(
            action=action,
            board_size=state.board_size,
            simulations=0,
            root_value=0.0,
            inference_batches=0,
            inference_positions=0,
            maximum_inference_batch_size=0,
            elapsed_seconds=monotonic() - started_at,
            model_identifier="old_champion",
            analysis=None,
        )

    def _old_champion(self) -> tuple[ModuleType, Any]:
        """Load the bundled legacy champion lazily without launching its GUI."""

        if (
            self._old_champion_module is not None
            and self._old_champion_computer is not None
        ):
            return self._old_champion_module, self._old_champion_computer
        path = (
            self._project_root
            / "000 Arena"
            / "old-champion"
            / "hexadeca_with_computer.py"
        )
        if not path.is_file():
            raise ValueError(f"Old champion is unavailable: {path}")
        module_name = "hexadeca_interactive_old_champion"
        module_spec = importlib.util.spec_from_file_location(module_name, path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"Cannot import old champion from {path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
        computer = module.LookaheadComputer(module.AIParameters())
        self._old_champion_module = module
        self._old_champion_computer = computer
        return module, computer

    def _run_loop(self) -> None:
        try:
            self._initialize_stack(None)
            while self._wait_for_iteration_permission():
                self._run_iteration()
        except Exception as error:
            message = f"Training runtime failed: {error}"
            self._logger.exception(message)
            with self._condition:
                self._state = RuntimeState.FAILED
                self._current_phase = "failed"
                self._last_error = message
            self._publish_error(message)
        else:
            with self._condition:
                if self._stop_requested:
                    self._state = RuntimeState.STOPPED
                    self._current_phase = "stopped"
                elif self._state is not RuntimeState.COMPLETED:
                    self._state = RuntimeState.IDLE
                    self._current_phase = "idle"
            self._publish_state("loop_finished")

    def _wait_for_iteration_permission(self) -> bool:
        with self._condition:
            while self._pause_requested and not self._stop_requested:
                self._state = RuntimeState.PAUSED
                self._current_phase = "paused"
                self._publish_state("paused")
                self._condition.wait()
            if self._stop_requested:
                return False
            trainer = self._required_trainer()
            limit = self._config.monitoring.iteration_limit
            if limit > 0 and trainer.completed_iteration >= limit:
                self._state = RuntimeState.COMPLETED
                self._current_phase = "completed"
                return False
            self._state = RuntimeState.RUNNING
            self._current_phase = "self_play"
            return True

    def _run_iteration(self) -> None:
        trainer = self._required_trainer()
        replay = self._required_replay()
        iteration = trainer.completed_iteration + 1
        self._publish_state("self_play_started")

        evaluator = TorchBatchEvaluator(
            trainer.model,
            self._specification,
            device=self._device,
            use_amp=(self._config.training.amp_enabled and self._device.type == "cuda"),
        )
        batch = SelfPlayCoordinator(
            self._config.rules,
            self._config.mcts,
            self._config.self_play,
            evaluator,
            model_identifier=self._active_model_identifier,
        ).run(
            master_seed=self._config.self_play.random_seed + iteration - 1,
            progress_callback=self._handle_self_play_progress,
        )
        positions_added = batch.commit(replay)
        with self._condition:
            self._last_self_play = batch
            self._total_self_play_games += len(batch.games)
            self._current_game = None
            self._current_phase = "replay"
        self._events.publish(
            "self_play_batch",
            {
                **_self_play_payload(batch),
                "positions_added": positions_added,
                "replay_size": len(replay),
            },
        )
        self._persist_replay_if_due(iteration)

        with self._condition:
            self._current_phase = "training"
        self._publish_state("training_started")
        metrics = trainer.train_iteration(
            replay, batch_callback=self._handle_batch_metrics
        )
        self._persist_iteration_metrics(iteration, metrics, batch)
        with self._condition:
            self._last_iteration_metrics = metrics
            self._current_phase = "checkpoint"
            if metrics.checkpoint_id is not None:
                self._active_model_identifier = metrics.checkpoint_id
                self._model_identifier = metrics.checkpoint_id
        self._events.publish("training_iteration", _iteration_payload(metrics))
        if metrics.checkpoint_id is not None:
            self._events.publish(
                "checkpoint",
                {
                    "checkpoint_id": metrics.checkpoint_id,
                    "iteration": metrics.iteration,
                    "latest_checkpoint": metrics.checkpoint_id,
                },
            )
            self._evaluate_checkpoint_if_due(metrics)
        self._publish_state("iteration_completed")

    def _handle_self_play_progress(self, progress: SelfPlayProgress) -> None:
        payload = _self_play_progress_payload(progress)
        with self._condition:
            self._current_game = payload
        self._events.publish("self_play_move", payload)

    def _handle_batch_metrics(self, metrics: TrainingBatchMetrics) -> None:
        with self._condition:
            self._last_batch_metrics = metrics
        self._events.publish("training_batch", _batch_payload(metrics))

    def _persist_iteration_metrics(
        self, iteration: int, metrics: TrainingIterationMetrics, batch: SelfPlayBatch
    ) -> None:
        """Append one complete, analysis-friendly record for every iteration."""

        payload = {
            "schema_version": 1,
            "recorded_at": datetime.now(UTC).isoformat(),
            "iteration": iteration,
            "training": asdict(metrics),
            "self_play": _self_play_payload(batch),
            "replay_size": len(self._required_replay()),
        }
        self._iteration_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with self._iteration_metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
            handle.flush()

    def _evaluate_checkpoint_if_due(self, metrics: TrainingIterationMetrics) -> None:
        checkpoint_id = metrics.checkpoint_id
        if checkpoint_id is None or not self._config.monitoring.evaluation_enabled:
            return
        first_best = not self._checkpoint_manager.has_alias("best")
        interval = self._config.monitoring.evaluation_interval_iterations
        if not first_best and metrics.iteration % interval != 0:
            return
        with self._condition:
            self._current_phase = "evaluation"
        self._publish_state("evaluation_started")
        outcome = CheckpointEvaluationGate(
            self._config,
            self._specification,
            self._checkpoint_manager,
            EvaluationReportStore(
                self._resolve_path(self._config.paths.evaluation_directory)
            ),
            device=self._device,
        ).evaluate(checkpoint_id)
        self._events.publish(
            "evaluation",
            {
                "decision": outcome.decision.value,
                "candidate_checkpoint_id": outcome.candidate_checkpoint_id,
                "previous_best_checkpoint_id": outcome.previous_best_checkpoint_id,
                "best_checkpoint": _alias_checkpoint(self._checkpoint_manager, "best"),
                "report_path": str(outcome.report_path),
                "candidate_points": (
                    outcome.arena.candidate_points
                    if outcome.arena is not None
                    else None
                ),
            },
        )

    def _initialize_stack(
        self, identifier: str | None, *, inference_only: bool = False
    ) -> None:
        with self._training_thread_lock:
            if self._trainer is not None:
                return
            if inference_only and identifier is None:
                raise ValueError(
                    "Inference-only checkpoint loading requires an identifier"
                )
            self._initialize_directories()
            specification = self._specification
            replay = ReplayBuffer.from_config(
                self._config.rules, specification, self._config.replay
            )
            if self._replay_store is not None and self._replay_path.is_file():
                replay = self._replay_store.load(
                    self._replay_path,
                    self._config.rules,
                    specification,
                    capacity_positions=self._config.replay.capacity_positions,
                ).buffer
            model = PolicyValueNetwork(specification)
            trainer = Trainer(
                self._config,
                specification,
                model,
                device=self._device,
                checkpoint_manager=self._checkpoint_manager,
                tensorboard_directory=self._resolve_path(
                    self._config.paths.run_directory
                )
                / "tensorboard",
            )
            resume_identifier = identifier
            if (
                resume_identifier is None
                and self._config.monitoring.auto_resume
                and self._checkpoint_manager.has_alias("latest")
            ):
                resume_identifier = "latest"
            try:
                if resume_identifier is not None:
                    metadata = (
                        self._checkpoint_manager.load(
                            resume_identifier,
                            model=trainer.model,
                            restore_rng=False,
                            map_location=self._device,
                        )
                        if inference_only
                        else trainer.resume(resume_identifier)
                    )
                    self._active_model_identifier = metadata.checkpoint_id
                    self._model_identifier = metadata.checkpoint_id
            except Exception:
                trainer.close()
                raise
            with self._condition:
                self._replay = replay
                self._trainer = trainer
                self._inference_only_checkpoint = inference_only
                self._loaded_checkpoint_iteration = (
                    metadata.iteration
                    if inference_only and resume_identifier is not None
                    else None
                )

    def _persist_replay_if_due(self, iteration: int) -> None:
        if self._replay_store is None:
            return
        if iteration % self._config.monitoring.replay_save_interval_iterations != 0:
            return
        replay = self._required_replay()
        self._replay_store.save(replay.snapshot(), self._replay_path)
        self._events.publish(
            "replay_persisted",
            {"path": str(self._replay_path), "positions": len(replay)},
        )

    def _start_telemetry(self) -> None:
        thread = Thread(
            target=self._telemetry_loop,
            name="hexadeca-monitoring-telemetry",
            daemon=True,
        )
        self._telemetry_thread = thread
        thread.start()

    def _telemetry_loop(self) -> None:
        while not self._shutdown_event.wait(
            self._config.monitoring.telemetry_interval_seconds
        ):
            telemetry = _collect_telemetry(
                self._config.monitoring.gpu_probe_timeout_seconds
            )
            with self._condition:
                self._last_telemetry = telemetry
            self._events.publish("telemetry", telemetry)

    def _publish_state(self, reason: str) -> MonitoringEvent:
        self._logger.info(
            "Runtime state published: %s", reason, extra={"event": reason}
        )
        return self._events.publish(
            "runtime_state", {"reason": reason, "status": self.status()}
        )

    def _publish_error(self, message: str) -> MonitoringEvent:
        self._logger.error(
            "Runtime error published: %s", message, extra={"event": "error"}
        )
        return self._events.publish(
            "error", {"message": message, "status": self.status()}
        )

    def _required_trainer(self) -> Trainer:
        trainer = self._trainer
        if trainer is None:
            raise RuntimeError("Training runtime has not been initialized")
        return trainer

    def _required_replay(self) -> ReplayBuffer:
        replay = self._replay
        if replay is None:
            raise RuntimeError("Training replay has not been initialized")
        return replay

    def _close_trainer(self) -> None:
        with self._training_thread_lock:
            if self._trainer is not None:
                self._trainer.close()
            self._trainer = None
            self._replay = None
            self._inference_only_checkpoint = False
            self._loaded_checkpoint_iteration = None

    def _initialize_directories(self) -> None:
        for path in (
            self._resolve_path(self._config.paths.run_directory),
            self._resolve_path(self._config.paths.checkpoint_directory),
            self._resolve_path(self._config.paths.log_directory),
            self._resolve_path(self._config.paths.evaluation_directory),
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, path: Path) -> Path:
        return path if path.is_absolute() else self._project_root / path


def _resolve_device(configured: str) -> torch.device:
    if configured == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if configured == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Monitoring configured CUDA but torch.cuda is unavailable")
    return torch.device(configured)


def _alias_checkpoint(manager: CheckpointManager, alias: str) -> str | None:
    if not manager.has_alias(alias):
        return None
    return manager.read_metadata(alias).checkpoint_id


def _state_payload(state: GameState) -> dict[str, object]:
    board_size = state.board_size
    return {
        "board": [
            list(state.cells[offset : offset + board_size])
            for offset in range(0, len(state.cells), board_size)
        ],
        "legal_mask": list(state.legal_mask),
        "legal_move_count": sum(state.legal_mask),
        "current_player": state.to_play.value,
        "ply": state.ply,
        "terminal": state.terminal,
        "zobrist_hash": state.zobrist_hash,
    }


def _analysis_payload(
    state: GameState,
    result: SearchResult,
    score_prediction: ScorePrediction,
    elapsed_seconds: float,
    model_identifier: str,
) -> InteractiveAnalysisResult:
    """Translate MCTS root statistics into a compact browser analysis model."""

    current_player = state.to_play.value
    legal_actions = [action for action, legal in enumerate(state.legal_mask) if legal]
    ranked = sorted(
        legal_actions,
        key=lambda action: (
            result.visit_counts[action],
            result.root.children[action].mean_value
            if action in result.root.children
            else float("-inf"),
            -action,
        ),
        reverse=True,
    )
    candidates: list[AnalysisCandidate] = []
    exact_action_outcomes = dict(result.exact_action_outcomes)
    for action in ranked[:5]:
        edge = result.root.children[action]
        q_value = float(edge.mean_value)
        candidates.append(
            AnalysisCandidate(
                action=action,
                row=action // state.board_size,
                column=action % state.board_size,
                player=current_player,
                visits=edge.visit_count,
                prior=float(edge.prior),
                probability=float(result.action_probabilities[action]),
                q_value=q_value,
                win_probability=(q_value + 1.0) / 2.0,
                solved=action in exact_action_outcomes,
                solved_outcome=exact_action_outcomes.get(action),
            )
        )

    principal_variation: list[AnalysisBranchStep] = []
    node = result.root
    player = current_player
    for _ in range(6):
        if not node.children:
            break
        action, edge = max(
            node.children.items(),
            key=lambda item: (item[1].visit_count, item[1].mean_value, -item[0]),
        )
        q_value = float(edge.mean_value)
        principal_variation.append(
            AnalysisBranchStep(
                action=action,
                row=action // state.board_size,
                column=action % state.board_size,
                player=player,
                visits=edge.visit_count,
                q_value=q_value,
                win_probability=(q_value + 1.0) / 2.0,
            )
        )
        if edge.child is None:
            break
        node = edge.child
        player = Player(player).opponent.value

    root_value = float(result.root_value)
    return InteractiveAnalysisResult(
        current_player=current_player,
        simulations=result.simulations,
        root_value=root_value,
        win_probability=(root_value + 1.0) / 2.0,
        black_win_probability=(root_value + 1.0) / 2.0
        if current_player == Player.BLACK.value
        else (1.0 - root_value) / 2.0,
        predicted_black_score=score_prediction.black_score,
        predicted_white_score=score_prediction.white_score,
        predicted_score_margin=score_prediction.margin,
        candidates=tuple(candidates),
        principal_variation=tuple(principal_variation),
        elapsed_seconds=elapsed_seconds,
        model_identifier=model_identifier,
        solved=result.solved_outcome is not None,
        solved_outcome=result.solved_outcome,
        exact_states_evaluated=result.exact_states_evaluated,
    )


def _self_play_progress_payload(progress: SelfPlayProgress) -> dict[str, object]:
    return {
        "game_index": progress.game_index,
        "worker_index": progress.worker_index,
        "process_id": progress.process_id,
        "model_identifier": progress.model_identifier,
        "seed": progress.seed,
        "action": progress.action,
        "row": progress.action // progress.state.board_size,
        "column": progress.action % progress.state.board_size,
        "player": progress.player.value,
        "simulations": progress.simulations,
        "root_value": progress.root_value,
        "inference_batches": progress.inference_batches,
        "inference_positions": progress.inference_positions,
        "maximum_inference_batch_size": progress.maximum_inference_batch_size,
        "search_elapsed_seconds": progress.search_elapsed_seconds,
        "state": _state_payload(progress.state),
    }


def _self_play_payload(batch: SelfPlayBatch) -> dict[str, object]:
    return {
        "model_identifier": batch.model_identifier,
        "games": len(batch.games),
        "plies": sum(game.plies for game in batch.games),
        "total_simulations": batch.total_simulations,
        "inference_batches": batch.inference_batches,
        "inference_positions": batch.inference_positions,
        "maximum_inference_batch_size": batch.maximum_inference_batch_size,
        "elapsed_seconds": batch.elapsed_seconds,
        "worker_processes": batch.worker_processes,
    }


def _batch_payload(metrics: TrainingBatchMetrics) -> dict[str, object]:
    return asdict(metrics)


def _iteration_payload(metrics: TrainingIterationMetrics) -> dict[str, object]:
    return asdict(metrics)


def _estimated_iteration_seconds(
    self_play: SelfPlayBatch | None, training: TrainingIterationMetrics | None
) -> float | None:
    if self_play is None or training is None:
        return None
    return self_play.elapsed_seconds + training.elapsed_seconds


def _collect_telemetry(gpu_probe_timeout_seconds: float) -> dict[str, object]:
    process = psutil.Process()
    memory = psutil.virtual_memory()
    telemetry: dict[str, object] = {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "process_cpu_percent": process.cpu_percent(interval=None),
        "ram_total_bytes": memory.total,
        "ram_used_bytes": memory.used,
        "ram_percent": memory.percent,
        "gpu": _gpu_telemetry(gpu_probe_timeout_seconds),
    }
    return telemetry


def _gpu_telemetry(gpu_probe_timeout_seconds: float) -> dict[str, object]:
    if not torch.cuda.is_available():
        return {"available": False, "utilization_percent": None}
    device_index = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    result: dict[str, object] = {
        "available": True,
        "device_index": device_index,
        "name": torch.cuda.get_device_name(device_index),
        "memory_total_bytes": total_bytes,
        "memory_free_bytes": free_bytes,
        "memory_used_bytes": total_bytes - free_bytes,
        "memory_allocated_bytes": torch.cuda.memory_allocated(device_index),
        "memory_reserved_bytes": torch.cuda.memory_reserved(device_index),
        "utilization_percent": None,
    }
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=gpu_probe_timeout_seconds,
        )
        row = completed.stdout.splitlines()[device_index]
        utilization, used_memory, total_memory = [
            value.strip() for value in row.split(",")
        ]
        result.update(
            {
                "utilization_percent": float(utilization),
                "nvidia_smi_memory_used_mib": float(used_memory),
                "nvidia_smi_memory_total_mib": float(total_memory),
            }
        )
    except (IndexError, OSError, subprocess.SubprocessError, ValueError):
        pass
    return result

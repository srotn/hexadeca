"""Checkpoint Arena gate and immutable versioned evaluation reports."""

from __future__ import annotations

import json
import os
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from uuid import uuid4

import torch

from config.schema import AppConfig
from mcts import TorchBatchEvaluator
from network import NetworkSpecification, PolicyValueNetwork
from training.arena import ArenaBatch, ArenaCoordinator
from training.checkpoint import CheckpointManager, CheckpointMetadata
from training.errors import EvaluationGateError, EvaluationReportError
from training.rating import ArenaStatistics, calculate_arena_statistics

EVALUATION_REPORT_SCHEMA_VERSION = 1


class EvaluationDecision(Enum):
    """Stable checkpoint gate outcomes."""

    INITIALIZED = "initialized"
    PROMOTED = "promoted"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    """One completed checkpoint gate operation."""

    decision: EvaluationDecision
    candidate_checkpoint_id: str
    previous_best_checkpoint_id: str | None
    arena: ArenaBatch | None
    statistics: ArenaStatistics | None
    report_path: Path


class EvaluationReportStore:
    """Publish immutable report bundles through an atomic directory rename."""

    __slots__ = ("_directory",)

    def __init__(self, directory: Path) -> None:
        self._directory = directory

    def publish(
        self,
        *,
        decision: EvaluationDecision,
        candidate: CheckpointMetadata,
        previous_best: CheckpointMetadata | None,
        arena: ArenaBatch | None,
        statistics: ArenaStatistics | None,
        config: AppConfig,
    ) -> Path:
        """Write one self-contained report without replacing prior evidence."""

        if (arena is None) is not (statistics is None):
            raise EvaluationReportError(
                "Arena result and statistics must both be present or absent"
            )
        _validate_report_inputs(decision, candidate, previous_best, arena, statistics)
        best_id = (
            previous_best.checkpoint_id if previous_best is not None else "initial"
        )
        report_id = f"{candidate.checkpoint_id}-vs-{best_id}"
        destination = self._directory / "bundles" / report_id
        temporary = destination.with_name(f".{report_id}.{uuid4().hex}.temporary")
        if destination.exists():
            return _validate_existing_report(
                destination / "report.json",
                decision=decision,
                candidate_id=candidate.checkpoint_id,
                best_id=(
                    previous_best.checkpoint_id if previous_best is not None else None
                ),
            )

        report = _build_report(
            decision=decision,
            candidate=candidate,
            previous_best=previous_best,
            arena=arena,
            statistics=statistics,
            config=config,
        )
        try:
            temporary.mkdir(parents=True, exist_ok=False)
            report_path = temporary / "report.json"
            with report_path.open("x", encoding="utf-8", newline="\n") as output:
                json.dump(report, output, indent=2, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.rename(temporary, destination)
        except Exception as error:
            _remove_temporary_report(temporary)
            raise EvaluationReportError(
                f"Failed to publish evaluation report: {report_id}"
            ) from error
        return destination / "report.json"


class CheckpointEvaluationGate:
    """Initialize or compare checkpoints and atomically update the best alias."""

    __slots__ = (
        "_checkpoint_manager",
        "_config",
        "_device",
        "_report_store",
        "_specification",
    )

    def __init__(
        self,
        config: AppConfig,
        specification: NetworkSpecification,
        checkpoint_manager: CheckpointManager,
        report_store: EvaluationReportStore,
        *,
        device: torch.device | str,
    ) -> None:
        configured = NetworkSpecification.from_config(config.rules, config.network)
        configured.ensure_compatible(specification)
        resolved_device = torch.device(device)
        if resolved_device.type not in {"cpu", "cuda"}:
            raise EvaluationGateError("Evaluation device must be CPU or CUDA")
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise EvaluationGateError("CUDA evaluation requested but unavailable")
        self._config = config
        self._specification = specification
        self._checkpoint_manager = checkpoint_manager
        self._report_store = report_store
        self._device = resolved_device

    def evaluate(self, candidate_identifier: str) -> EvaluationOutcome:
        """Evaluate one candidate or initialize the first valid best checkpoint."""

        candidate = self._checkpoint_manager.read_metadata(candidate_identifier)
        if not self._checkpoint_manager.has_alias("best"):
            self._load_evaluator(candidate.checkpoint_id)
            report_path = self._report_store.publish(
                decision=EvaluationDecision.INITIALIZED,
                candidate=candidate,
                previous_best=None,
                arena=None,
                statistics=None,
                config=self._config,
            )
            self._checkpoint_manager.publish_alias("best", candidate.checkpoint_id)
            return EvaluationOutcome(
                decision=EvaluationDecision.INITIALIZED,
                candidate_checkpoint_id=candidate.checkpoint_id,
                previous_best_checkpoint_id=None,
                arena=None,
                statistics=None,
                report_path=report_path,
            )

        previous_best = self._checkpoint_manager.read_metadata("best")
        if candidate.checkpoint_id == previous_best.checkpoint_id:
            raise EvaluationGateError(
                "Candidate checkpoint is already the active best checkpoint"
            )

        candidate_evaluator = self._load_evaluator(candidate.checkpoint_id)
        best_evaluator = self._load_evaluator(previous_best.checkpoint_id)
        arena = ArenaCoordinator(
            self._config.rules,
            self._config.mcts,
            self._config.evaluation,
            candidate_evaluator,
            best_evaluator,
            candidate_identifier=candidate.checkpoint_id,
            best_identifier=previous_best.checkpoint_id,
        ).run()
        statistics = calculate_arena_statistics(arena, self._config.evaluation)
        decision = (
            EvaluationDecision.PROMOTED
            if statistics.promoted
            else EvaluationDecision.REJECTED
        )
        report_path = self._report_store.publish(
            decision=decision,
            candidate=candidate,
            previous_best=previous_best,
            arena=arena,
            statistics=statistics,
            config=self._config,
        )
        if decision is EvaluationDecision.PROMOTED:
            self._checkpoint_manager.publish_alias("best", candidate.checkpoint_id)
        return EvaluationOutcome(
            decision=decision,
            candidate_checkpoint_id=candidate.checkpoint_id,
            previous_best_checkpoint_id=previous_best.checkpoint_id,
            arena=arena,
            statistics=statistics,
            report_path=report_path,
        )

    def _load_evaluator(self, checkpoint_id: str) -> TorchBatchEvaluator:
        model = PolicyValueNetwork(self._specification)
        self._checkpoint_manager.load(
            checkpoint_id,
            model=model,
            restore_rng=False,
            map_location="cpu",
        )
        return TorchBatchEvaluator(
            model,
            self._specification,
            device=self._device,
            use_amp=(
                self._config.evaluation.amp_enabled and self._device.type == "cuda"
            ),
        )


def _build_report(
    *,
    decision: EvaluationDecision,
    candidate: CheckpointMetadata,
    previous_best: CheckpointMetadata | None,
    arena: ArenaBatch | None,
    statistics: ArenaStatistics | None,
    config: AppConfig,
) -> dict[str, object]:
    return {
        "schema_version": EVALUATION_REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "decision": decision.value,
        "candidate": _checkpoint_report(candidate),
        "previous_best": (
            _checkpoint_report(previous_best) if previous_best is not None else None
        ),
        "configuration": {
            "evaluation": asdict(config.evaluation),
            "mcts": asdict(config.mcts),
        },
        "statistics": asdict(statistics) if statistics is not None else None,
        "arena": _arena_report(arena) if arena is not None else None,
    }


def _checkpoint_report(metadata: CheckpointMetadata) -> dict[str, object]:
    return {
        "checkpoint_id": metadata.checkpoint_id,
        "created_at": metadata.created_at,
        "iteration": metadata.iteration,
        "parent_checkpoint_id": metadata.parent_checkpoint_id,
        "state_sha256": metadata.state_sha256,
    }


def _validate_report_inputs(
    decision: EvaluationDecision,
    candidate: CheckpointMetadata,
    previous_best: CheckpointMetadata | None,
    arena: ArenaBatch | None,
    statistics: ArenaStatistics | None,
) -> None:
    if decision is EvaluationDecision.INITIALIZED:
        if previous_best is not None or arena is not None or statistics is not None:
            raise EvaluationReportError(
                "Initial best report cannot contain an Arena comparison"
            )
        return
    if previous_best is None or arena is None or statistics is None:
        raise EvaluationReportError(
            "Compared evaluation report requires best, Arena, and statistics"
        )
    if (
        arena.candidate_identifier != candidate.checkpoint_id
        or arena.best_identifier != previous_best.checkpoint_id
    ):
        raise EvaluationReportError(
            "Arena model identifiers disagree with checkpoint metadata"
        )
    if statistics.promoted is not (decision is EvaluationDecision.PROMOTED):
        raise EvaluationReportError(
            "Evaluation decision disagrees with promotion statistics"
        )


def _validate_existing_report(
    report_path: Path,
    *,
    decision: EvaluationDecision,
    candidate_id: str,
    best_id: str | None,
) -> Path:
    try:
        raw = json.loads(report_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise TypeError("report root is not an object")
        candidate = raw.get("candidate")
        previous_best = raw.get("previous_best")
        stored_candidate = (
            candidate.get("checkpoint_id") if isinstance(candidate, dict) else None
        )
        stored_best = (
            previous_best.get("checkpoint_id")
            if isinstance(previous_best, dict)
            else None
        )
        if (
            raw.get("schema_version") != EVALUATION_REPORT_SCHEMA_VERSION
            or raw.get("decision") != decision.value
            or stored_candidate != candidate_id
            or stored_best != best_id
        ):
            raise ValueError("report identity does not match the retry")
    except Exception as error:
        raise EvaluationReportError(
            "Existing evaluation report is invalid or belongs to another decision"
        ) from error
    return report_path


def _arena_report(arena: ArenaBatch) -> dict[str, object]:
    return {
        "schema_version": arena.schema_version,
        "candidate_identifier": arena.candidate_identifier,
        "best_identifier": arena.best_identifier,
        "random_seed": arena.random_seed,
        "worker_processes": arena.worker_processes,
        "elapsed_seconds": arena.elapsed_seconds,
        "candidate_inference_batches": arena.candidate_inference_batches,
        "candidate_inference_positions": arena.candidate_inference_positions,
        "best_inference_batches": arena.best_inference_batches,
        "best_inference_positions": arena.best_inference_positions,
        "maximum_inference_batch_size": arena.maximum_inference_batch_size,
        "total_simulations": arena.total_simulations,
        "openings": [
            {"index": opening.index, "actions": list(opening.actions)}
            for opening in arena.openings
        ],
        "games": [
            {
                "game_index": game.game_index,
                "opening_index": game.opening_index,
                "worker_index": game.worker_index,
                "process_id": game.process_id,
                "candidate_player": game.candidate_player.name.lower(),
                "seed": game.seed,
                "opening_actions": list(game.opening_actions),
                "actions": list(game.actions),
                "black_score": game.black_score,
                "white_score": game.white_score,
                "winner": game.winner.name.lower() if game.winner is not None else None,
                "candidate_points": game.candidate_points,
                "total_simulations": game.total_simulations,
                "search_elapsed_seconds": game.search_elapsed_seconds,
            }
            for game in arena.games
        ],
    }


def _remove_temporary_report(directory: Path) -> None:
    if not directory.exists():
        return
    for child in directory.iterdir():
        if child.is_file():
            child.unlink(missing_ok=True)
    with suppress(OSError):
        directory.rmdir()

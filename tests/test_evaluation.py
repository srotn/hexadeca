"""Arena statistics, immutable reports, and checkpoint gate tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import training.evaluation as evaluation_module
from config import load_config
from config.schema import AppConfig
from game import Player
from network import NetworkSpecification, PolicyValueNetwork
from training import (
    ARENA_SCHEMA_VERSION,
    ArenaBatch,
    ArenaGame,
    ArenaOpening,
    CheckpointEvaluationGate,
    CheckpointManager,
    EvaluationDecision,
    EvaluationGateError,
    EvaluationReportStore,
    calculate_arena_statistics,
    paired_bootstrap_interval,
    performance_elo,
)


def test_paired_bootstrap_and_performance_elo_are_deterministic() -> None:
    """Strong paired results produce finite Elo, confidence, and promotion."""

    config = load_config().evaluation
    batch = _synthetic_batch(candidate_wins=80)

    first = calculate_arena_statistics(batch, config)
    second = calculate_arena_statistics(batch, config)

    assert first == second
    assert first.games == 100
    assert first.opening_pairs == 50
    assert first.candidate_wins == 80
    assert first.best_wins == 20
    assert first.draws == 0
    assert first.score_rate == pytest.approx(0.8)
    assert first.score_confidence_lower > 0.5
    assert first.performance_elo > 0.0
    assert first.promoted is True
    assert paired_bootstrap_interval(
        tuple(
            batch.games[index].candidate_points
            + batch.games[index + 1].candidate_points
            for index in range(0, 100, 2)
        ),
        config,
    ) == (first.score_confidence_lower, first.score_confidence_upper)
    assert performance_elo(0.5, 100, config) == pytest.approx(0.0)
    assert torch.isfinite(torch.tensor(performance_elo(1.0, 100, config)))


def test_equal_arena_score_is_rejected() -> None:
    """A color-balanced 50 percent candidate cannot replace the incumbent."""

    statistics = calculate_arena_statistics(
        _synthetic_batch(candidate_wins=50), load_config().evaluation
    )

    assert statistics.score_rate == pytest.approx(0.5)
    assert statistics.performance_elo == pytest.approx(0.0)
    assert statistics.promoted is False


def test_checkpoint_gate_initializes_first_best_and_publishes_report(
    tmp_path: Path,
) -> None:
    """The first fully loadable checkpoint becomes best without an Arena."""

    config, specification, manager = _checkpoint_stack(tmp_path)
    model = PolicyValueNetwork(specification)
    _save_checkpoint(manager, "candidate-1", model, config, iteration=1)
    next_model = PolicyValueNetwork(specification)
    next_model.load_state_dict(model.state_dict())
    _save_checkpoint(manager, "candidate-2", next_model, config, iteration=2)
    store = EvaluationReportStore(tmp_path / "evaluation")
    gate = CheckpointEvaluationGate(
        config,
        specification,
        manager,
        store,
        device="cpu",
    )

    outcome = gate.evaluate("candidate-1")

    assert outcome.decision is EvaluationDecision.INITIALIZED
    assert outcome.previous_best_checkpoint_id is None
    assert outcome.arena is None
    assert outcome.statistics is None
    assert manager.resolve_identifier("best") == "candidate-1"
    report = json.loads(outcome.report_path.read_text(encoding="utf-8"))
    assert report["schema_version"] == 1
    assert report["decision"] == "initialized"
    assert report["arena"] is None
    assert (
        store.publish(
            decision=EvaluationDecision.INITIALIZED,
            candidate=manager.read_metadata("candidate-1"),
            previous_best=None,
            arena=None,
            statistics=None,
            config=config,
        )
        == outcome.report_path
    )
    with pytest.raises(EvaluationGateError, match="already"):
        gate.evaluate("candidate-1")

    rejected = gate.evaluate("candidate-2")

    assert rejected.decision is EvaluationDecision.REJECTED
    assert rejected.previous_best_checkpoint_id == "candidate-1"
    assert rejected.arena is not None
    assert rejected.statistics is not None
    assert rejected.statistics.score_rate == pytest.approx(0.5)
    assert rejected.statistics.promoted is False
    assert manager.resolve_identifier("best") == "candidate-1"
    rejected_report = json.loads(rejected.report_path.read_text(encoding="utf-8"))
    assert rejected_report["decision"] == "rejected"
    assert len(rejected_report["arena"]["games"]) == 2


def test_checkpoint_gate_publishes_promoted_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A strong validated result publishes its report before changing best."""

    config, specification, manager = _checkpoint_stack(tmp_path)
    model = PolicyValueNetwork(specification)
    _save_checkpoint(manager, "incumbent", model, config, iteration=1)
    _save_checkpoint(manager, "candidate", model, config, iteration=2)
    manager.publish_alias("best", "incumbent")
    arena = _two_game_winning_batch("candidate", "incumbent")

    class StubCoordinator:
        def __init__(self, *args: object, **kwargs: object) -> None:
            assert kwargs["candidate_identifier"] == "candidate"
            assert kwargs["best_identifier"] == "incumbent"

        def run(self) -> ArenaBatch:
            return arena

    monkeypatch.setattr(evaluation_module, "ArenaCoordinator", StubCoordinator)
    gate = CheckpointEvaluationGate(
        config,
        specification,
        manager,
        EvaluationReportStore(tmp_path / "evaluation"),
        device="cpu",
    )

    outcome = gate.evaluate("candidate")

    assert outcome.decision is EvaluationDecision.PROMOTED
    assert outcome.statistics is not None
    assert outcome.statistics.promoted is True
    assert outcome.report_path.is_file()
    assert manager.resolve_identifier("best") == "candidate"


def _checkpoint_stack(
    tmp_path: Path,
) -> tuple[AppConfig, NetworkSpecification, CheckpointManager]:
    config = load_config()
    network = replace(
        config.network,
        residual_blocks=1,
        channels=4,
        value_hidden_features=8,
    )
    evaluation = replace(
        config.evaluation,
        game_count=2,
        opening_plies=1,
        bootstrap_samples=100,
        worker_processes=1,
        inference_max_batch_size=4,
    )
    mcts = replace(
        config.mcts,
        training_simulations=1,
        evaluation_simulations=1,
        max_inference_batch_size=1,
    )
    configured = replace(
        config,
        network=network,
        evaluation=evaluation,
        mcts=mcts,
    )
    specification = NetworkSpecification.from_config(
        configured.rules, configured.network
    )
    return (
        configured,
        specification,
        CheckpointManager(tmp_path / "checkpoints", specification),
    )


def _save_checkpoint(
    manager: CheckpointManager,
    checkpoint_id: str,
    model: PolicyValueNetwork,
    config: AppConfig,
    *,
    iteration: int,
) -> None:
    manager.save(
        checkpoint_id,
        model=model,
        optimizer=None,
        scheduler=None,
        iteration=iteration,
        config=config,
        metrics={},
        aliases=(),
    )


def _synthetic_batch(candidate_wins: int) -> ArenaBatch:
    openings = tuple(ArenaOpening(index, (index,)) for index in range(50))
    games: list[ArenaGame] = []
    for game_index in range(100):
        candidate_player = Player.BLACK if game_index % 2 == 0 else Player.WHITE
        candidate_won = game_index < candidate_wins
        winner = candidate_player if candidate_won else candidate_player.opponent
        games.append(
            ArenaGame(
                schema_version=ARENA_SCHEMA_VERSION,
                game_index=game_index,
                opening_index=game_index // 2,
                worker_index=game_index % 4,
                process_id=1,
                candidate_player=candidate_player,
                seed=game_index,
                opening_actions=openings[game_index // 2].actions,
                actions=openings[game_index // 2].actions,
                black_score=1 if winner is Player.BLACK else 0,
                white_score=1 if winner is Player.WHITE else 0,
                winner=winner,
                candidate_points=1.0 if candidate_won else 0.0,
                total_simulations=0,
                search_elapsed_seconds=0.0,
            )
        )
    return ArenaBatch(
        schema_version=ARENA_SCHEMA_VERSION,
        candidate_identifier="candidate",
        best_identifier="best",
        random_seed=20_260_803,
        openings=openings,
        games=tuple(games),
        worker_processes=4,
        candidate_inference_batches=1,
        candidate_inference_positions=1,
        best_inference_batches=1,
        best_inference_positions=1,
        maximum_inference_batch_size=1,
        elapsed_seconds=1.0,
    )


def _two_game_winning_batch(candidate_id: str, best_id: str) -> ArenaBatch:
    opening = ArenaOpening(0, (0,))
    games = tuple(
        ArenaGame(
            schema_version=ARENA_SCHEMA_VERSION,
            game_index=index,
            opening_index=0,
            worker_index=0,
            process_id=1,
            candidate_player=candidate_player,
            seed=index,
            opening_actions=opening.actions,
            actions=opening.actions,
            black_score=1 if candidate_player is Player.BLACK else 0,
            white_score=1 if candidate_player is Player.WHITE else 0,
            winner=candidate_player,
            candidate_points=1.0,
            total_simulations=0,
            search_elapsed_seconds=0.0,
        )
        for index, candidate_player in enumerate((Player.BLACK, Player.WHITE))
    )
    return ArenaBatch(
        schema_version=ARENA_SCHEMA_VERSION,
        candidate_identifier=candidate_id,
        best_identifier=best_id,
        random_seed=20_260_803,
        openings=(opening,),
        games=games,
        worker_processes=1,
        candidate_inference_batches=1,
        candidate_inference_positions=1,
        best_inference_batches=1,
        best_inference_positions=1,
        maximum_inference_batch_size=1,
        elapsed_seconds=1.0,
    )

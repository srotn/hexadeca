"""Paired-opening Arena and centralized dual-model inference tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from config import load_config
from game import Board, GameState, Player
from mcts import Evaluation
from training import ArenaCoordinator, generate_openings


class UniformEvaluator:
    """Deterministic evaluator with observable parent-process batch counters."""

    def __init__(self) -> None:
        self.batch_count = 0
        self.position_count = 0

    def evaluate(self, states: Sequence[GameState]) -> tuple[Evaluation, ...]:
        self.batch_count += 1
        self.position_count += len(states)
        return tuple(_uniform_evaluation(state) for state in states)


def test_configured_openings_are_unique_legal_and_reproducible() -> None:
    """Fifty opening pairs have distinct first moves and exact legal prefixes."""

    config = load_config()
    first = generate_openings(
        config.rules,
        pair_count=config.evaluation.opening_pair_count,
        opening_plies=config.evaluation.opening_plies,
        random_seed=config.evaluation.random_seed,
    )
    second = generate_openings(
        config.rules,
        pair_count=config.evaluation.opening_pair_count,
        opening_plies=config.evaluation.opening_plies,
        random_seed=config.evaluation.random_seed,
    )

    assert first == second
    assert len(first) == 50
    assert len({opening.actions for opening in first}) == 50
    assert len({opening.actions[0] for opening in first}) == 50
    for index, opening in enumerate(first):
        assert opening.index == index
        assert len(opening.actions) == 2
        board = Board(config.rules)
        for action in opening.actions:
            board.apply(action)
        assert board.is_terminal is False


def test_arena_coordinator_completes_one_color_swapped_pair() -> None:
    """Spawn workers search both colors while evaluators stay in the parent."""

    config = load_config()
    rules = replace(config.rules, board_size=4)
    mcts = replace(
        config.mcts,
        training_simulations=2,
        evaluation_simulations=2,
        max_inference_batch_size=2,
    )
    evaluation = replace(
        config.evaluation,
        game_count=2,
        opening_plies=1,
        bootstrap_samples=100,
        worker_processes=1,
        inference_max_batch_size=4,
        inference_batch_wait_seconds=0.001,
    )
    candidate = UniformEvaluator()
    best = UniformEvaluator()
    coordinator = ArenaCoordinator(
        rules,
        mcts,
        evaluation,
        candidate,
        best,
        candidate_identifier="candidate-1",
        best_identifier="best-1",
    )

    batch = coordinator.run()

    assert len(batch.openings) == 1
    assert len(batch.games) == 2
    assert batch.games[0].candidate_player is Player.BLACK
    assert batch.games[1].candidate_player is Player.WHITE
    assert batch.games[0].opening_actions == batch.games[1].opening_actions
    assert batch.candidate_wins + batch.best_wins + batch.draws == 2
    assert batch.candidate_points == 1.0
    assert batch.total_simulations > 0
    assert batch.worker_processes == 1
    assert batch.maximum_inference_batch_size <= 4
    assert candidate.batch_count == batch.candidate_inference_batches > 0
    assert candidate.position_count == batch.candidate_inference_positions > 0
    assert best.batch_count == batch.best_inference_batches > 0
    assert best.position_count == batch.best_inference_positions > 0


def _uniform_evaluation(state: GameState) -> Evaluation:
    legal_count = sum(state.legal_mask)
    return Evaluation(
        policy=tuple(1.0 / legal_count if legal else 0.0 for legal in state.legal_mask),
        value=0.0,
    )

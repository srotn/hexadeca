"""Paired-opening Arena and centralized dual-model inference tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from config import load_config
from game import Board, GameState, Player
from mcts import Evaluation
from training import ArenaCoordinator, generate_openings, play_arena_game


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
        exact_endgame_enabled=False,
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


def test_arena_supports_distinct_search_budgets_per_model() -> None:
    """A search-only comparison accounts for each side's exact simulations."""

    config = load_config()
    rules = replace(config.rules, board_size=4)
    shared_mcts = replace(
        config.mcts,
        training_simulations=1,
        evaluation_simulations=1,
        max_inference_batch_size=2,
        exact_endgame_enabled=False,
    )
    candidate_mcts = replace(shared_mcts, evaluation_simulations=2)
    evaluation = replace(
        config.evaluation,
        game_count=2,
        opening_plies=1,
        bootstrap_samples=100,
        worker_processes=1,
        inference_max_batch_size=4,
        inference_batch_wait_seconds=0.001,
    )
    coordinator = ArenaCoordinator(
        rules,
        shared_mcts,
        evaluation,
        UniformEvaluator(),
        UniformEvaluator(),
        candidate_identifier="same-model-s2",
        best_identifier="same-model-s1",
        candidate_mcts=candidate_mcts,
        best_mcts=shared_mcts,
    )

    batch = coordinator.run()

    expected_simulations = 0
    for game in batch.games:
        for ply in range(len(game.opening_actions), game.plies):
            player = Player.BLACK if ply % 2 == 0 else Player.WHITE
            expected_simulations += 2 if player is game.candidate_player else 1
    assert batch.total_simulations == expected_simulations


def test_arena_resumes_from_a_validated_partial_game() -> None:
    """Only missing games are searched and newly finished games are reported."""

    config = load_config()
    rules = replace(config.rules, board_size=4)
    mcts = replace(
        config.mcts,
        training_simulations=2,
        evaluation_simulations=2,
        max_inference_batch_size=2,
        exact_endgame_enabled=False,
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
    opening = generate_openings(
        rules,
        pair_count=1,
        opening_plies=1,
        random_seed=evaluation.random_seed,
    )[0]
    prior_game = play_arena_game(
        rules,
        mcts,
        UniformEvaluator(),
        UniformEvaluator(),
        game_index=0,
        opening=opening,
        candidate_player=Player.BLACK,
        seed=_derive_seed(evaluation.random_seed, 0),
        draw_score=evaluation.draw_score,
    )
    completed = []
    batch = ArenaCoordinator(
        rules,
        mcts,
        evaluation,
        UniformEvaluator(),
        UniformEvaluator(),
        candidate_identifier="candidate-1",
        best_identifier="best-1",
    ).run(prior_games=(prior_game,), game_callback=completed.append)

    assert batch.games[0] == prior_game
    assert len(batch.games) == 2
    assert [game.game_index for game in completed] == [1]

    fully_resumed = ArenaCoordinator(
        rules,
        mcts,
        evaluation,
        UniformEvaluator(),
        UniformEvaluator(),
        candidate_identifier="candidate-1",
        best_identifier="best-1",
    ).run(prior_games=batch.games)
    assert fully_resumed.games == batch.games
    assert fully_resumed.worker_processes == 0


def _derive_seed(master_seed: int, game_index: int) -> int:
    value = (master_seed + (game_index + 1) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & ((1 << 64) - 1)
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & ((1 << 64) - 1)
    return (value ^ (value >> 31)) & ((1 << 64) - 1)


def _uniform_evaluation(state: GameState) -> Evaluation:
    legal_count = sum(state.legal_mask)
    return Evaluation(
        policy=tuple(1.0 / legal_count if legal else 0.0 for legal in state.legal_mask),
        value=0.0,
    )

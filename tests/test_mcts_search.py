"""Deterministic integration tests for exact and batched neural MCTS."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import replace

import pytest

from config import load_config
from game import GameEnvironment, GameState, Player
from mcts import (
    Evaluation,
    InvalidEvaluationError,
    MctsSearch,
    SearchMode,
    SearchStateError,
)


class RecordingEvaluator:
    """Return legal uniform priors while recording every requested batch."""

    def __init__(self, value: float = 0.0) -> None:
        self.value = value
        self.batch_sizes: list[int] = []

    def evaluate(self, states: Sequence[GameState]) -> tuple[Evaluation, ...]:
        self.batch_sizes.append(len(states))
        return tuple(_uniform_evaluation(state, self.value) for state in states)


class InvalidSecondBatchEvaluator(RecordingEvaluator):
    """Return an illegal policy after root expansion."""

    def evaluate(self, states: Sequence[GameState]) -> tuple[Evaluation, ...]:
        self.batch_sizes.append(len(states))
        if len(self.batch_sizes) == 1:
            return tuple(_uniform_evaluation(state, 0.0) for state in states)
        return tuple(
            Evaluation(policy=(1.0,) + (0.0,) * (state.action_size - 1), value=0.0)
            for state in states
        )


def test_evaluation_search_completes_exact_simulations_in_real_batches() -> None:
    """Root edge visits equal the configured count and inference batches fill."""

    config = load_config()
    mcts_config = replace(
        config.mcts,
        evaluation_simulations=8,
        inference_batch_size=4,
    )
    evaluator = RecordingEvaluator()
    state = GameEnvironment(config.rules).state

    result = MctsSearch(config.rules, mcts_config, evaluator).run(
        state, SearchMode.EVALUATION
    )

    assert result.simulations == 8
    assert sum(result.visit_counts) == 8
    assert result.root.visit_count == 8
    assert evaluator.batch_sizes == [1, 4, 4]
    assert result.inference_batches == 3
    assert result.inference_positions == 9
    assert result.maximum_batch_size == 4
    assert result.action_probabilities[0] == 1.0
    assert sum(result.policy_target) == pytest.approx(1.0)
    assert result.policy_target != result.action_probabilities
    assert all(
        edge.virtual_visit_count == 0 and edge.virtual_value_sum == 0.0
        for edge in result.root.children.values()
    )


def test_training_search_applies_reproducible_root_noise() -> None:
    """Training mode consumes configured epsilon and caller-owned RNG only."""

    config = load_config()
    mcts_config = replace(
        config.mcts,
        training_simulations=4,
        inference_batch_size=4,
    )
    state = GameEnvironment(config.rules).state

    first = MctsSearch(config.rules, mcts_config, RecordingEvaluator()).run(
        state,
        SearchMode.TRAINING,
        random_source=random.Random(23),
    )
    second = MctsSearch(config.rules, mcts_config, RecordingEvaluator()).run(
        state,
        SearchMode.TRAINING,
        random_source=random.Random(23),
    )

    uniform_prior = 1.0 / state.action_size
    assert first.root_priors == second.root_priors
    assert any(prior != uniform_prior for prior in first.root_priors)
    assert sum(first.root_priors) == pytest.approx(1.0)
    assert first.policy_target == first.action_probabilities
    assert sum(first.action_probabilities) == pytest.approx(1.0)


def test_terminal_leaf_uses_official_score_without_neural_evaluation() -> None:
    """A one-cell game backs up Black's official win from White's leaf view."""

    config = load_config()
    rules = replace(config.rules, board_size=1)
    mcts_config = replace(
        config.mcts,
        evaluation_simulations=5,
        inference_batch_size=4,
    )
    evaluator = RecordingEvaluator(value=-1.0)

    result = MctsSearch(rules, mcts_config, evaluator).run(
        GameEnvironment(rules).state, SearchMode.EVALUATION
    )

    assert evaluator.batch_sizes == [1]
    assert result.inference_positions == 1
    assert result.visit_counts == (5,)
    assert result.root_value == pytest.approx(1.0)
    assert result.root.children[0].mean_value == pytest.approx(1.0)
    assert result.root.children[0].child is not None
    assert result.root.children[0].child.to_play is Player.WHITE


def test_neural_value_is_negated_from_child_to_root_perspective() -> None:
    """A winning value for the child side is losing for the root side."""

    config = load_config()
    mcts_config = replace(
        config.mcts,
        evaluation_simulations=1,
        inference_batch_size=1,
    )

    result = MctsSearch(config.rules, mcts_config, RecordingEvaluator(value=1.0)).run(
        GameEnvironment(config.rules).state, SearchMode.EVALUATION
    )

    assert result.root_value == pytest.approx(-1.0)
    visited_edge = next(
        edge for edge in result.root.children.values() if edge.visit_count == 1
    )
    assert visited_edge.mean_value == pytest.approx(-1.0)


def test_invalid_root_and_training_without_rng_fail_before_search() -> None:
    """Search rejects corrupted state and implicit global randomness."""

    config = load_config()
    state = GameEnvironment(config.rules).state
    corrupted = replace(state, zobrist_hash=0)
    evaluator = RecordingEvaluator()
    search = MctsSearch(config.rules, config.mcts, evaluator)

    with pytest.raises(SearchStateError, match="replayed history"):
        search.run(corrupted, SearchMode.EVALUATION)
    with pytest.raises(SearchStateError, match="random source"):
        search.run(state, SearchMode.TRAINING)
    assert evaluator.batch_sizes == []


def test_invalid_leaf_evaluation_is_rejected() -> None:
    """Malformed evaluator output cannot be silently normalized by search."""

    config = load_config()
    mcts_config = replace(
        config.mcts,
        evaluation_simulations=2,
        inference_batch_size=2,
    )

    with pytest.raises(InvalidEvaluationError, match="illegal"):
        MctsSearch(config.rules, mcts_config, InvalidSecondBatchEvaluator()).run(
            GameEnvironment(config.rules).state, SearchMode.EVALUATION
        )


def _uniform_evaluation(state: GameState, value: float) -> Evaluation:
    legal_count = sum(state.legal_mask)
    probability = 1.0 / legal_count
    return Evaluation(
        policy=tuple(probability if legal else 0.0 for legal in state.legal_mask),
        value=value,
    )

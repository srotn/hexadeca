"""Exact endgame solving and MCTS integration regressions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import pytest

from config import load_config
from game import GameEnvironment, GameState, Player
from mcts import Evaluation, MctsSearch, SearchMode
from native import is_available

_REGRESSION_HISTORY = (
    120,
    77,
    26,
    211,
    125,
    158,
    48,
    29,
    23,
    5,
    170,
    229,
    132,
    50,
    114,
    251,
    197,
    84,
    221,
    70,
    52,
    111,
    91,
    3,
    243,
    150,
    160,
    122,
    102,
    192,
    190,
    203,
    255,
    233,
    146,
    31,
    241,
    156,
    179,
    88,
    57,
    80,
    200,
    112,
    1,
    247,
    152,
)


class FailingEvaluator:
    """Prove that exact roots do not depend on neural inference."""

    def evaluate(self, states: Sequence[GameState]) -> tuple[Evaluation, ...]:
        raise AssertionError("An exact endgame must not call the evaluator")


class UniformEvaluator:
    """Provide neutral priors until search crosses the exact horizon."""

    def evaluate(self, states: Sequence[GameState]) -> tuple[Evaluation, ...]:
        return tuple(
            Evaluation(
                policy=tuple(
                    1.0 / sum(state.legal_mask) if legal else 0.0
                    for legal in state.legal_mask
                ),
                value=0.0,
            )
            for state in states
        )


def _state(actions: tuple[int, ...]) -> GameState:
    config = load_config()
    environment = GameEnvironment(config.rules)
    for action in actions:
        environment.step(action)
    return environment.state


@pytest.mark.parametrize("engine", ["reference", "native"])
def test_regression_endgame_is_proven_and_child_consistent(engine: str) -> None:
    """The former 95%-wrong six-move position is solved exactly by both engines."""

    config = load_config()
    mcts = replace(
        config.mcts,
        engine=engine,
        evaluation_simulations=17,
        exact_endgame_max_legal_moves=6,
    )
    state = _state(_REGRESSION_HISTORY)
    result = MctsSearch(config.rules, mcts, FailingEvaluator()).run(
        state, SearchMode.EVALUATION
    )

    assert state.to_play is Player.WHITE
    assert result.solved_outcome == -1
    assert result.root_value == -1.0
    assert result.inference_batches == 0
    assert result.inference_positions == 0
    assert result.exact_states_evaluated > 0
    assert sum(result.visit_counts) == 17
    action_outcomes = dict(result.exact_action_outcomes)
    assert set(action_outcomes) == {59, 63, 79, 82, 223, 253}
    assert set(action_outcomes.values()) == {-1}

    for action, parent_outcome in result.exact_action_outcomes:
        child = _state((*_REGRESSION_HISTORY, action))
        child_result = MctsSearch(config.rules, mcts, FailingEvaluator()).run(
            child, SearchMode.EVALUATION
        )
        assert parent_outcome == -child_result.solved_outcome


def test_exact_root_selects_only_proven_optimal_actions() -> None:
    """Exact visit targets exclude proven inferior moves at every temperature."""

    config = load_config()
    history = (*_REGRESSION_HISTORY, 253, 59, 82)
    state = _state(history)
    mcts = replace(
        config.mcts,
        engine="reference",
        evaluation_simulations=19,
        exact_endgame_max_legal_moves=3,
    )

    result = MctsSearch(config.rules, mcts, FailingEvaluator()).run(
        state, SearchMode.EVALUATION
    )

    assert state.to_play is Player.BLACK
    assert result.solved_outcome == 1
    assert dict(result.exact_action_outcomes) == {63: 0, 79: -1, 223: 1}
    assert result.action_probabilities[223] == 1.0
    assert sum(result.visit_counts) == 19
    assert result.visit_counts[223] == 19
    assert result.root.children[63].mean_value == 0.0
    assert result.root.children[79].mean_value == -1.0
    assert result.root.children[223].mean_value == 1.0


@pytest.mark.parametrize("engine", ["reference", "native"])
def test_search_propagates_solved_children_across_exact_horizon(engine: str) -> None:
    """A root just outside the horizon becomes solved from exact child values."""

    if engine == "native" and not is_available():
        pytest.skip("The native extension was not built")
    config = load_config()
    state = _state((*_REGRESSION_HISTORY, 253, 59))
    assert sum(state.legal_mask) == 4
    mcts = replace(
        config.mcts,
        engine=engine,
        evaluation_simulations=64,
        max_inference_batch_size=4,
        exact_endgame_max_legal_moves=3,
    )

    result = MctsSearch(config.rules, mcts, UniformEvaluator()).run(
        state, SearchMode.EVALUATION
    )

    assert result.solved_outcome == -1
    assert result.root_value == -1.0
    assert result.exact_states_evaluated > 0
    assert set(dict(result.exact_action_outcomes).values()) == {-1}

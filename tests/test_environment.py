"""Lifecycle and self-play contract tests for the Stage 5 environment."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from config import load_config
from game import (
    GameEnvironment,
    IllegalMoveError,
    InvalidActionError,
    NonTerminalBoardError,
    Player,
    TerminalStateError,
    encode_action,
)


def test_initial_environment_exposes_an_immutable_empty_state() -> None:
    """A new environment exposes Black to play and every action as legal."""

    rules = load_config().rules
    environment = GameEnvironment(rules)
    state = environment.state

    assert state.ruleset_id == rules.ruleset_id
    assert state.board_size == 16
    assert state.action_size == 256
    assert state.cells == (0,) * 256
    assert state.to_play is Player.BLACK
    assert state.ply == 0
    assert state.history == ()
    assert state.legal_mask == (True,) * 256
    assert state.terminal is False
    assert environment.legal_moves() == tuple(range(256))
    assert environment.terminal() is False


def test_step_returns_move_new_state_and_no_ongoing_result() -> None:
    """A normal step advances the turn while preserving the prior snapshot."""

    environment = GameEnvironment(load_config().rules)
    initial_state = environment.state
    action = encode_action(8, 8, initial_state.board_size)

    transition = environment.step(action)

    assert transition.move.action == action
    assert transition.move.player is Player.BLACK
    assert transition.result is None
    assert transition.state == environment.state
    assert transition.state.to_play is Player.WHITE
    assert transition.state.ply == 1
    assert transition.state.history == (transition.move,)
    assert transition.state.cells[action] == Player.BLACK.value
    assert initial_state.cells == (0,) * 256
    assert initial_state.history == ()


def test_illegal_and_invalid_steps_are_atomic() -> None:
    """Rejected actions leave all observable environment state unchanged."""

    environment = GameEnvironment(load_config().rules)
    environment.step(encode_action(8, 8, 16))
    state_before = environment.state

    with pytest.raises(IllegalMoveError):
        environment.step(encode_action(7, 7, 16))
    assert environment.state == state_before

    with pytest.raises(InvalidActionError):
        environment.step(256)
    assert environment.state == state_before


def test_terminal_step_caches_score_winner_and_rejects_further_actions() -> None:
    """The terminal transition owns one cached official result."""

    rules = replace(load_config().rules, board_size=1)
    environment = GameEnvironment(rules)

    transition = environment.step(0)

    assert transition.state.terminal is True
    assert transition.state.legal_mask == (False,)
    assert transition.result is not None
    assert transition.result.black_score == 1
    assert transition.result.white_score == 0
    assert transition.result.zobrist_hash == transition.state.zobrist_hash
    assert environment.terminal() is True
    assert environment.legal_moves() == ()
    assert environment.result() is transition.result
    assert environment.winner() is Player.BLACK

    state_before = environment.state
    with pytest.raises(TerminalStateError):
        environment.step(0)
    assert environment.state == state_before
    assert environment.result() is transition.result


def test_non_terminal_game_has_no_result_or_winner() -> None:
    """A missing winner is not confused with a terminal draw."""

    environment = GameEnvironment(load_config().rules)

    with pytest.raises(NonTerminalBoardError):
        environment.result()
    with pytest.raises(NonTerminalBoardError):
        environment.winner()


def test_reset_replaces_game_and_invalidates_terminal_result() -> None:
    """Reset returns the deterministic initial state without mutating old snapshots."""

    rules = replace(load_config().rules, board_size=1)
    environment = GameEnvironment(rules)
    completed_transition = environment.step(0)
    completed_state = completed_transition.state

    reset_state = environment.reset()

    assert reset_state == environment.state
    assert reset_state.cells == (0,)
    assert reset_state.to_play is Player.BLACK
    assert reset_state.ply == 0
    assert reset_state.history == ()
    assert reset_state.legal_mask == (True,)
    assert reset_state.terminal is False
    assert completed_state.cells == (Player.BLACK.value,)
    assert completed_state.terminal is True
    with pytest.raises(NonTerminalBoardError):
        environment.result()


def test_seeded_self_play_reaches_one_consistent_terminal_result() -> None:
    """A full random self-play game can be consumed only through public APIs."""

    environment = GameEnvironment(load_config().rules)
    random_source = random.Random(20260805)
    initial_state = environment.state
    transitions = []

    while not environment.terminal():
        state_before = environment.state
        legal_moves = environment.legal_moves()
        assert legal_moves == tuple(
            action for action, legal in enumerate(state_before.legal_mask) if legal
        )

        transition = environment.step(random_source.choice(legal_moves))
        transitions.append(transition)
        assert transition.move.player is state_before.to_play
        assert transition.state.history == (*state_before.history, transition.move)
        assert (transition.result is not None) is transition.state.terminal

    result = environment.result()
    final_state = environment.state
    assert transitions[-1].result is result
    assert final_state.ply == len(transitions)
    assert final_state.history == tuple(item.move for item in transitions)
    assert final_state.zobrist_hash == result.zobrist_hash
    assert environment.winner() is result.winner
    assert initial_state.cells == (0,) * initial_state.action_size
    assert initial_state.history == ()

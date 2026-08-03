"""Contract and invariant tests for the Stage 3 board engine."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

import pytest

from config import load_config
from game import (
    Board,
    IllegalMoveError,
    InvalidActionError,
    Player,
    UndoError,
    decode_action,
    encode_action,
)


def test_empty_board_has_all_actions_available() -> None:
    """An empty configured board starts with Black and all cells legal."""

    board = Board(load_config().rules)

    assert board.size == 16
    assert board.action_size == 256
    assert board.to_play is Player.BLACK
    assert board.ply == 0
    assert board.history == ()
    assert board.cells == (0,) * 256
    assert board.legal_count == 256
    assert board.legal_actions() == tuple(range(256))
    assert board.legal_mask() == (True,) * 256
    assert board.is_terminal is False


@pytest.mark.parametrize(
    ("row", "column", "action"),
    [(0, 0, 0), (0, 15, 15), (8, 7, 135), (15, 15, 255)],
)
def test_action_coordinate_mapping_is_row_major(
    row: int, column: int, action: int
) -> None:
    """The public coordinate mapping is stable and exactly reversible."""

    assert encode_action(row, column, 16) == action
    assert decode_action(action, 16) == (row, column)


@pytest.mark.parametrize(
    ("operation", "arguments"),
    [
        (encode_action, (-1, 0, 16)),
        (encode_action, (0, 16, 16)),
        (encode_action, (True, 0, 16)),
        (decode_action, (-1, 16)),
        (decode_action, (256, 16)),
        (decode_action, (True, 16)),
        (decode_action, (0, 0)),
    ],
)
def test_invalid_action_coordinates_are_rejected(
    operation: Any, arguments: tuple[Any, ...]
) -> None:
    """Invalid coordinates, action indices, and boolean integers fail clearly."""

    with pytest.raises(InvalidActionError):
        operation(*arguments)


def test_center_and_corner_moves_block_only_in_bounds_neighbours() -> None:
    """A placement blocks its clipped Chebyshev-distance-one neighbourhood."""

    rules = load_config().rules
    center_board = Board(rules)
    corner_board = Board(rules)

    center_move = center_board.place(8, 8)
    corner_board.place(0, 0)

    assert center_move.player is Player.BLACK
    assert center_move.action == encode_action(8, 8, 16)
    assert center_board.piece_at(8, 8) is Player.BLACK
    assert center_board.to_play is Player.WHITE
    assert center_board.legal_count == 256 - 9
    assert all(
        not center_board.is_legal_at(row, column)
        for row in range(7, 10)
        for column in range(7, 10)
    )
    assert center_board.is_legal_at(6, 8)
    assert corner_board.legal_count == 256 - 4


def test_overlapping_blocked_neighbourhoods_use_reference_counts() -> None:
    """Undo preserves cells that remain blocked by another nearby stone."""

    rules = replace(load_config().rules, board_size=5)
    board = Board(rules)

    board.place(1, 1)
    assert board.legal_count == 16

    board.place(1, 3)
    assert board.legal_count == 10
    assert not board.is_legal_at(0, 2)

    undone = board.undo()

    assert (undone.row, undone.column) == (1, 3)
    assert board.legal_count == 16
    assert not board.is_legal_at(0, 2)
    assert board.is_legal_at(1, 3)


def test_rejected_move_does_not_mutate_observable_state() -> None:
    """Both illegal and invalid placements are atomic failures."""

    board = Board(load_config().rules)
    board.place(8, 8)
    before = _observable_state(board)

    with pytest.raises(IllegalMoveError):
        board.place(7, 7)
    assert _observable_state(board) == before

    with pytest.raises(InvalidActionError):
        board.apply(256)
    assert _observable_state(board) == before


def test_history_records_players_and_undo_restores_exact_state() -> None:
    """History alternates players and undo is an exact inverse of apply."""

    board = Board(load_config().rules)
    initial_state = _observable_state(board)

    black_move = board.place(0, 0)
    after_black = _observable_state(board)
    white_move = board.place(0, 2)

    assert board.history == (black_move, white_move)
    assert black_move.player is Player.BLACK
    assert white_move.player is Player.WHITE
    assert board.to_play is Player.BLACK

    assert board.undo() == white_move
    assert _observable_state(board) == after_black
    assert board.undo() == black_move
    assert _observable_state(board) == initial_state

    with pytest.raises(UndoError):
        board.undo()
    assert _observable_state(board) == initial_state


def test_zobrist_hash_is_deterministic_and_seeded() -> None:
    """Equal state construction hashes equally and changing the seed changes it."""

    rules = load_config().rules
    first = Board(rules)
    second = Board(rules)
    different_seed = Board(replace(rules, zobrist_seed=rules.zobrist_seed + 1))

    actions = [encode_action(0, 0, 16), encode_action(0, 2, 16), 64]
    for action in actions:
        first.apply(action)
        second.apply(action)
        different_seed.apply(action)

    assert first.zobrist_hash == second.zobrist_hash
    assert first.zobrist_hash != different_seed.zobrist_hash


def test_dense_even_coordinate_position_is_terminal_at_64_moves() -> None:
    """The maximum 8 by 8 spacing pattern covers all cells on a 16 board."""

    board = Board(load_config().rules)

    for row in range(0, 16, 2):
        for column in range(0, 16, 2):
            board.place(row, column)

    assert board.ply == 64
    assert board.legal_count == 0
    assert board.legal_actions() == ()
    assert board.legal_mask() == (False,) * 256
    assert board.is_terminal is True

    board.undo()
    assert board.is_terminal is False
    assert board.is_legal_at(15, 15)


def test_incremental_legality_matches_independent_reference_through_undo() -> None:
    """A seeded full game agrees with a naive scanner before and after every move."""

    board = Board(load_config().rules)
    random_source = random.Random(20260803)
    previous_states: list[tuple[object, ...]] = []

    while not board.is_terminal:
        expected_actions = _naive_legal_actions(
            board.cells, board.size, board.rules.neighborhood_radius
        )
        assert board.legal_actions() == expected_actions
        assert board.legal_count == len(expected_actions)
        previous_states.append(_observable_state(board))
        board.apply(random_source.choice(expected_actions))

    assert board.legal_actions() == _naive_legal_actions(
        board.cells, board.size, board.rules.neighborhood_radius
    )

    while previous_states:
        expected_state = previous_states.pop()
        board.undo()
        assert _observable_state(board) == expected_state


def _observable_state(board: Board) -> tuple[object, ...]:
    return (
        board.cells,
        board.to_play,
        board.history,
        board.legal_actions(),
        board.legal_mask(),
        board.legal_count,
        board.is_terminal,
        board.zobrist_hash,
    )


def _naive_legal_actions(
    cells: tuple[int, ...], board_size: int, radius: int
) -> tuple[int, ...]:
    legal_actions: list[int] = []
    for action, cell in enumerate(cells):
        if cell != 0:
            continue
        row, column = divmod(action, board_size)
        blocked = False
        for stone_action, stone in enumerate(cells):
            if stone == 0:
                continue
            stone_row, stone_column = divmod(stone_action, board_size)
            if max(abs(row - stone_row), abs(column - stone_column)) <= radius:
                blocked = True
                break
        if not blocked:
            legal_actions.append(action)
    return tuple(legal_actions)

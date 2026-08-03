"""Correctness-first Hexadeca board state and incremental legal-move engine."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

from config.schema import RulesConfig
from game.errors import IllegalMoveError, InvalidActionError, UndoError

_EMPTY = 0
_UINT64_MASK = (1 << 64) - 1
_SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15
_SPLITMIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
_SPLITMIX_MULTIPLIER_2 = 0x94D049BB133111EB


class Player(Enum):
    """The player colours and their stable board encodings."""

    BLACK = 1
    WHITE = 2

    @property
    def opponent(self) -> Player:
        """Return the player who moves after this player."""

        return Player.WHITE if self is Player.BLACK else Player.BLACK


@dataclass(frozen=True, slots=True)
class Move:
    """One committed placement in immutable history form."""

    action: int
    row: int
    column: int
    player: Player


class Board:
    """Mutable Hexadeca board with reversible, incrementally cached state.

    The flat cell encoding is 0 for empty, 1 for Black, and 2 for White.
    Legal placement status is maintained with a reference count per cell, so
    overlapping blocked neighbourhoods are restored correctly during undo.
    """

    __slots__ = (
        "_blocked_counts",
        "_cells",
        "_history",
        "_legal_count",
        "_piece_keys",
        "_rules",
        "_side_keys",
        "_to_play",
        "_zobrist_hash",
    )

    def __init__(self, rules: RulesConfig) -> None:
        """Create an empty board using a validated immutable rules config."""

        self._rules = rules
        self._cells = [_EMPTY] * rules.action_size
        self._blocked_counts = [0] * rules.action_size
        self._history: list[Move] = []
        self._legal_count = rules.action_size
        self._to_play = Player.BLACK
        self._piece_keys, self._side_keys = _zobrist_keys(
            rules.board_size, rules.zobrist_seed
        )
        self._zobrist_hash = self._side_keys[self._player_index(self._to_play)]

    @property
    def rules(self) -> RulesConfig:
        """Return the immutable rules configuration used by this board."""

        return self._rules

    @property
    def size(self) -> int:
        """Return the board width and height."""

        return self._rules.board_size

    @property
    def action_size(self) -> int:
        """Return the number of placement actions on the board."""

        return self._rules.action_size

    @property
    def to_play(self) -> Player:
        """Return the player required to place the next stone."""

        return self._to_play

    @property
    def ply(self) -> int:
        """Return the number of committed placements."""

        return len(self._history)

    @property
    def history(self) -> tuple[Move, ...]:
        """Return an immutable snapshot of the committed move history."""

        return tuple(self._history)

    @property
    def cells(self) -> tuple[int, ...]:
        """Return an immutable row-major snapshot using encodings 0, 1, and 2."""

        return tuple(self._cells)

    @property
    def legal_count(self) -> int:
        """Return the number of currently legal placement actions."""

        return self._legal_count

    @property
    def is_terminal(self) -> bool:
        """Return whether no legal placement remains."""

        return self._legal_count == 0

    @property
    def zobrist_hash(self) -> int:
        """Return the deterministic 64-bit hash including side to play."""

        return self._zobrist_hash

    def piece_at(self, row: int, column: int) -> Player | None:
        """Return the stone at a coordinate, or ``None`` when it is empty."""

        action = encode_action(row, column, self.size)
        value = self._cells[action]
        return None if value == _EMPTY else Player(value)

    def is_legal(self, action: int) -> bool:
        """Return whether an action is in range and currently legal."""

        if type(action) is not int or not 0 <= action < self.action_size:
            return False
        return self._cells[action] == _EMPTY and self._blocked_counts[action] == 0

    def is_legal_at(self, row: int, column: int) -> bool:
        """Return whether a coordinate is in range and currently legal."""

        try:
            action = encode_action(row, column, self.size)
        except InvalidActionError:
            return False
        return self.is_legal(action)

    def legal_actions(self) -> tuple[int, ...]:
        """Return all legal row-major action indices in stable order."""

        return tuple(
            action
            for action, (cell, blocked) in enumerate(
                zip(self._cells, self._blocked_counts, strict=True)
            )
            if cell == _EMPTY and blocked == 0
        )

    def legal_mask(self) -> tuple[bool, ...]:
        """Return a stable action-sized mask of current legal placements."""

        return tuple(
            cell == _EMPTY and blocked == 0
            for cell, blocked in zip(self._cells, self._blocked_counts, strict=True)
        )

    def place(self, row: int, column: int) -> Move:
        """Place at a coordinate for the current player and return its record."""

        return self.apply(encode_action(row, column, self.size))

    def apply(self, action: int) -> Move:
        """Commit one legal action and update all reversible derived state."""

        self._validate_action(action)
        if not self.is_legal(action):
            row, column = decode_action(action, self.size)
            raise IllegalMoveError(f"Illegal placement at ({row}, {column})")

        player = self._to_play
        row, column = decode_action(action, self.size)
        move = Move(action=action, row=row, column=column, player=player)

        for blocked_action in self._neighbourhood_actions(action):
            if (
                self._cells[blocked_action] == _EMPTY
                and self._blocked_counts[blocked_action] == 0
            ):
                self._legal_count -= 1
            self._blocked_counts[blocked_action] += 1

        self._cells[action] = player.value
        self._toggle_hash_for_move(action, player)
        self._to_play = player.opponent
        self._history.append(move)
        return move

    def undo(self) -> Move:
        """Undo the most recent placement and restore all observable state."""

        if not self._history:
            raise UndoError("Cannot undo an empty board history")

        move = self._history[-1]
        blocked_actions = tuple(self._neighbourhood_actions(move.action))
        if any(self._blocked_counts[action] <= 0 for action in blocked_actions):
            raise RuntimeError("Blocked-count state is internally inconsistent")

        self._cells[move.action] = _EMPTY
        for blocked_action in blocked_actions:
            self._blocked_counts[blocked_action] -= 1
            if (
                self._cells[blocked_action] == _EMPTY
                and self._blocked_counts[blocked_action] == 0
            ):
                self._legal_count += 1

        self._to_play = move.player
        self._toggle_hash_for_move(move.action, move.player)
        self._history.pop()
        return move

    def _validate_action(self, action: int) -> None:
        if type(action) is not int or not 0 <= action < self.action_size:
            raise InvalidActionError(
                f"Action must be an integer in [0, {self.action_size - 1}]"
            )

    def _neighbourhood_actions(self, action: int) -> Iterator[int]:
        row, column = divmod(action, self.size)
        radius = self._rules.neighborhood_radius
        row_start = max(0, row - radius)
        row_stop = min(self.size, row + radius + 1)
        column_start = max(0, column - radius)
        column_stop = min(self.size, column + radius + 1)

        for neighbour_row in range(row_start, row_stop):
            base_action = neighbour_row * self.size
            for neighbour_column in range(column_start, column_stop):
                yield base_action + neighbour_column

    def _toggle_hash_for_move(self, action: int, player: Player) -> None:
        old_player = self._to_play
        new_player = old_player.opponent
        player_index = self._player_index(player)
        self._zobrist_hash ^= self._side_keys[self._player_index(old_player)]
        self._zobrist_hash ^= self._piece_keys[action][player_index]
        self._zobrist_hash ^= self._side_keys[self._player_index(new_player)]
        self._zobrist_hash &= _UINT64_MASK

    @staticmethod
    def _player_index(player: Player) -> int:
        return player.value - 1


def encode_action(row: int, column: int, board_size: int) -> int:
    """Encode an in-range coordinate as a row-major action index."""

    _validate_board_size(board_size)
    if type(row) is not int or type(column) is not int:
        raise InvalidActionError("Row and column must be integers")
    if not 0 <= row < board_size or not 0 <= column < board_size:
        raise InvalidActionError(
            f"Coordinate must be within a {board_size} by {board_size} board"
        )
    return row * board_size + column


def decode_action(action: int, board_size: int) -> tuple[int, int]:
    """Decode an in-range row-major action into ``(row, column)``."""

    _validate_board_size(board_size)
    action_size = board_size * board_size
    if type(action) is not int or not 0 <= action < action_size:
        raise InvalidActionError(f"Action must be an integer in [0, {action_size - 1}]")
    return divmod(action, board_size)


def _validate_board_size(board_size: int) -> None:
    if type(board_size) is not int or board_size <= 0:
        raise InvalidActionError("Board size must be a positive integer")


@lru_cache(maxsize=16)
def _zobrist_keys(
    board_size: int, seed: int
) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    """Build and cache deterministic piece and side-to-play hash keys."""

    state = seed & _UINT64_MASK
    piece_keys: list[tuple[int, int]] = []
    for _ in range(board_size * board_size):
        state, black_key = _next_splitmix64(state)
        state, white_key = _next_splitmix64(state)
        piece_keys.append((black_key, white_key))

    state, black_side_key = _next_splitmix64(state)
    _, white_side_key = _next_splitmix64(state)
    return tuple(piece_keys), (black_side_key, white_side_key)


def _next_splitmix64(state: int) -> tuple[int, int]:
    """Advance SplitMix64 and return ``(new_state, mixed_value)``."""

    new_state = (state + _SPLITMIX_INCREMENT) & _UINT64_MASK
    value = new_state
    value = ((value ^ (value >> 30)) * _SPLITMIX_MULTIPLIER_1) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * _SPLITMIX_MULTIPLIER_2) & _UINT64_MASK
    value ^= value >> 31
    return new_state, value & _UINT64_MASK

"""Exact memoized win/draw/loss solver for small Hexadeca endgames."""

from __future__ import annotations

from dataclasses import dataclass

from config.schema import RulesConfig
from game import Board, score_terminal


@dataclass(frozen=True, slots=True)
class ExactEndgameResult:
    """One proven root outcome and every legal action's proven outcome."""

    outcome: int
    action_outcomes: tuple[tuple[int, int], ...]
    states_evaluated: int
    cache_hits: int


class ExactEndgameSolver:
    """Solve deterministic endgames by negamax with collision-free memoization."""

    __slots__ = ("_cache", "_cache_hits", "_rules", "_states_evaluated")

    def __init__(self, rules: RulesConfig) -> None:
        """Bind one solver and its reusable cache to an immutable ruleset."""

        self._rules = rules
        self._cache: dict[tuple[bytes, int], int] = {}
        self._states_evaluated = 0
        self._cache_hits = 0

    def solve(self, board: Board) -> ExactEndgameResult:
        """Return exact current-player outcomes without changing ``board``."""

        if board.rules != self._rules:
            raise ValueError("Endgame board rules do not match the solver rules")

        states_before = self._states_evaluated
        hits_before = self._cache_hits
        if board.is_terminal:
            outcome = self._solve_position(board)
            action_outcomes: tuple[tuple[int, int], ...] = ()
        else:
            resolved: list[tuple[int, int]] = []
            for action in board.legal_actions():
                board.apply(action)
                try:
                    action_outcome = -self._solve_position(board)
                finally:
                    board.undo()
                resolved.append((action, action_outcome))
            action_outcomes = tuple(resolved)
            outcome = max(value for _, value in action_outcomes)

        return ExactEndgameResult(
            outcome=outcome,
            action_outcomes=action_outcomes,
            states_evaluated=self._states_evaluated - states_before,
            cache_hits=self._cache_hits - hits_before,
        )

    def _solve_position(self, board: Board) -> int:
        key = (bytes(board.cells), board.to_play.value)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            return cached

        self._states_evaluated += 1
        if board.is_terminal:
            winner = score_terminal(board).winner
            value = 0 if winner is None else 1 if winner is board.to_play else -1
        else:
            value = -1
            for action in board.legal_actions():
                board.apply(action)
                try:
                    candidate = -self._solve_position(board)
                finally:
                    board.undo()
                value = max(value, candidate)
                if value == 1:
                    break

        self._cache[key] = value
        return value

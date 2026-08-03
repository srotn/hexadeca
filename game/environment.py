"""Self-play-ready lifecycle wrapper around board and scoring primitives."""

from __future__ import annotations

from dataclasses import dataclass

from config.schema import RulesConfig
from game.board import Board, Move, Player
from game.errors import NonTerminalBoardError, TerminalStateError
from game.scoring import ScoreResult, score_terminal


@dataclass(frozen=True, slots=True)
class GameState:
    """An immutable, serializable snapshot of one game position."""

    ruleset_id: str
    board_size: int
    cells: tuple[int, ...]
    to_play: Player
    ply: int
    history: tuple[Move, ...]
    legal_mask: tuple[bool, ...]
    zobrist_hash: int
    terminal: bool

    @property
    def action_size(self) -> int:
        """Return the number of placement actions represented by this state."""

        return len(self.cells)


@dataclass(frozen=True, slots=True)
class StepResult:
    """The immutable outcome of one successful environment action."""

    state: GameState
    move: Move
    result: ScoreResult | None


class GameEnvironment:
    """Own one game lifecycle for interactive play or a self-play worker.

    Instances are intentionally single-owner and not thread-safe. Consumers
    receive immutable snapshots and cannot mutate the underlying board except
    through ``step`` and ``reset``.
    """

    __slots__ = ("_board", "_rules", "_terminal_result")

    def __init__(self, rules: RulesConfig) -> None:
        """Create a new game using one validated immutable rules config."""

        self._rules = rules
        self._board = Board(rules)
        self._terminal_result: ScoreResult | None = None

    @property
    def rules(self) -> RulesConfig:
        """Return the immutable rules configuration for this environment."""

        return self._rules

    @property
    def state(self) -> GameState:
        """Return an immutable snapshot of the current position."""

        return GameState(
            ruleset_id=self._rules.ruleset_id,
            board_size=self._board.size,
            cells=self._board.cells,
            to_play=self._board.to_play,
            ply=self._board.ply,
            history=self._board.history,
            legal_mask=self._board.legal_mask(),
            zobrist_hash=self._board.zobrist_hash,
            terminal=self._board.is_terminal,
        )

    def reset(self) -> GameState:
        """Discard the current game and return a fresh initial state."""

        self._board = Board(self._rules)
        self._terminal_result = None
        return self.state

    def step(self, action: int) -> StepResult:
        """Apply one legal action and return its new state and optional result."""

        if self._board.is_terminal:
            raise TerminalStateError("Cannot step a terminal game")

        move = self._board.apply(action)
        result = score_terminal(self._board) if self._board.is_terminal else None
        self._terminal_result = result
        return StepResult(state=self.state, move=move, result=result)

    def legal_moves(self) -> tuple[int, ...]:
        """Return all currently legal action indices in stable row-major order."""

        return self._board.legal_actions()

    def terminal(self) -> bool:
        """Return whether this game has no legal placement remaining."""

        return self._board.is_terminal

    def result(self) -> ScoreResult:
        """Return the cached official result of a terminal game."""

        if not self._board.is_terminal:
            raise NonTerminalBoardError("A non-terminal game has no official result")
        if self._terminal_result is None:
            self._terminal_result = score_terminal(self._board)
        return self._terminal_result

    def winner(self) -> Player | None:
        """Return the terminal winner, or ``None`` for a final draw."""

        return self.result().winner

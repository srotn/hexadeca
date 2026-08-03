"""Domain exceptions raised by the Hexadeca board engine."""


class BoardError(Exception):
    """Base class for board-engine failures."""


class InvalidActionError(BoardError, ValueError):
    """Raised when an action or coordinate is outside its valid domain."""


class IllegalMoveError(BoardError):
    """Raised when an in-range action violates the placement rules."""


class UndoError(BoardError):
    """Raised when undo is requested without a move to restore."""


class ScoringError(BoardError):
    """Base class for terminal-scoring failures."""


class NonTerminalBoardError(ScoringError):
    """Raised when official scoring is requested before the game ends."""


class UnsupportedScoringRuleError(ScoringError):
    """Raised when a board requests scoring semantics not implemented here."""


class GameEnvironmentError(BoardError):
    """Base class for game-environment lifecycle failures."""


class TerminalStateError(GameEnvironmentError):
    """Raised when an action is requested after the game has ended."""

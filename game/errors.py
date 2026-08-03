"""Domain exceptions raised by the Hexadeca board engine."""


class BoardError(Exception):
    """Base class for board-engine failures."""


class InvalidActionError(BoardError, ValueError):
    """Raised when an action or coordinate is outside its valid domain."""


class IllegalMoveError(BoardError):
    """Raised when an in-range action violates the placement rules."""


class UndoError(BoardError):
    """Raised when undo is requested without a move to restore."""

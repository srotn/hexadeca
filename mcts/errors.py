"""Domain exceptions for neural Monte Carlo tree search."""


class MctsError(Exception):
    """Base class for Stage 7 search failures."""


class NodeStateError(MctsError, RuntimeError):
    """Raised when a tree node or edge violates its lifecycle."""


class InvalidEvaluationError(MctsError, ValueError):
    """Raised when a neural evaluator returns an invalid policy or value."""


class SearchStateError(MctsError, ValueError):
    """Raised when a root state cannot be searched safely."""


class VirtualLossError(MctsError, RuntimeError):
    """Raised when a virtual-loss reservation cannot be balanced."""

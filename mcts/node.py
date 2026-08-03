"""PUCT node, edge, virtual-loss, expansion, and backup primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isclose, isfinite, sqrt

from game import Player
from mcts.errors import NodeStateError, VirtualLossError


@dataclass(slots=True)
class MctsEdge:
    """One action edge with statistics from its parent player's perspective."""

    prior: float
    child: MctsNode | None = None
    visit_count: int = 0
    value_sum: float = 0.0
    virtual_visit_count: int = 0
    virtual_value_sum: float = 0.0

    def __post_init__(self) -> None:
        if not isfinite(self.prior) or not 0.0 <= self.prior <= 1.0:
            raise NodeStateError("Edge prior must be finite and within [0, 1]")

    @property
    def mean_value(self) -> float:
        """Return committed Q from the parent player's perspective."""

        return self.value_sum / self.visit_count if self.visit_count else 0.0

    @property
    def effective_visit_count(self) -> int:
        """Return committed plus in-flight visits used during selection."""

        return self.visit_count + self.virtual_visit_count

    @property
    def effective_mean_value(self) -> float:
        """Return Q after applying all current virtual-loss reservations."""

        visits = self.effective_visit_count
        if visits == 0:
            return 0.0
        return (self.value_sum - self.virtual_value_sum) / visits

    def reserve(self, virtual_loss: int) -> None:
        """Apply one reversible in-flight reservation to this edge."""

        if type(virtual_loss) is not int or virtual_loss <= 0:
            raise VirtualLossError("Virtual loss must be a positive integer")
        self.virtual_visit_count += 1
        self.virtual_value_sum += float(virtual_loss)

    def release(self, virtual_loss: int) -> None:
        """Remove exactly one prior in-flight reservation."""

        if type(virtual_loss) is not int or virtual_loss <= 0:
            raise VirtualLossError("Virtual loss must be a positive integer")
        if self.virtual_visit_count <= 0:
            raise VirtualLossError("Cannot release an unreserved edge")
        self.virtual_visit_count -= 1
        self.virtual_value_sum -= float(virtual_loss)
        if self.virtual_value_sum < -1e-12:
            raise VirtualLossError("Virtual-loss value accounting became negative")
        if self.virtual_visit_count == 0:
            self.virtual_value_sum = 0.0

    def record(self, value: float) -> None:
        """Commit one bounded backed-up value for the parent player."""

        _validate_value(value)
        self.visit_count += 1
        self.value_sum += value


@dataclass(slots=True)
class MctsNode:
    """One canonical tree position with lazily materialized child nodes."""

    to_play: Player
    zobrist_hash: int
    terminal: bool
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, MctsEdge] = field(default_factory=dict)
    expanded: bool = False
    evaluation_in_flight: bool = False

    @property
    def mean_value(self) -> float:
        """Return committed Q from this node's side-to-play perspective."""

        return self.value_sum / self.visit_count if self.visit_count else 0.0

    def expand(self, priors: dict[int, float]) -> None:
        """Create one legal edge per normalized prior exactly once."""

        if self.terminal:
            raise NodeStateError("A terminal node cannot be expanded")
        if self.expanded:
            raise NodeStateError("A node cannot be expanded twice")
        if not priors:
            raise NodeStateError("A non-terminal node requires legal priors")
        if any(type(action) is not int or action < 0 for action in priors):
            raise NodeStateError("Expansion actions must be nonnegative integers")
        if any(not isfinite(prior) or prior < 0.0 for prior in priors.values()):
            raise NodeStateError("Expansion priors must be finite and nonnegative")
        if not isclose(sum(priors.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise NodeStateError("Expansion priors must sum to one")
        self.children = {
            action: MctsEdge(prior=prior) for action, prior in sorted(priors.items())
        }
        self.expanded = True

    def select_edge(self, c_puct: float) -> tuple[int, MctsEdge] | None:
        """Select the highest PUCT edge with a stable lower-action tie-break."""

        if not self.expanded or not self.children:
            raise NodeStateError("Only an expanded non-terminal node can select")
        if not isfinite(c_puct) or c_puct <= 0:
            raise NodeStateError("c_puct must be finite and positive")

        virtual_visits = sum(
            edge.virtual_visit_count for edge in self.children.values()
        )
        parent_visits = max(1, self.visit_count + virtual_visits)
        exploration_scale = c_puct * sqrt(parent_visits)
        selected: tuple[int, MctsEdge] | None = None
        selected_score = float("-inf")
        for action, edge in self.children.items():
            child = edge.child
            if child is not None and child.evaluation_in_flight and not child.expanded:
                continue
            score = edge.effective_mean_value + (
                exploration_scale * edge.prior / (1 + edge.effective_visit_count)
            )
            if score > selected_score:
                selected = (action, edge)
                selected_score = score
        return selected

    def record(self, value: float) -> None:
        """Commit one value from this node's side-to-play perspective."""

        _validate_value(value)
        self.visit_count += 1
        self.value_sum += value


def backup_path(
    nodes: tuple[MctsNode, ...],
    edges: tuple[MctsEdge, ...],
    leaf_value: float,
    *,
    virtual_loss: int,
) -> None:
    """Release reservations and back up a leaf value with alternating perspective."""

    if len(nodes) != len(edges) + 1:
        raise NodeStateError("A backup path must contain one more node than edge")
    _validate_value(leaf_value)
    if any(
        edge.virtual_visit_count <= 0
        or edge.virtual_value_sum + 1e-12 < float(virtual_loss)
        for edge in edges
    ):
        raise VirtualLossError("Every backed-up edge must hold a reservation")

    value = leaf_value
    nodes[-1].record(value)
    for index in range(len(edges) - 1, -1, -1):
        edge = edges[index]
        edge.release(virtual_loss)
        value = -value
        edge.record(value)
        nodes[index].record(value)


def cancel_path(edges: tuple[MctsEdge, ...], *, virtual_loss: int) -> None:
    """Release every reservation without committing search statistics."""

    for edge in reversed(edges):
        edge.release(virtual_loss)


def _validate_value(value: float) -> None:
    if not isfinite(value) or not -1.0 <= value <= 1.0:
        raise NodeStateError("Backed-up values must be finite and within [-1, 1]")

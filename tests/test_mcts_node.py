"""Unit tests for PUCT statistics, virtual loss, and perspective backup."""

from __future__ import annotations

import pytest

from game import Player
from mcts import (
    MctsEdge,
    MctsNode,
    NodeStateError,
    VirtualLossError,
    backup_path,
)


def test_virtual_loss_is_reversible_and_separate_from_committed_statistics() -> None:
    """An in-flight reservation changes selection Q but not committed Q."""

    edge = MctsEdge(prior=0.5)
    edge.record(0.5)

    edge.reserve(3)

    assert edge.visit_count == 1
    assert edge.mean_value == pytest.approx(0.5)
    assert edge.effective_visit_count == 2
    assert edge.effective_mean_value == pytest.approx(-1.25)

    edge.release(3)

    assert edge.virtual_visit_count == 0
    assert edge.virtual_value_sum == 0.0
    assert edge.effective_mean_value == pytest.approx(0.5)
    with pytest.raises(VirtualLossError, match="unreserved"):
        edge.release(3)


def test_puct_uses_prior_then_virtual_loss_with_stable_action_ties() -> None:
    """Initial ties choose lower actions and reservations diversify a batch."""

    node = MctsNode(Player.BLACK, zobrist_hash=1, terminal=False)
    node.expand({2: 0.5, 4: 0.5})

    first = node.select_edge(c_puct=2.3)
    assert first is not None
    assert first[0] == 2
    first[1].reserve(3)

    second = node.select_edge(c_puct=2.3)
    assert second is not None
    assert second[0] == 4


def test_backup_alternates_value_perspective_and_clears_reservations() -> None:
    """Each player transition negates value exactly once during backup."""

    root = MctsNode(Player.BLACK, zobrist_hash=1, terminal=False)
    child = MctsNode(Player.WHITE, zobrist_hash=2, terminal=False)
    leaf = MctsNode(Player.BLACK, zobrist_hash=3, terminal=False)
    first = MctsEdge(prior=1.0, child=child)
    second = MctsEdge(prior=1.0, child=leaf)
    first.reserve(3)
    second.reserve(3)

    backup_path(
        (root, child, leaf),
        (first, second),
        0.75,
        virtual_loss=3,
    )

    assert leaf.mean_value == pytest.approx(0.75)
    assert second.mean_value == pytest.approx(-0.75)
    assert child.mean_value == pytest.approx(-0.75)
    assert first.mean_value == pytest.approx(0.75)
    assert root.mean_value == pytest.approx(0.75)
    assert first.virtual_visit_count == second.virtual_visit_count == 0


def test_backup_rejects_missing_reservation_before_mutating_nodes() -> None:
    """An invalid path fails atomically before any committed visit is written."""

    root = MctsNode(Player.BLACK, zobrist_hash=1, terminal=False)
    leaf = MctsNode(Player.WHITE, zobrist_hash=2, terminal=False)
    edge = MctsEdge(prior=1.0, child=leaf)

    with pytest.raises(VirtualLossError, match="reservation"):
        backup_path((root, leaf), (edge,), 0.5, virtual_loss=3)

    assert root.visit_count == leaf.visit_count == edge.visit_count == 0


def test_node_rejects_non_normalized_expansion() -> None:
    """Expansion cannot silently renormalize malformed neural priors."""

    node = MctsNode(Player.BLACK, zobrist_hash=1, terminal=False)

    with pytest.raises(NodeStateError, match="sum to one"):
        node.expand({0: 0.2, 1: 0.2})

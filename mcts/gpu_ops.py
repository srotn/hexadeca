"""CUDA-side tensor operations used by the native MCTS boundary.

The native tree remains the owner of mutable search state (nodes, boards,
virtual loss and backup).  This module moves the dense, embarrassingly
parallel part of the MCTS boundary to CUDA: root PUCT scoring and visit-target
normalisation.  It is deliberately optional so the reference engine and CPU
diagnostics keep exactly the old behaviour.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from time import perf_counter

import torch


@dataclass(slots=True)
class GpuMctsStats:
    """Lifetime counters for the CUDA MCTS tensor path."""

    calls: int = 0
    visit_policy_calls: int = 0
    puct_calls: int = 0
    positions: int = 0
    visit_policy_positions: int = 0
    puct_positions: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "calls": self.calls,
            "visit_policy_calls": self.visit_policy_calls,
            "puct_calls": self.puct_calls,
            "positions": self.positions,
            "visit_policy_positions": self.visit_policy_positions,
            "puct_positions": self.puct_positions,
            "elapsed_seconds": self.elapsed_seconds,
        }


class GpuMctsOps:
    """Run dense MCTS arithmetic on one CUDA device.

    The operation is intentionally small and deterministic.  C++ still
    selects/reserves tree paths, so no mutable tree state is duplicated on the
    device.  Keeping this boundary explicit makes it possible to replace the
    tree with a fully CUDA-resident implementation later without changing the
    evaluator contract.
    """

    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self.enabled = self.device.type == "cuda" and torch.cuda.is_available()
        self.stats = GpuMctsStats()

    def visit_probabilities(
        self, visit_counts: Sequence[int], temperature: float
    ) -> tuple[float, ...]:
        """Convert visit counts to a policy using CUDA tensor arithmetic."""

        if not self.enabled:
            raise RuntimeError("CUDA MCTS operations are not enabled")
        if not visit_counts or any(
            type(count) is not int or count < 0 for count in visit_counts
        ):
            raise ValueError("visit counts must be non-empty nonnegative integers")
        if not isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        started = perf_counter()
        counts = torch.as_tensor(visit_counts, dtype=torch.float32, device=self.device)
        maximum = counts.max()
        if float(maximum.item()) <= 0:
            raise ValueError("at least one action must have a visit")
        if temperature == 0:
            selected = torch.argmax(counts)
            policy = torch.zeros_like(counts)
            policy[selected] = 1.0
        else:
            weights = torch.where(
                counts > 0,
                torch.pow(counts / maximum, 1.0 / temperature),
                torch.zeros_like(counts),
            )
            policy = weights / weights.sum()
        # The copy synchronises the short CUDA kernel before C++ receives the
        # Python tuple.  All arithmetic above (including argmax) is GPU work.
        result = tuple(float(value) for value in policy.cpu().tolist())
        self.stats.calls += 1
        self.stats.visit_policy_calls += 1
        self.stats.positions += len(visit_counts)
        self.stats.visit_policy_positions += len(visit_counts)
        self.stats.elapsed_seconds += perf_counter() - started
        return result

    def puct_scores(
        self,
        *,
        priors: Sequence[float],
        visit_counts: Sequence[int],
        value_sums: Sequence[float],
        virtual_visit_counts: Sequence[int],
        virtual_value_sums: Sequence[float],
        solved_values: Sequence[float] | None,
        parent_visit_count: int,
        parent_value_sum: float,
        c_puct: float,
        fpu_reduction: float,
    ) -> tuple[float, ...]:
        """Compute one node's complete PUCT score vector on CUDA.

        This mirrors ``NativeSearchTree::select_edge``.  The native tree still
        performs the authoritative reservation and tie-breaking; the result is
        used for the root recommendation/diagnostics and is therefore safe to
        enable without changing tree consistency.
        """

        if not self.enabled:
            raise RuntimeError("CUDA MCTS operations are not enabled")
        length = len(priors)
        if not (
            length
            and len(visit_counts) == length
            and len(value_sums) == length
            and len(virtual_visit_counts) == length
            and len(virtual_value_sums) == length
        ):
            raise ValueError("PUCT vectors must have equal non-zero length")
        if solved_values is None:
            solved_values = [float("nan")] * length
        if len(solved_values) != length:
            raise ValueError("solved_values length mismatch")
        started = perf_counter()
        kwargs = {"dtype": torch.float32, "device": self.device}
        prior = torch.as_tensor(priors, **kwargs)
        visits = torch.as_tensor(visit_counts, **kwargs)
        values = torch.as_tensor(value_sums, **kwargs)
        virtual_visits = torch.as_tensor(virtual_visit_counts, **kwargs)
        virtual_values = torch.as_tensor(virtual_value_sums, **kwargs)
        solved = torch.as_tensor(solved_values, **kwargs)
        effective_visits = visits + virtual_visits
        parent_visits = max(
            1.0,
            float(parent_visit_count) + float(virtual_visits.sum().item()),
        )
        parent_mean = (
            float(parent_value_sum) / float(parent_visit_count)
            if parent_visit_count > 0
            else 0.0
        )
        effective_value = torch.where(
            torch.isfinite(solved) & (virtual_visits == 0),
            solved,
            torch.where(
                effective_visits == 0,
                torch.full_like(effective_visits, parent_mean - fpu_reduction),
                (values - virtual_values) / effective_visits,
            ),
        )
        scale = float(c_puct) * torch.sqrt(
            torch.as_tensor(parent_visits, dtype=torch.float32, device=self.device)
        )
        scores = effective_value + scale * prior / (1.0 + effective_visits)
        result = tuple(float(value) for value in scores.cpu().tolist())
        self.stats.puct_calls += 1
        self.stats.positions += length
        self.stats.puct_positions += length
        self.stats.elapsed_seconds += perf_counter() - started
        return result


__all__ = ["GpuMctsOps", "GpuMctsStats"]

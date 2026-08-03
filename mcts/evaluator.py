"""Batched evaluator contract and PyTorch policy-value implementation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from game import GameState
from mcts.errors import InvalidEvaluationError
from network import (
    NetworkSpecification,
    PolicyValueNetwork,
    encode_batch,
    masked_policy_probabilities,
)


@dataclass(frozen=True, slots=True)
class Evaluation:
    """One masked policy and current-player value in ``[-1, 1]``."""

    policy: tuple[float, ...]
    value: float


class BatchEvaluator(Protocol):
    """Interface used by MCTS to evaluate one non-empty state batch."""

    def evaluate(self, states: Sequence[GameState]) -> Sequence[Evaluation]:
        """Return evaluations in the exact input order."""


class TorchBatchEvaluator:
    """Run version-compatible PyTorch inference with optional CUDA AMP."""

    def __init__(
        self,
        model: PolicyValueNetwork,
        specification: NetworkSpecification,
        *,
        device: torch.device | str,
        use_amp: bool,
    ) -> None:
        specification.validate()
        model.specification.ensure_compatible(specification)
        self._device = torch.device(device)
        if use_amp and self._device.type != "cuda":
            raise InvalidEvaluationError("AMP inference currently requires CUDA")
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise InvalidEvaluationError("CUDA inference requested but unavailable")
        self._model = model.to(self._device).eval()
        self._specification = specification
        self._use_amp = use_amp
        self._batch_count = 0
        self._position_count = 0

    @property
    def batch_count(self) -> int:
        """Return the lifetime number of completed evaluator calls."""

        return self._batch_count

    @property
    def position_count(self) -> int:
        """Return the lifetime number of evaluated positions."""

        return self._position_count

    def evaluate(self, states: Sequence[GameState]) -> tuple[Evaluation, ...]:
        """Encode, infer, mask, and transfer one state batch to CPU values."""

        if not states:
            raise InvalidEvaluationError("Cannot evaluate an empty state batch")
        features = encode_batch(states, self._specification, device=self._device)
        legal_mask = torch.tensor(
            [state.legal_mask for state in states],
            dtype=torch.bool,
            device=self._device,
        )

        with torch.inference_mode():
            if self._use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    policy, values = self._forward(features, legal_mask)
            else:
                policy, values = self._forward(features, legal_mask)

        cpu_policy = policy.cpu()
        cpu_values = values.cpu()
        self._batch_count += 1
        self._position_count += len(states)
        return tuple(
            Evaluation(
                policy=tuple(float(value) for value in cpu_policy[index].tolist()),
                value=float(cpu_values[index].item()),
            )
            for index in range(len(states))
        )

    def _forward(
        self, features: torch.Tensor, legal_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self._model(features)
        policy = masked_policy_probabilities(output.policy_logits.float(), legal_mask)
        values = output.win_probability.float() * 2.0 - 1.0
        return policy, values

"""Composite AlphaZero training objective for Stage 6 outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as functional

from config.schema import TrainingConfig
from network.errors import TrainingTargetError
from network.model import NetworkOutput
from network.policy import (
    masked_policy_log_probabilities,
    masked_policy_probabilities,
)


@dataclass(frozen=True, slots=True)
class LossWeights:
    """Explicit coefficients for every learned objective."""

    policy: float
    win: float
    black_score: float
    white_score: float

    @classmethod
    def from_config(cls, config: TrainingConfig) -> LossWeights:
        """Read loss coefficients from the unified training config."""

        weights = cls(
            policy=config.policy_loss_weight,
            win=config.win_loss_weight,
            black_score=config.black_score_loss_weight,
            white_score=config.white_score_loss_weight,
        )
        weights.validate()
        return weights

    def validate(self) -> None:
        """Require finite nonnegative weights and one active objective."""

        values = (self.policy, self.win, self.black_score, self.white_score)
        if any(not torch.isfinite(torch.tensor(value)).item() for value in values):
            raise TrainingTargetError("Loss weights must be finite")
        if any(value < 0 for value in values):
            raise TrainingTargetError("Loss weights must be nonnegative")
        if not any(value > 0 for value in values):
            raise TrainingTargetError("At least one loss weight must be positive")


class TrainingTargets(NamedTuple):
    """Batched visit, legal, outcome, and normalized score targets."""

    policy: torch.Tensor
    legal_mask: torch.Tensor
    win: torch.Tensor
    black_score: torch.Tensor
    white_score: torch.Tensor


class LossOutput(NamedTuple):
    """Total loss and detached-friendly component metrics."""

    total: torch.Tensor
    policy: torch.Tensor
    win: torch.Tensor
    black_score: torch.Tensor
    white_score: torch.Tensor
    policy_entropy: torch.Tensor
    target_policy_entropy: torch.Tensor
    policy_kl_divergence: torch.Tensor


class AlphaZeroLoss(nn.Module):
    """Policy CE, outcome BCE, and two normalized score MSE objectives."""

    def __init__(self, weights: LossWeights) -> None:
        super().__init__()
        weights.validate()
        self.weights = weights

    def forward(self, output: NetworkOutput, targets: TrainingTargets) -> LossOutput:
        """Validate targets and compute the configured weighted total."""

        _validate_targets(output, targets)
        log_probabilities = masked_policy_log_probabilities(
            output.policy_logits, targets.legal_mask
        )
        policy_loss = -torch.sum(targets.policy * log_probabilities, dim=1).mean()
        win_loss = functional.binary_cross_entropy_with_logits(
            output.win_logit, targets.win
        )
        black_score_loss = functional.mse_loss(output.black_score, targets.black_score)
        white_score_loss = functional.mse_loss(output.white_score, targets.white_score)
        total = (
            self.weights.policy * policy_loss
            + self.weights.win * win_loss
            + self.weights.black_score * black_score_loss
            + self.weights.white_score * white_score_loss
        )
        probabilities = masked_policy_probabilities(
            output.policy_logits, targets.legal_mask
        )
        policy_entropy = -torch.sum(probabilities * log_probabilities, dim=1).mean()
        target_policy_entropy = -torch.sum(
            torch.xlogy(targets.policy, targets.policy), dim=1
        ).mean()
        policy_kl_divergence = policy_loss - target_policy_entropy
        return LossOutput(
            total=total,
            policy=policy_loss,
            win=win_loss,
            black_score=black_score_loss,
            white_score=white_score_loss,
            policy_entropy=policy_entropy,
            target_policy_entropy=target_policy_entropy,
            policy_kl_divergence=policy_kl_divergence,
        )


def _validate_targets(output: NetworkOutput, targets: TrainingTargets) -> None:
    logits = output.policy_logits
    if logits.ndim != 2:
        raise TrainingTargetError("Policy output must have shape [N, action_size]")
    batch_size, action_size = logits.shape
    if targets.policy.shape != (batch_size, action_size):
        raise TrainingTargetError("Policy target shape must match policy output")
    if targets.legal_mask.shape != (batch_size, action_size):
        raise TrainingTargetError("Legal mask shape must match policy output")
    if targets.policy.device != logits.device:
        raise TrainingTargetError("Policy targets and outputs must share a device")
    if targets.legal_mask.device != logits.device:
        raise TrainingTargetError("Legal masks and outputs must share a device")
    if not targets.policy.is_floating_point():
        raise TrainingTargetError("Policy targets must use a floating dtype")
    if targets.legal_mask.dtype is not torch.bool:
        raise TrainingTargetError("Legal masks must use torch.bool")
    if not torch.all(torch.isfinite(targets.policy)).item():
        raise TrainingTargetError("Policy targets must be finite")
    if torch.any(targets.policy < 0).item():
        raise TrainingTargetError("Policy targets must be nonnegative")
    if torch.any(targets.policy.masked_select(~targets.legal_mask) != 0).item():
        raise TrainingTargetError(
            "Policy targets must assign zero mass to illegal moves"
        )
    target_mass = targets.policy.sum(dim=1)
    if not torch.allclose(target_mass, torch.ones_like(target_mass), atol=1e-6):
        raise TrainingTargetError("Each policy target must sum to one")

    scalar_pairs = (
        ("win", output.win_logit, targets.win),
        ("black_score", output.black_score, targets.black_score),
        ("white_score", output.white_score, targets.white_score),
    )
    for name, prediction, target in scalar_pairs:
        if prediction.shape != (batch_size,) or target.shape != (batch_size,):
            raise TrainingTargetError(f"{name} values must have shape [N]")
        if target.device != prediction.device:
            raise TrainingTargetError(f"{name} targets and outputs must share a device")
        if (
            not target.is_floating_point()
            or not torch.all(torch.isfinite(target)).item()
        ):
            raise TrainingTargetError(f"{name} targets must be finite floating values")
        if torch.any((target < 0) | (target > 1)).item():
            raise TrainingTargetError(f"{name} targets must be normalized to [0, 1]")

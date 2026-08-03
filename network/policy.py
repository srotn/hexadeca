"""Numerically stable legal-action masking for policy logits."""

from __future__ import annotations

import torch

from network.errors import PolicyMaskError


def mask_policy_logits(
    policy_logits: torch.Tensor, legal_mask: torch.Tensor
) -> torch.Tensor:
    """Replace illegal logits while requiring one legal action per sample."""

    _validate_policy_mask(policy_logits, legal_mask)
    return policy_logits.masked_fill(~legal_mask, torch.finfo(policy_logits.dtype).min)


def masked_policy_log_probabilities(
    policy_logits: torch.Tensor, legal_mask: torch.Tensor
) -> torch.Tensor:
    """Return legal-action log probabilities with illegal mass equal to zero."""

    return torch.log_softmax(mask_policy_logits(policy_logits, legal_mask), dim=1)


def masked_policy_probabilities(
    policy_logits: torch.Tensor, legal_mask: torch.Tensor
) -> torch.Tensor:
    """Return a normalized legal-action probability distribution."""

    probabilities = torch.softmax(mask_policy_logits(policy_logits, legal_mask), dim=1)
    return probabilities.masked_fill(~legal_mask, 0.0)


def _validate_policy_mask(
    policy_logits: torch.Tensor, legal_mask: torch.Tensor
) -> None:
    if policy_logits.ndim != 2:
        raise PolicyMaskError("Policy logits must have shape [N, action_size]")
    if not policy_logits.is_floating_point():
        raise PolicyMaskError("Policy logits must use a floating dtype")
    if legal_mask.dtype is not torch.bool:
        raise PolicyMaskError("Legal mask must use torch.bool")
    if legal_mask.shape != policy_logits.shape:
        raise PolicyMaskError("Legal mask shape must equal policy logits shape")
    if legal_mask.device != policy_logits.device:
        raise PolicyMaskError("Legal mask and logits must use the same device")
    if not torch.all(torch.any(legal_mask, dim=1)).item():
        raise PolicyMaskError("Every policy sample must have at least one legal action")

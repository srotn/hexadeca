"""Tests for legal policy normalization and the composite objective."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from config import load_config
from network import (
    AlphaZeroLoss,
    LossWeights,
    NetworkOutput,
    NetworkSpecification,
    PolicyMaskError,
    PolicyValueNetwork,
    TrainingTargetError,
    TrainingTargets,
    masked_policy_probabilities,
)


def test_policy_mask_assigns_exactly_zero_illegal_probability() -> None:
    """An arbitrarily high illegal logit cannot receive probability mass."""

    logits = torch.tensor([[1.0, 100.0, 3.0]])
    legal_mask = torch.tensor([[True, False, True]])

    probabilities = masked_policy_probabilities(logits, legal_mask)

    assert probabilities[0, 1].item() == 0.0
    assert probabilities.sum().item() == pytest.approx(1.0)
    assert probabilities[0, 2] > probabilities[0, 0]


def test_policy_mask_rejects_terminal_samples() -> None:
    """MCTS and loss must not evaluate positions without a legal action."""

    with pytest.raises(PolicyMaskError, match="at least one legal"):
        masked_policy_probabilities(
            torch.zeros(1, 4), torch.zeros(1, 4, dtype=torch.bool)
        )


def test_composite_loss_is_finite_and_backpropagates() -> None:
    """All four configured objectives contribute to differentiable training."""

    logits = torch.tensor(
        [[0.1, 0.2, 9.0, -0.1], [0.3, 8.0, 0.1, 0.4]], requires_grad=True
    )
    raw_values = torch.tensor([[0.2, -0.3, 0.5], [-0.4, 0.6, -0.2]], requires_grad=True)
    values = torch.sigmoid(raw_values)
    output = NetworkOutput(
        policy_logits=logits,
        black_score=values[:, 0],
        white_score=values[:, 1],
        win_logit=raw_values[:, 2],
    )
    targets = TrainingTargets(
        policy=torch.tensor([[0.75, 0.25, 0.0, 0.0], [0.25, 0.0, 0.25, 0.5]]),
        legal_mask=torch.tensor(
            [[True, True, False, False], [True, False, True, True]]
        ),
        win=torch.tensor([1.0, 0.0]),
        black_score=torch.tensor([0.25, 0.75]),
        white_score=torch.tensor([0.75, 0.25]),
    )
    weights = LossWeights.from_config(load_config().training)

    result = AlphaZeroLoss(weights)(output, targets)
    result.total.backward()

    expected_total = (
        result.policy + result.win + result.black_score + result.white_score
    )
    assert result.total.item() == pytest.approx(expected_total.item())
    assert result.policy_entropy.item() > 0
    expected_target_entropy = -torch.sum(
        torch.xlogy(targets.policy, targets.policy), dim=1
    ).mean()
    assert result.target_policy_entropy.item() == pytest.approx(
        expected_target_entropy.item()
    )
    assert result.policy_kl_divergence.item() == pytest.approx(
        (result.policy - result.target_policy_entropy).item()
    )
    assert logits.grad is not None
    assert logits.grad[0, 2].item() == 0.0
    assert logits.grad[0, 3].item() == 0.0
    assert logits.grad[1, 1].item() == 0.0
    assert raw_values.grad is not None


def test_loss_rejects_illegal_policy_mass_and_raw_scores() -> None:
    """Replay targets must already be legal distributions normalized to [0, 1]."""

    output = NetworkOutput(
        policy_logits=torch.zeros(1, 2),
        black_score=torch.tensor([0.5]),
        white_score=torch.tensor([0.5]),
        win_logit=torch.tensor([0.0]),
    )
    weights = LossWeights(1.0, 1.0, 1.0, 1.0)
    illegal_policy = TrainingTargets(
        policy=torch.tensor([[0.5, 0.5]]),
        legal_mask=torch.tensor([[True, False]]),
        win=torch.tensor([0.5]),
        black_score=torch.tensor([0.5]),
        white_score=torch.tensor([0.5]),
    )
    with pytest.raises(TrainingTargetError, match="illegal"):
        AlphaZeroLoss(weights)(output, illegal_policy)

    raw_score = illegal_policy._replace(
        policy=torch.tensor([[1.0, 0.0]]), black_score=torch.tensor([27.0])
    )
    with pytest.raises(TrainingTargetError, match="normalized"):
        AlphaZeroLoss(weights)(output, raw_score)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_amp_forward_loss_and_backward_are_supported() -> None:
    """The win objective uses the autocast-safe logits formulation."""

    config = load_config()
    specification = replace(
        NetworkSpecification.from_config(config.rules, config.network),
        residual_blocks=1,
        channels=8,
        value_hidden_features=16,
    )
    model = PolicyValueNetwork(specification).to("cuda").train()
    inputs = torch.randn(2, 16, 16, 16, device="cuda")
    targets = TrainingTargets(
        policy=torch.full((2, 256), 1.0 / 256, device="cuda"),
        legal_mask=torch.ones(2, 256, dtype=torch.bool, device="cuda"),
        win=torch.tensor([1.0, 0.0], device="cuda"),
        black_score=torch.tensor([0.2, 0.7], device="cuda"),
        white_score=torch.tensor([0.8, 0.3], device="cuda"),
    )

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(inputs)
        result = AlphaZeroLoss(LossWeights.from_config(config.training))(
            output, targets
        )
    result.total.backward()

    assert torch.isfinite(result.total)
    assert model.stem[0].weight.grad is not None

"""Tests for real CPU and CUDA batched policy-value evaluation."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from config import load_config
from game import GameEnvironment
from mcts import MctsSearch, SearchMode, TorchBatchEvaluator
from network import NetworkSpecification, PolicyValueNetwork, encode_batch
from training.inference_transport import pack_state_batch


def _small_model() -> tuple[PolicyValueNetwork, NetworkSpecification]:
    config = load_config()
    specification = replace(
        NetworkSpecification.from_config(config.rules, config.network),
        residual_blocks=1,
        channels=8,
        value_hidden_features=16,
    )
    return PolicyValueNetwork(specification), specification


def test_torch_evaluator_masks_and_batches_cpu_states() -> None:
    """The real evaluator returns legal normalized policy and bounded values."""

    config = load_config()
    model, specification = _small_model()
    environment = GameEnvironment(config.rules)
    initial = environment.state
    environment.step(0)
    after_move = environment.state
    evaluator = TorchBatchEvaluator(model, specification, device="cpu", use_amp=False)

    evaluations = evaluator.evaluate((initial, after_move))

    assert evaluator.batch_count == 1
    assert evaluator.position_count == 2
    for evaluation, state in zip(evaluations, (initial, after_move), strict=True):
        assert sum(evaluation.policy) == pytest.approx(1.0)
        assert all(
            probability == 0.0
            for probability, legal in zip(
                evaluation.policy, state.legal_mask, strict=True
            )
            if not legal
        )
        assert -1.0 <= evaluation.value <= 1.0


def test_torch_evaluator_exposes_denormalized_score_predictions() -> None:
    """Interactive diagnostics receive Black/White estimates in board points."""

    config = load_config()
    model, specification = _small_model()
    state = GameEnvironment(config.rules).state
    evaluator = TorchBatchEvaluator(model, specification, device="cpu", use_amp=False)

    prediction = evaluator.predict_scores((state,))[0]

    assert 0.0 <= prediction.black_score <= specification.score_normalizer
    assert 0.0 <= prediction.white_score <= specification.score_normalizer
    assert prediction.margin == pytest.approx(
        prediction.black_score - prediction.white_score
    )


def test_torch_evaluator_masks_packed_cpu_states(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compact NCHW features must flatten the legal plane for policy masking."""

    config = load_config()
    model, specification = _small_model()
    environment = GameEnvironment(config.rules)
    initial = environment.state
    environment.step(0)
    after_move = environment.state
    states = (initial, after_move)
    features = encode_batch(states, specification)

    monkeypatch.setattr(
        "mcts.evaluator.encode_packed_batch",
        lambda payload, resolved_specification, *, device: features.to(device=device),
    )
    evaluator = TorchBatchEvaluator(model, specification, device="cpu", use_amp=False)

    evaluations = evaluator.evaluate_packed(pack_state_batch(states))

    assert evaluator.batch_count == 1
    assert evaluator.position_count == len(states)
    for evaluation, state in zip(evaluations, states, strict=True):
        assert sum(evaluation.policy) == pytest.approx(1.0)
        assert all(
            probability == 0.0
            for probability, legal in zip(
                evaluation.policy, state.legal_mask, strict=True
            )
            if not legal
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_amp_evaluator_integrates_with_batched_search() -> None:
    """CUDA AMP supplies legal leaf batches to a complete MCTS search."""

    config = load_config()
    model, specification = _small_model()
    mcts_config = replace(
        config.mcts,
        evaluation_simulations=8,
        max_inference_batch_size=4,
    )
    evaluator = TorchBatchEvaluator(model, specification, device="cuda", use_amp=True)

    result = MctsSearch(config.rules, mcts_config, evaluator).run(
        GameEnvironment(config.rules).state, SearchMode.EVALUATION
    )

    assert result.simulations == 8
    assert sum(result.visit_counts) == 8
    assert evaluator.batch_count == result.inference_batches
    assert evaluator.position_count == result.inference_positions
    assert result.maximum_batch_size == 4

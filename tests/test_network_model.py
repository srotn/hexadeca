"""Shape, gradient, metadata, CPU, CUDA, and AMP network tests."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from config import load_config
from network import (
    NetworkInputError,
    NetworkSpecification,
    NetworkSpecificationError,
    PolicyValueNetwork,
)


def _small_specification() -> NetworkSpecification:
    config = load_config()
    default = NetworkSpecification.from_config(config.rules, config.network)
    return replace(
        default,
        residual_blocks=1,
        channels=8,
        policy_head_channels=2,
        value_head_channels=1,
        value_hidden_features=16,
    )


def test_default_model_matches_configured_architecture() -> None:
    """The production model is built solely from unified configuration."""

    config = load_config()
    specification = NetworkSpecification.from_config(config.rules, config.network)
    model = PolicyValueNetwork(specification)

    assert len(model.residual_tower) == 10
    assert model.policy_linear.out_features == 256
    assert model.value_output.out_features == 3
    assert sum(parameter.numel() for parameter in model.parameters()) == 3_171_209


def test_cpu_forward_shapes_ranges_and_gradients() -> None:
    """All heads have stable batch shapes and support shared-trunk backprop."""

    torch.manual_seed(11)
    specification = _small_specification()
    model = PolicyValueNetwork(specification)
    inputs = torch.randn(2, 16, 16, 16)

    output = model(inputs)
    objective = (
        output.policy_logits.mean()
        + output.black_score.mean()
        + output.white_score.mean()
        + output.win_probability.mean()
    )
    objective.backward()

    assert output.policy_logits.shape == (2, 256)
    assert output.black_score.shape == (2,)
    assert output.white_score.shape == (2,)
    assert output.win_probability.shape == (2,)
    assert torch.all((output.black_score >= 0) & (output.black_score <= 1))
    assert torch.all((output.white_score >= 0) & (output.white_score <= 1))
    assert torch.all((output.win_probability >= 0) & (output.win_probability <= 1))
    assert model.stem[0].weight.grad is not None


def test_model_rejects_wrong_tensor_shape_and_dtype() -> None:
    """Malformed inputs fail at the network boundary with domain errors."""

    model = PolicyValueNetwork(_small_specification())

    with pytest.raises(NetworkInputError, match="Expected input shape"):
        model(torch.zeros(1, 15, 16, 16))
    with pytest.raises(NetworkInputError, match="floating"):
        model(torch.zeros(1, 16, 16, 16, dtype=torch.int64))
    with pytest.raises(NetworkInputError, match="must not be empty"):
        model(torch.zeros(0, 16, 16, 16))


def test_network_specification_round_trip_and_mismatch_detection() -> None:
    """Checkpoint metadata round-trips exactly and rejects architecture drift."""

    specification = _small_specification()
    restored = NetworkSpecification.from_dict(specification.to_dict())

    assert restored == specification
    specification.ensure_compatible(restored)
    with pytest.raises(NetworkSpecificationError, match="channels"):
        specification.ensure_compatible(replace(restored, channels=16))
    invalid = specification.to_dict()
    invalid["unknown"] = 1
    with pytest.raises(NetworkSpecificationError, match="unexpected"):
        NetworkSpecification.from_dict(invalid)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_forward_and_amp_are_finite() -> None:
    """The configured CUDA device runs both FP32 and automatic mixed precision."""

    specification = _small_specification()
    model = PolicyValueNetwork(specification).to("cuda").eval()
    inputs = torch.randn(4, 16, 16, 16, device="cuda")

    with torch.inference_mode():
        full_precision = model(inputs)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            mixed_precision = model(inputs)

    assert full_precision.policy_logits.device.type == "cuda"
    assert mixed_precision.policy_logits.dtype is torch.float16
    assert torch.all(torch.isfinite(full_precision.policy_logits))
    assert torch.all(torch.isfinite(mixed_precision.policy_logits))

"""Versioned features and residual policy-value network for Hexadeca."""

from network.errors import (
    FeatureEncodingError,
    NetworkError,
    NetworkInputError,
    NetworkSpecificationError,
    PolicyMaskError,
    TrainingTargetError,
)
from network.features import (
    HISTORY_POSITIONS,
    encode_batch,
    encode_batch_gpu,
    encode_packed_batch,
    encode_state,
)
from network.loss import AlphaZeroLoss, LossOutput, LossWeights, TrainingTargets
from network.model import NetworkOutput, PolicyValueNetwork, ResidualBlock
from network.policy import (
    mask_policy_logits,
    masked_policy_log_probabilities,
    masked_policy_probabilities,
)
from network.specification import (
    MODEL_SPEC_SCHEMA_VERSION,
    SUPPORTED_FEATURE_SCHEMA_ID,
    NetworkSpecification,
)

__all__ = [
    "HISTORY_POSITIONS",
    "MODEL_SPEC_SCHEMA_VERSION",
    "SUPPORTED_FEATURE_SCHEMA_ID",
    "AlphaZeroLoss",
    "FeatureEncodingError",
    "LossOutput",
    "LossWeights",
    "NetworkError",
    "NetworkInputError",
    "NetworkOutput",
    "NetworkSpecification",
    "NetworkSpecificationError",
    "PolicyMaskError",
    "PolicyValueNetwork",
    "ResidualBlock",
    "TrainingTargetError",
    "TrainingTargets",
    "encode_batch",
    "encode_batch_gpu",
    "encode_packed_batch",
    "encode_state",
    "mask_policy_logits",
    "masked_policy_log_probabilities",
    "masked_policy_probabilities",
]

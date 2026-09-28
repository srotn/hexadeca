"""Immutable, serializable policy-value network specification."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar

from config.schema import NetworkConfig, RulesConfig
from network.errors import NetworkSpecificationError

MODEL_SPEC_SCHEMA_VERSION = 1
# The public release contains one versioned 16-plane encoder. Checkpoints are
# kept separate by their ruleset and feature-schema identifiers.
SUPPORTED_FEATURE_SCHEMA_ID = "hexadeca-v1-16p"
SUPPORTED_FEATURE_SCHEMA_IDS = frozenset({SUPPORTED_FEATURE_SCHEMA_ID})
SUPPORTED_INPUT_PLANES = 16

_SpecValue = int | float | str


@dataclass(frozen=True, slots=True)
class NetworkSpecification:
    """All metadata required to reconstruct and validate one model family."""

    FIELD_NAMES: ClassVar[tuple[str, ...]]

    schema_version: int
    ruleset_id: str
    feature_schema_id: str
    board_size: int
    input_planes: int
    residual_blocks: int
    channels: int
    policy_head_channels: int
    value_head_channels: int
    value_hidden_features: int
    policy_size: int
    score_normalizer: float
    batch_norm_epsilon: float
    batch_norm_momentum: float

    @classmethod
    def from_config(
        cls, rules: RulesConfig, network: NetworkConfig
    ) -> NetworkSpecification:
        """Construct a validated specification from the unified config."""

        specification = cls(
            schema_version=MODEL_SPEC_SCHEMA_VERSION,
            ruleset_id=rules.ruleset_id,
            feature_schema_id=network.feature_schema_id,
            board_size=rules.board_size,
            input_planes=network.input_planes,
            residual_blocks=network.residual_blocks,
            channels=network.channels,
            policy_head_channels=network.policy_head_channels,
            value_head_channels=network.value_head_channels,
            value_hidden_features=network.value_hidden_features,
            policy_size=network.policy_size,
            score_normalizer=network.score_normalizer,
            batch_norm_epsilon=network.batch_norm_epsilon,
            batch_norm_momentum=network.batch_norm_momentum,
        )
        specification.validate()
        return specification

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> NetworkSpecification:
        """Parse exact checkpoint metadata without accepting unknown fields."""

        missing = sorted(set(cls.FIELD_NAMES) - set(raw))
        unexpected = sorted(set(raw) - set(cls.FIELD_NAMES))
        if missing or unexpected:
            details: list[str] = []
            if missing:
                details.append(f"missing: {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected: {', '.join(unexpected)}")
            raise NetworkSpecificationError(
                f"Invalid network specification fields ({'; '.join(details)})"
            )

        specification = cls(
            schema_version=_exact_int(raw, "schema_version"),
            ruleset_id=_exact_string(raw, "ruleset_id"),
            feature_schema_id=_exact_string(raw, "feature_schema_id"),
            board_size=_exact_int(raw, "board_size"),
            input_planes=_exact_int(raw, "input_planes"),
            residual_blocks=_exact_int(raw, "residual_blocks"),
            channels=_exact_int(raw, "channels"),
            policy_head_channels=_exact_int(raw, "policy_head_channels"),
            value_head_channels=_exact_int(raw, "value_head_channels"),
            value_hidden_features=_exact_int(raw, "value_hidden_features"),
            policy_size=_exact_int(raw, "policy_size"),
            score_normalizer=_exact_float(raw, "score_normalizer"),
            batch_norm_epsilon=_exact_float(raw, "batch_norm_epsilon"),
            batch_norm_momentum=_exact_float(raw, "batch_norm_momentum"),
        )
        specification.validate()
        return specification

    def to_dict(self) -> dict[str, _SpecValue]:
        """Return JSON-safe metadata for future checkpoint persistence."""

        return {
            "schema_version": self.schema_version,
            "ruleset_id": self.ruleset_id,
            "feature_schema_id": self.feature_schema_id,
            "board_size": self.board_size,
            "input_planes": self.input_planes,
            "residual_blocks": self.residual_blocks,
            "channels": self.channels,
            "policy_head_channels": self.policy_head_channels,
            "value_head_channels": self.value_head_channels,
            "value_hidden_features": self.value_hidden_features,
            "policy_size": self.policy_size,
            "score_normalizer": self.score_normalizer,
            "batch_norm_epsilon": self.batch_norm_epsilon,
            "batch_norm_momentum": self.batch_norm_momentum,
        }

    def validate(self) -> None:
        """Reject specifications unsupported by the Stage 6 implementation."""

        if self.schema_version != MODEL_SPEC_SCHEMA_VERSION:
            raise NetworkSpecificationError(
                f"Unsupported model specification version: {self.schema_version}"
            )
        if not self.ruleset_id:
            raise NetworkSpecificationError("ruleset_id must not be empty")
        if self.feature_schema_id not in SUPPORTED_FEATURE_SCHEMA_IDS:
            raise NetworkSpecificationError(
                f"Unsupported feature schema: {self.feature_schema_id}"
            )
        if self.input_planes != SUPPORTED_INPUT_PLANES:
            raise NetworkSpecificationError(
                f"{self.feature_schema_id} requires {SUPPORTED_INPUT_PLANES} planes"
            )
        positive_integers = {
            "board_size": self.board_size,
            "residual_blocks": self.residual_blocks,
            "channels": self.channels,
            "policy_head_channels": self.policy_head_channels,
            "value_head_channels": self.value_head_channels,
            "value_hidden_features": self.value_hidden_features,
            "policy_size": self.policy_size,
        }
        invalid = [name for name, value in positive_integers.items() if value <= 0]
        if invalid:
            raise NetworkSpecificationError(
                f"Network dimensions must be positive: {', '.join(invalid)}"
            )
        if self.board_size * self.board_size != self.policy_size:
            raise NetworkSpecificationError("policy_size must equal board_size squared")
        if self.score_normalizer <= 0:
            raise NetworkSpecificationError("score_normalizer must be positive")
        if self.batch_norm_epsilon <= 0:
            raise NetworkSpecificationError("batch_norm_epsilon must be positive")
        if not 0 < self.batch_norm_momentum <= 1:
            raise NetworkSpecificationError(
                "batch_norm_momentum must be greater than zero and at most one"
            )

    def ensure_compatible(self, other: NetworkSpecification) -> None:
        """Raise with exact field names if checkpoint metadata differs."""

        mismatches = [
            field.name
            for field in fields(self)
            if getattr(self, field.name) != getattr(other, field.name)
        ]
        if mismatches:
            raise NetworkSpecificationError(
                f"Incompatible network specification fields: {', '.join(mismatches)}"
            )


NetworkSpecification.FIELD_NAMES = tuple(
    field.name for field in fields(NetworkSpecification)
)


def _exact_int(raw: dict[str, object], key: str) -> int:
    value = raw[key]
    if type(value) is not int:
        raise NetworkSpecificationError(f"{key} must be an integer")
    return value


def _exact_float(raw: dict[str, object], key: str) -> float:
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NetworkSpecificationError(f"{key} must be numeric")
    return float(value)


def _exact_string(raw: dict[str, object], key: str) -> str:
    value = raw[key]
    if not isinstance(value, str) or not value:
        raise NetworkSpecificationError(f"{key} must be a non-empty string")
    return value

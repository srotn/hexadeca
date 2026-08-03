"""Domain exceptions for feature encoding and policy-value networks."""


class NetworkError(Exception):
    """Base class for Stage 6 network failures."""


class NetworkSpecificationError(NetworkError, ValueError):
    """Raised when model metadata is invalid or incompatible."""


class FeatureEncodingError(NetworkError, ValueError):
    """Raised when a game snapshot cannot satisfy the feature schema."""


class NetworkInputError(NetworkError, ValueError):
    """Raised when a tensor does not match the configured model input."""


class PolicyMaskError(NetworkError, ValueError):
    """Raised when policy logits and legal actions are incompatible."""


class TrainingTargetError(NetworkError, ValueError):
    """Raised when a supervised training target violates its contract."""

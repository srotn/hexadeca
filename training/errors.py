"""Typed failures raised by training data and checkpoint infrastructure."""


class TrainingInfrastructureError(RuntimeError):
    """Base error for Stage 8 storage and loading boundaries."""


class ReplayError(TrainingInfrastructureError):
    """Base error for replay-buffer operations."""


class ReplayValidationError(ReplayError, ValueError):
    """Raised when a replay sample violates the versioned data contract."""


class ReplayPersistenceError(ReplayError):
    """Raised when a replay snapshot cannot be safely stored or restored."""


class ReplayDataLoaderError(ReplayError, ValueError):
    """Raised when a replay dataset cannot produce valid training batches."""


class SelfPlayError(TrainingInfrastructureError):
    """Base error for self-play generation and orchestration."""


class SelfPlayValidationError(SelfPlayError, ValueError):
    """Raised when a self-play request or result violates its contract."""


class SelfPlayInferenceError(SelfPlayError):
    """Raised when centralized inference cannot serve a worker request."""


class SelfPlayWorkerError(SelfPlayError):
    """Raised when a worker fails, exits early, or cannot shut down cleanly."""


class CheckpointError(TrainingInfrastructureError):
    """Base error for immutable checkpoint bundle operations."""


class CheckpointCompatibilityError(CheckpointError, ValueError):
    """Raised before loading checkpoint state with incompatible metadata."""

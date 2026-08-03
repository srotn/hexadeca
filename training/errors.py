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


class TrainerError(TrainingInfrastructureError):
    """Base error for optimizer-driven model training."""


class TrainerConfigurationError(TrainerError, ValueError):
    """Raised when a Trainer cannot honor its runtime configuration."""


class TrainerStateError(TrainerError):
    """Raised when an iteration or resume request violates Trainer state."""


class TrainerStepError(TrainerError):
    """Raised when a model update cannot complete safely."""


class ArenaError(TrainingInfrastructureError):
    """Base error for candidate-versus-best Arena evaluation."""


class ArenaValidationError(ArenaError, ValueError):
    """Raised when an Arena request or completed game violates its contract."""


class ArenaInferenceError(ArenaError):
    """Raised when centralized dual-model inference cannot serve a worker."""


class ArenaWorkerError(ArenaError):
    """Raised when an Arena worker fails or cannot shut down cleanly."""


class EvaluationError(TrainingInfrastructureError):
    """Base error for rating, reporting, and checkpoint promotion."""


class EvaluationStatisticsError(EvaluationError, ValueError):
    """Raised when Arena outcomes cannot produce valid rating statistics."""


class EvaluationReportError(EvaluationError):
    """Raised when a versioned evaluation report cannot be published."""


class EvaluationGateError(EvaluationError):
    """Raised when checkpoint comparison or promotion cannot complete safely."""

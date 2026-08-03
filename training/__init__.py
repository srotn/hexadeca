"""Replay, data-loading, and checkpoint infrastructure for AlphaZero training."""

from training.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointManager,
    CheckpointMetadata,
    MetricValue,
    Stateful,
)
from training.data import ReplayBatch, ReplayCollator, build_replay_data_loader
from training.errors import (
    CheckpointCompatibilityError,
    CheckpointError,
    ReplayDataLoaderError,
    ReplayError,
    ReplayPersistenceError,
    ReplayValidationError,
    SelfPlayError,
    SelfPlayInferenceError,
    SelfPlayValidationError,
    SelfPlayWorkerError,
    TrainingInfrastructureError,
)
from training.replay import (
    REPLAY_SAMPLE_SCHEMA_VERSION,
    ReplayBuffer,
    ReplaySample,
    ReplaySnapshot,
)
from training.replay_store import (
    REPLAY_STORE_SCHEMA_VERSION,
    ReplayLoadResult,
    SqliteReplayStore,
)
from training.self_play import (
    SELF_PLAY_SCHEMA_VERSION,
    ProgressCallback,
    SelfPlayBatch,
    SelfPlayCoordinator,
    SelfPlayGame,
    SelfPlayProgress,
    derive_game_seed,
    play_self_play_game,
)

__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "REPLAY_SAMPLE_SCHEMA_VERSION",
    "REPLAY_STORE_SCHEMA_VERSION",
    "SELF_PLAY_SCHEMA_VERSION",
    "CheckpointCompatibilityError",
    "CheckpointError",
    "CheckpointManager",
    "CheckpointMetadata",
    "MetricValue",
    "ProgressCallback",
    "ReplayBatch",
    "ReplayBuffer",
    "ReplayCollator",
    "ReplayDataLoaderError",
    "ReplayError",
    "ReplayLoadResult",
    "ReplayPersistenceError",
    "ReplaySample",
    "ReplaySnapshot",
    "ReplayValidationError",
    "SelfPlayBatch",
    "SelfPlayCoordinator",
    "SelfPlayError",
    "SelfPlayGame",
    "SelfPlayInferenceError",
    "SelfPlayProgress",
    "SelfPlayValidationError",
    "SelfPlayWorkerError",
    "SqliteReplayStore",
    "Stateful",
    "TrainingInfrastructureError",
    "build_replay_data_loader",
    "derive_game_seed",
    "play_self_play_game",
]

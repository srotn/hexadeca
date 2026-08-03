"""Typed configuration schema and validation rules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any, TypeVar

_T = TypeVar("_T")
CONFIG_SCHEMA_VERSION = 6


class SchemaError(ValueError):
    """Raised when a parsed configuration does not meet the schema."""


@dataclass(frozen=True)
class ProjectConfig:
    """Repository-level metadata for a run configuration."""

    name: str
    config_schema_version: int


@dataclass(frozen=True)
class RulesConfig:
    """Immutable properties of the active Hexadeca ruleset."""

    ruleset_id: str
    board_size: int
    neighborhood_radius: int
    action_encoding: str
    starting_player: str
    distance_metric: str
    score_occupied_cells: bool
    tie_break: str
    majority_award: str
    zobrist_seed: int

    @property
    def action_size(self) -> int:
        """Return the square-board action count implied by this ruleset."""

        return self.board_size * self.board_size


@dataclass(frozen=True)
class NetworkConfig:
    """Static dimensions of the policy-value model family."""

    feature_schema_id: str
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


@dataclass(frozen=True)
class MctsConfig:
    """Confirmed neural-MCTS search parameters."""

    training_simulations: int
    evaluation_simulations: int
    c_puct: float
    root_noise_enabled: bool
    root_noise_only: bool
    dirichlet_alpha: float
    dirichlet_epsilon: float
    virtual_loss: int
    fpu_reduction: float
    max_inference_batch_size: int
    stochastic_plies: int
    opening_temperature: float
    endgame_temperature: float


@dataclass(frozen=True)
class ReplayConfig:
    """Replay-buffer capacity settings."""

    capacity_positions: int
    persistence_enabled: bool
    persistence_file_name: str
    persistence_chunk_size: int


@dataclass(frozen=True)
class SelfPlayConfig:
    """Multiprocess self-play and centralized inference controls."""

    games_per_iteration: int
    worker_processes: int
    worker_torch_threads: int
    inference_max_batch_size: int
    inference_batch_wait_seconds: float
    inference_response_timeout_seconds: float
    worker_shutdown_timeout_seconds: float
    random_seed: int


@dataclass(frozen=True)
class TrainingConfig:
    """Confirmed optimizer and batch settings."""

    optimizer: str
    learning_rate: float
    weight_decay: float
    adamw_beta1: float
    adamw_beta2: float
    adamw_epsilon: float
    adamw_amsgrad: bool
    gradient_clip_norm: float
    batch_size: int
    batches_per_iteration: int
    minimum_replay_positions: int
    data_loader_shuffle: bool
    data_loader_workers: int
    data_loader_prefetch_factor: int
    data_loader_pin_memory: bool
    data_loader_drop_last: bool
    amp_enabled: bool
    amp_dtype: str
    amp_initial_scale: float
    amp_growth_factor: float
    amp_backoff_factor: float
    amp_growth_interval: int
    amp_max_step_retries: int
    scheduler: str
    scheduler_warmup_steps: int
    scheduler_decay_steps: int
    scheduler_minimum_learning_rate: float
    checkpoint_interval_iterations: int
    tensorboard_enabled: bool
    tensorboard_log_interval_batches: int
    tensorboard_flush_seconds: int
    tensorboard_max_queue: int
    random_seed: int
    policy_loss_weight: float
    win_loss_weight: float
    black_score_loss_weight: float
    white_score_loss_weight: float


@dataclass(frozen=True)
class BenchmarkConfig:
    """Reproducible benchmark settings for implemented project stages."""

    board_warmup_iterations: int
    board_measurement_iterations: int
    board_position_plies: int
    scoring_warmup_iterations: int
    scoring_measurement_iterations: int
    environment_warmup_iterations: int
    environment_measurement_iterations: int
    network_warmup_iterations: int
    network_measurement_iterations: int
    network_cpu_batch_size: int
    network_gpu_batch_sizes: tuple[int, ...]
    mcts_warmup_iterations: int
    mcts_measurement_iterations: int
    mcts_benchmark_simulations: int
    replay_warmup_iterations: int
    replay_measurement_iterations: int
    replay_benchmark_positions: int
    replay_benchmark_batch_size: int
    self_play_benchmark_games: int
    self_play_benchmark_simulations: int
    trainer_benchmark_warmup_batches: int
    trainer_benchmark_measurement_batches: int
    random_seed: int


@dataclass(frozen=True)
class PathsConfig:
    """Repository-relative persistent artifact locations."""

    run_directory: Path
    checkpoint_directory: Path
    log_directory: Path
    benchmark_directory: Path


@dataclass(frozen=True)
class LoggingConfig:
    """Logging destination and formatting controls."""

    level: str
    console_enabled: bool
    file_enabled: bool
    file_name: str


@dataclass(frozen=True)
class AppConfig:
    """Fully validated configuration consumed by application components."""

    project: ProjectConfig
    rules: RulesConfig
    network: NetworkConfig
    mcts: MctsConfig
    replay: ReplayConfig
    self_play: SelfPlayConfig
    training: TrainingConfig
    benchmark: BenchmarkConfig
    paths: PathsConfig
    logging: LoggingConfig


def build_config(raw: Mapping[str, Any]) -> AppConfig:
    """Validate parsed TOML data and construct an immutable configuration."""

    _reject_unknown(
        raw,
        {
            "project",
            "rules",
            "network",
            "mcts",
            "replay",
            "self_play",
            "training",
            "benchmark",
            "paths",
            "logging",
        },
        "root",
    )
    project = _build_project(_section(raw, "project"))
    rules = _build_rules(_section(raw, "rules"))
    network = _build_network(_section(raw, "network"))
    mcts = _build_mcts(_section(raw, "mcts"))
    replay = _build_replay(_section(raw, "replay"))
    self_play = _build_self_play(_section(raw, "self_play"))
    training = _build_training(_section(raw, "training"))
    benchmark = _build_benchmark(_section(raw, "benchmark"))
    paths = _build_paths(_section(raw, "paths"))
    logging = _build_logging(_section(raw, "logging"))

    if network.policy_size != rules.action_size:
        raise SchemaError(
            "network.policy_size must equal board_size squared for the active ruleset"
        )
    if not any(
        weight > 0
        for weight in (
            training.policy_loss_weight,
            training.win_loss_weight,
            training.black_score_loss_weight,
            training.white_score_loss_weight,
        )
    ):
        raise SchemaError("At least one training loss weight must be positive")
    if not mcts.root_noise_only:
        raise SchemaError("Only root-scoped MCTS noise is supported")
    if training.batch_size > replay.capacity_positions:
        raise SchemaError("training.batch_size must not exceed replay capacity")
    if training.minimum_replay_positions < training.batch_size:
        raise SchemaError(
            "training.minimum_replay_positions must be at least training.batch_size"
        )
    if training.minimum_replay_positions > replay.capacity_positions:
        raise SchemaError(
            "training.minimum_replay_positions must not exceed replay capacity"
        )
    if training.scheduler_minimum_learning_rate > training.learning_rate:
        raise SchemaError(
            "training.scheduler_minimum_learning_rate must not exceed learning_rate"
        )
    if training.adamw_beta1 >= 1.0 or training.adamw_beta2 >= 1.0:
        raise SchemaError("training AdamW beta values must be less than 1.0")
    if training.amp_growth_factor <= 1.0:
        raise SchemaError("training.amp_growth_factor must be greater than 1.0")
    if training.amp_backoff_factor >= 1.0:
        raise SchemaError("training.amp_backoff_factor must be less than 1.0")
    if benchmark.replay_benchmark_batch_size > benchmark.replay_benchmark_positions:
        raise SchemaError(
            "benchmark.replay_benchmark_batch_size must not exceed benchmark positions"
        )
    if self_play.inference_max_batch_size < mcts.max_inference_batch_size:
        raise SchemaError(
            "self_play.inference_max_batch_size must be at least the MCTS batch size"
        )
    return AppConfig(
        project=project,
        rules=rules,
        network=network,
        mcts=mcts,
        replay=replay,
        self_play=self_play,
        training=training,
        benchmark=benchmark,
        paths=paths,
        logging=logging,
    )


def _build_project(raw: Mapping[str, Any]) -> ProjectConfig:
    _reject_unknown(raw, {"name", "config_schema_version"}, "project")
    project = ProjectConfig(
        name=_string(raw, "name", "project"),
        config_schema_version=_positive_int(raw, "config_schema_version", "project"),
    )
    if project.config_schema_version != CONFIG_SCHEMA_VERSION:
        raise SchemaError(
            "Unsupported project.config_schema_version: "
            f"{project.config_schema_version}"
        )
    return project


def _build_rules(raw: Mapping[str, Any]) -> RulesConfig:
    _reject_unknown(
        raw,
        {
            "ruleset_id",
            "board_size",
            "neighborhood_radius",
            "action_encoding",
            "starting_player",
            "distance_metric",
            "score_occupied_cells",
            "tie_break",
            "majority_award",
            "zobrist_seed",
        },
        "rules",
    )
    rules = RulesConfig(
        ruleset_id=_string(raw, "ruleset_id", "rules"),
        board_size=_positive_int(raw, "board_size", "rules"),
        neighborhood_radius=_positive_int(raw, "neighborhood_radius", "rules"),
        action_encoding=_choice(raw, "action_encoding", "rules", {"row-major"}),
        starting_player=_choice(raw, "starting_player", "rules", {"black"}),
        distance_metric=_choice(raw, "distance_metric", "rules", {"squared-euclidean"}),
        score_occupied_cells=_bool(raw, "score_occupied_cells", "rules"),
        tie_break=_choice(raw, "tie_break", "rules", {"next-distance-layer"}),
        majority_award=_choice(raw, "majority_award", "rules", {"majority-count"}),
        zobrist_seed=_nonnegative_int(raw, "zobrist_seed", "rules"),
    )
    return rules


def _build_network(raw: Mapping[str, Any]) -> NetworkConfig:
    _reject_unknown(
        raw,
        {
            "feature_schema_id",
            "input_planes",
            "residual_blocks",
            "channels",
            "policy_head_channels",
            "value_head_channels",
            "value_hidden_features",
            "policy_size",
            "score_normalizer",
            "batch_norm_epsilon",
            "batch_norm_momentum",
        },
        "network",
    )
    network = NetworkConfig(
        feature_schema_id=_string(raw, "feature_schema_id", "network"),
        input_planes=_positive_int(raw, "input_planes", "network"),
        residual_blocks=_positive_int(raw, "residual_blocks", "network"),
        channels=_positive_int(raw, "channels", "network"),
        policy_head_channels=_positive_int(raw, "policy_head_channels", "network"),
        value_head_channels=_positive_int(raw, "value_head_channels", "network"),
        value_hidden_features=_positive_int(raw, "value_hidden_features", "network"),
        policy_size=_positive_int(raw, "policy_size", "network"),
        score_normalizer=_positive_float(raw, "score_normalizer", "network"),
        batch_norm_epsilon=_positive_float(raw, "batch_norm_epsilon", "network"),
        batch_norm_momentum=_bounded_float(
            raw,
            "batch_norm_momentum",
            "network",
            lower_bound=0.0,
            upper_bound=1.0,
        ),
    )
    return network


def _build_mcts(raw: Mapping[str, Any]) -> MctsConfig:
    _reject_unknown(
        raw,
        {
            "training_simulations",
            "evaluation_simulations",
            "c_puct",
            "root_noise_enabled",
            "root_noise_only",
            "dirichlet_alpha",
            "dirichlet_epsilon",
            "virtual_loss",
            "fpu_reduction",
            "max_inference_batch_size",
            "stochastic_plies",
            "opening_temperature",
            "endgame_temperature",
        },
        "mcts",
    )
    return MctsConfig(
        training_simulations=_positive_int(raw, "training_simulations", "mcts"),
        evaluation_simulations=_positive_int(raw, "evaluation_simulations", "mcts"),
        c_puct=_positive_float(raw, "c_puct", "mcts"),
        root_noise_enabled=_bool(raw, "root_noise_enabled", "mcts"),
        root_noise_only=_bool(raw, "root_noise_only", "mcts"),
        dirichlet_alpha=_positive_float(raw, "dirichlet_alpha", "mcts"),
        dirichlet_epsilon=_bounded_float(
            raw,
            "dirichlet_epsilon",
            "mcts",
            lower_bound=0.0,
            upper_bound=1.0,
        ),
        virtual_loss=_positive_int(raw, "virtual_loss", "mcts"),
        fpu_reduction=_nonnegative_float(raw, "fpu_reduction", "mcts"),
        max_inference_batch_size=_positive_int(raw, "max_inference_batch_size", "mcts"),
        stochastic_plies=_nonnegative_int(raw, "stochastic_plies", "mcts"),
        opening_temperature=_positive_float(raw, "opening_temperature", "mcts"),
        endgame_temperature=_nonnegative_float(raw, "endgame_temperature", "mcts"),
    )


def _build_replay(raw: Mapping[str, Any]) -> ReplayConfig:
    _reject_unknown(
        raw,
        {
            "capacity_positions",
            "persistence_enabled",
            "persistence_file_name",
            "persistence_chunk_size",
        },
        "replay",
    )
    persistence_file_name = _string(raw, "persistence_file_name", "replay")
    if Path(
        persistence_file_name
    ).name != persistence_file_name or not persistence_file_name.endswith(".sqlite3"):
        raise SchemaError("replay.persistence_file_name must be a .sqlite3 file name")
    return ReplayConfig(
        capacity_positions=_positive_int(raw, "capacity_positions", "replay"),
        persistence_enabled=_bool(raw, "persistence_enabled", "replay"),
        persistence_file_name=persistence_file_name,
        persistence_chunk_size=_positive_int(raw, "persistence_chunk_size", "replay"),
    )


def _build_self_play(raw: Mapping[str, Any]) -> SelfPlayConfig:
    _reject_unknown(
        raw,
        {
            "games_per_iteration",
            "worker_processes",
            "worker_torch_threads",
            "inference_max_batch_size",
            "inference_batch_wait_seconds",
            "inference_response_timeout_seconds",
            "worker_shutdown_timeout_seconds",
            "random_seed",
        },
        "self_play",
    )
    return SelfPlayConfig(
        games_per_iteration=_positive_int(raw, "games_per_iteration", "self_play"),
        worker_processes=_positive_int(raw, "worker_processes", "self_play"),
        worker_torch_threads=_positive_int(raw, "worker_torch_threads", "self_play"),
        inference_max_batch_size=_positive_int(
            raw, "inference_max_batch_size", "self_play"
        ),
        inference_batch_wait_seconds=_positive_float(
            raw, "inference_batch_wait_seconds", "self_play"
        ),
        inference_response_timeout_seconds=_positive_float(
            raw, "inference_response_timeout_seconds", "self_play"
        ),
        worker_shutdown_timeout_seconds=_positive_float(
            raw, "worker_shutdown_timeout_seconds", "self_play"
        ),
        random_seed=_nonnegative_int(raw, "random_seed", "self_play"),
    )


def _build_training(raw: Mapping[str, Any]) -> TrainingConfig:
    _reject_unknown(
        raw,
        {
            "optimizer",
            "learning_rate",
            "weight_decay",
            "adamw_beta1",
            "adamw_beta2",
            "adamw_epsilon",
            "adamw_amsgrad",
            "gradient_clip_norm",
            "batch_size",
            "batches_per_iteration",
            "minimum_replay_positions",
            "data_loader_shuffle",
            "data_loader_workers",
            "data_loader_prefetch_factor",
            "data_loader_pin_memory",
            "data_loader_drop_last",
            "amp_enabled",
            "amp_dtype",
            "amp_initial_scale",
            "amp_growth_factor",
            "amp_backoff_factor",
            "amp_growth_interval",
            "amp_max_step_retries",
            "scheduler",
            "scheduler_warmup_steps",
            "scheduler_decay_steps",
            "scheduler_minimum_learning_rate",
            "checkpoint_interval_iterations",
            "tensorboard_enabled",
            "tensorboard_log_interval_batches",
            "tensorboard_flush_seconds",
            "tensorboard_max_queue",
            "random_seed",
            "policy_loss_weight",
            "win_loss_weight",
            "black_score_loss_weight",
            "white_score_loss_weight",
        },
        "training",
    )
    return TrainingConfig(
        optimizer=_choice(raw, "optimizer", "training", {"adamw"}),
        learning_rate=_positive_float(raw, "learning_rate", "training"),
        weight_decay=_nonnegative_float(raw, "weight_decay", "training"),
        adamw_beta1=_nonnegative_float(raw, "adamw_beta1", "training"),
        adamw_beta2=_nonnegative_float(raw, "adamw_beta2", "training"),
        adamw_epsilon=_positive_float(raw, "adamw_epsilon", "training"),
        adamw_amsgrad=_bool(raw, "adamw_amsgrad", "training"),
        gradient_clip_norm=_positive_float(raw, "gradient_clip_norm", "training"),
        batch_size=_positive_int(raw, "batch_size", "training"),
        batches_per_iteration=_positive_int(raw, "batches_per_iteration", "training"),
        minimum_replay_positions=_positive_int(
            raw, "minimum_replay_positions", "training"
        ),
        data_loader_shuffle=_bool(raw, "data_loader_shuffle", "training"),
        data_loader_workers=_nonnegative_int(raw, "data_loader_workers", "training"),
        data_loader_prefetch_factor=_positive_int(
            raw, "data_loader_prefetch_factor", "training"
        ),
        data_loader_pin_memory=_bool(raw, "data_loader_pin_memory", "training"),
        data_loader_drop_last=_bool(raw, "data_loader_drop_last", "training"),
        amp_enabled=_bool(raw, "amp_enabled", "training"),
        amp_dtype=_choice(raw, "amp_dtype", "training", {"float16"}),
        amp_initial_scale=_positive_float(raw, "amp_initial_scale", "training"),
        amp_growth_factor=_positive_float(raw, "amp_growth_factor", "training"),
        amp_backoff_factor=_positive_float(raw, "amp_backoff_factor", "training"),
        amp_growth_interval=_positive_int(raw, "amp_growth_interval", "training"),
        amp_max_step_retries=_nonnegative_int(raw, "amp_max_step_retries", "training"),
        scheduler=_choice(raw, "scheduler", "training", {"warmup-cosine"}),
        scheduler_warmup_steps=_positive_int(raw, "scheduler_warmup_steps", "training"),
        scheduler_decay_steps=_positive_int(raw, "scheduler_decay_steps", "training"),
        scheduler_minimum_learning_rate=_positive_float(
            raw, "scheduler_minimum_learning_rate", "training"
        ),
        checkpoint_interval_iterations=_nonnegative_int(
            raw, "checkpoint_interval_iterations", "training"
        ),
        tensorboard_enabled=_bool(raw, "tensorboard_enabled", "training"),
        tensorboard_log_interval_batches=_positive_int(
            raw, "tensorboard_log_interval_batches", "training"
        ),
        tensorboard_flush_seconds=_positive_int(
            raw, "tensorboard_flush_seconds", "training"
        ),
        tensorboard_max_queue=_positive_int(raw, "tensorboard_max_queue", "training"),
        random_seed=_nonnegative_int(raw, "random_seed", "training"),
        policy_loss_weight=_nonnegative_float(raw, "policy_loss_weight", "training"),
        win_loss_weight=_nonnegative_float(raw, "win_loss_weight", "training"),
        black_score_loss_weight=_nonnegative_float(
            raw, "black_score_loss_weight", "training"
        ),
        white_score_loss_weight=_nonnegative_float(
            raw, "white_score_loss_weight", "training"
        ),
    )


def _build_benchmark(raw: Mapping[str, Any]) -> BenchmarkConfig:
    _reject_unknown(
        raw,
        {
            "board_warmup_iterations",
            "board_measurement_iterations",
            "board_position_plies",
            "scoring_warmup_iterations",
            "scoring_measurement_iterations",
            "environment_warmup_iterations",
            "environment_measurement_iterations",
            "network_warmup_iterations",
            "network_measurement_iterations",
            "network_cpu_batch_size",
            "network_gpu_batch_sizes",
            "mcts_warmup_iterations",
            "mcts_measurement_iterations",
            "mcts_benchmark_simulations",
            "replay_warmup_iterations",
            "replay_measurement_iterations",
            "replay_benchmark_positions",
            "replay_benchmark_batch_size",
            "self_play_benchmark_games",
            "self_play_benchmark_simulations",
            "trainer_benchmark_warmup_batches",
            "trainer_benchmark_measurement_batches",
            "random_seed",
        },
        "benchmark",
    )
    return BenchmarkConfig(
        board_warmup_iterations=_nonnegative_int(
            raw, "board_warmup_iterations", "benchmark"
        ),
        board_measurement_iterations=_positive_int(
            raw, "board_measurement_iterations", "benchmark"
        ),
        board_position_plies=_nonnegative_int(raw, "board_position_plies", "benchmark"),
        scoring_warmup_iterations=_nonnegative_int(
            raw, "scoring_warmup_iterations", "benchmark"
        ),
        scoring_measurement_iterations=_positive_int(
            raw, "scoring_measurement_iterations", "benchmark"
        ),
        environment_warmup_iterations=_nonnegative_int(
            raw, "environment_warmup_iterations", "benchmark"
        ),
        environment_measurement_iterations=_positive_int(
            raw, "environment_measurement_iterations", "benchmark"
        ),
        network_warmup_iterations=_nonnegative_int(
            raw, "network_warmup_iterations", "benchmark"
        ),
        network_measurement_iterations=_positive_int(
            raw, "network_measurement_iterations", "benchmark"
        ),
        network_cpu_batch_size=_positive_int(
            raw, "network_cpu_batch_size", "benchmark"
        ),
        network_gpu_batch_sizes=_positive_int_tuple(
            raw, "network_gpu_batch_sizes", "benchmark"
        ),
        mcts_warmup_iterations=_nonnegative_int(
            raw, "mcts_warmup_iterations", "benchmark"
        ),
        mcts_measurement_iterations=_positive_int(
            raw, "mcts_measurement_iterations", "benchmark"
        ),
        mcts_benchmark_simulations=_positive_int(
            raw, "mcts_benchmark_simulations", "benchmark"
        ),
        replay_warmup_iterations=_nonnegative_int(
            raw, "replay_warmup_iterations", "benchmark"
        ),
        replay_measurement_iterations=_positive_int(
            raw, "replay_measurement_iterations", "benchmark"
        ),
        replay_benchmark_positions=_positive_int(
            raw, "replay_benchmark_positions", "benchmark"
        ),
        replay_benchmark_batch_size=_positive_int(
            raw, "replay_benchmark_batch_size", "benchmark"
        ),
        self_play_benchmark_games=_positive_int(
            raw, "self_play_benchmark_games", "benchmark"
        ),
        self_play_benchmark_simulations=_positive_int(
            raw, "self_play_benchmark_simulations", "benchmark"
        ),
        trainer_benchmark_warmup_batches=_nonnegative_int(
            raw, "trainer_benchmark_warmup_batches", "benchmark"
        ),
        trainer_benchmark_measurement_batches=_positive_int(
            raw, "trainer_benchmark_measurement_batches", "benchmark"
        ),
        random_seed=_nonnegative_int(raw, "random_seed", "benchmark"),
    )


def _build_paths(raw: Mapping[str, Any]) -> PathsConfig:
    _reject_unknown(
        raw,
        {
            "run_directory",
            "checkpoint_directory",
            "log_directory",
            "benchmark_directory",
        },
        "paths",
    )
    return PathsConfig(
        run_directory=_relative_path(raw, "run_directory", "paths"),
        checkpoint_directory=_relative_path(raw, "checkpoint_directory", "paths"),
        log_directory=_relative_path(raw, "log_directory", "paths"),
        benchmark_directory=_relative_path(raw, "benchmark_directory", "paths"),
    )


def _build_logging(raw: Mapping[str, Any]) -> LoggingConfig:
    _reject_unknown(
        raw,
        {"level", "console_enabled", "file_enabled", "file_name"},
        "logging",
    )
    file_name = _string(raw, "file_name", "logging")
    if Path(file_name).name != file_name:
        raise SchemaError("logging.file_name must not include a directory")
    return LoggingConfig(
        level=_choice(
            raw,
            "level",
            "logging",
            {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"},
        ),
        console_enabled=_bool(raw, "console_enabled", "logging"),
        file_enabled=_bool(raw, "file_enabled", "logging"),
        file_name=file_name,
    )


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = _value(raw, name, "root")
    if not isinstance(value, Mapping):
        raise SchemaError(f"root.{name} must be a table")
    return value


def _value(raw: Mapping[str, Any], key: str, path: str) -> Any:
    try:
        return raw[key]
    except KeyError as error:
        raise SchemaError(
            f"Missing required configuration key: {path}.{key}"
        ) from error


def _string(raw: Mapping[str, Any], key: str, path: str) -> str:
    value = _value(raw, key, path)
    if not isinstance(value, str) or not value:
        raise SchemaError(f"{path}.{key} must be a non-empty string")
    return value


def _bool(raw: Mapping[str, Any], key: str, path: str) -> bool:
    value = _value(raw, key, path)
    if type(value) is not bool:
        raise SchemaError(f"{path}.{key} must be a boolean")
    return value


def _positive_int(raw: Mapping[str, Any], key: str, path: str) -> int:
    value = _integer(raw, key, path)
    if value <= 0:
        raise SchemaError(f"{path}.{key} must be greater than zero")
    return value


def _nonnegative_int(raw: Mapping[str, Any], key: str, path: str) -> int:
    value = _integer(raw, key, path)
    if value < 0:
        raise SchemaError(f"{path}.{key} must be zero or greater")
    return value


def _integer(raw: Mapping[str, Any], key: str, path: str) -> int:
    value = _value(raw, key, path)
    if type(value) is not int:
        raise SchemaError(f"{path}.{key} must be an integer")
    return value


def _positive_float(raw: Mapping[str, Any], key: str, path: str) -> float:
    value = _value(raw, key, path)
    if type(value) not in {int, float} or not isfinite(value) or value <= 0:
        raise SchemaError(f"{path}.{key} must be a positive number")
    return float(value)


def _nonnegative_float(raw: Mapping[str, Any], key: str, path: str) -> float:
    value = _value(raw, key, path)
    if type(value) not in {int, float} or not isfinite(value) or value < 0:
        raise SchemaError(f"{path}.{key} must be a nonnegative number")
    return float(value)


def _bounded_float(
    raw: Mapping[str, Any],
    key: str,
    path: str,
    *,
    lower_bound: float,
    upper_bound: float,
) -> float:
    value = _value(raw, key, path)
    if (
        type(value) not in {int, float}
        or not isfinite(value)
        or not lower_bound < value <= upper_bound
    ):
        raise SchemaError(
            f"{path}.{key} must be greater than {lower_bound} and at most {upper_bound}"
        )
    return float(value)


def _positive_int_tuple(raw: Mapping[str, Any], key: str, path: str) -> tuple[int, ...]:
    value = _value(raw, key, path)
    if not isinstance(value, list) or not value:
        raise SchemaError(f"{path}.{key} must be a non-empty array")
    if any(type(item) is not int or item <= 0 for item in value):
        raise SchemaError(f"{path}.{key} values must be positive integers")
    if len(set(value)) != len(value):
        raise SchemaError(f"{path}.{key} values must be unique")
    return tuple(value)


def _choice(raw: Mapping[str, Any], key: str, path: str, choices: set[str]) -> str:
    value = _string(raw, key, path)
    if value not in choices:
        allowed = ", ".join(sorted(choices))
        raise SchemaError(f"{path}.{key} must be one of: {allowed}")
    return value


def _relative_path(raw: Mapping[str, Any], key: str, path: str) -> Path:
    value = Path(_string(raw, key, path))
    if value.is_absolute() or ".." in value.parts:
        raise SchemaError(f"{path}.{key} must be a project-relative path")
    return value


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unexpected = sorted(set(raw) - allowed)
    if unexpected:
        values = ", ".join(unexpected)
        raise SchemaError(f"Unexpected configuration keys in {path}: {values}")

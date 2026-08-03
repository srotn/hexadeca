"""Typed configuration schema and validation rules."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

_T = TypeVar("_T")


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
    policy_size: int


@dataclass(frozen=True)
class MctsConfig:
    """Confirmed neural-MCTS search parameters."""

    training_simulations: int
    evaluation_simulations: int
    c_puct: float
    dirichlet_alpha: float
    stochastic_plies: int


@dataclass(frozen=True)
class ReplayConfig:
    """Replay-buffer capacity settings."""

    capacity_positions: int


@dataclass(frozen=True)
class TrainingConfig:
    """Confirmed optimizer and batch settings."""

    optimizer: str
    learning_rate: float
    batch_size: int


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
    training = _build_training(_section(raw, "training"))
    benchmark = _build_benchmark(_section(raw, "benchmark"))
    paths = _build_paths(_section(raw, "paths"))
    logging = _build_logging(_section(raw, "logging"))

    if network.policy_size != rules.action_size:
        raise SchemaError(
            "network.policy_size must equal board_size squared for the active ruleset"
        )
    return AppConfig(
        project=project,
        rules=rules,
        network=network,
        mcts=mcts,
        replay=replay,
        training=training,
        benchmark=benchmark,
        paths=paths,
        logging=logging,
    )


def _build_project(raw: Mapping[str, Any]) -> ProjectConfig:
    _reject_unknown(raw, {"name", "config_schema_version"}, "project")
    return ProjectConfig(
        name=_string(raw, "name", "project"),
        config_schema_version=_positive_int(raw, "config_schema_version", "project"),
    )


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
            "policy_size",
        },
        "network",
    )
    network = NetworkConfig(
        feature_schema_id=_string(raw, "feature_schema_id", "network"),
        input_planes=_positive_int(raw, "input_planes", "network"),
        residual_blocks=_positive_int(raw, "residual_blocks", "network"),
        channels=_positive_int(raw, "channels", "network"),
        policy_size=_positive_int(raw, "policy_size", "network"),
    )
    return network


def _build_mcts(raw: Mapping[str, Any]) -> MctsConfig:
    _reject_unknown(
        raw,
        {
            "training_simulations",
            "evaluation_simulations",
            "c_puct",
            "dirichlet_alpha",
            "stochastic_plies",
        },
        "mcts",
    )
    return MctsConfig(
        training_simulations=_positive_int(raw, "training_simulations", "mcts"),
        evaluation_simulations=_positive_int(raw, "evaluation_simulations", "mcts"),
        c_puct=_positive_float(raw, "c_puct", "mcts"),
        dirichlet_alpha=_positive_float(raw, "dirichlet_alpha", "mcts"),
        stochastic_plies=_nonnegative_int(raw, "stochastic_plies", "mcts"),
    )


def _build_replay(raw: Mapping[str, Any]) -> ReplayConfig:
    _reject_unknown(raw, {"capacity_positions"}, "replay")
    return ReplayConfig(
        capacity_positions=_positive_int(raw, "capacity_positions", "replay")
    )


def _build_training(raw: Mapping[str, Any]) -> TrainingConfig:
    _reject_unknown(raw, {"optimizer", "learning_rate", "batch_size"}, "training")
    return TrainingConfig(
        optimizer=_choice(raw, "optimizer", "training", {"adamw"}),
        learning_rate=_positive_float(raw, "learning_rate", "training"),
        batch_size=_positive_int(raw, "batch_size", "training"),
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
    if type(value) not in {int, float} or value <= 0:
        raise SchemaError(f"{path}.{key} must be a positive number")
    return float(value)


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

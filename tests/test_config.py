"""Tests for staged configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import ConfigError, load_config


def test_default_config_matches_confirmed_project_parameters() -> None:
    """The default TOML contains the confirmed rules and training settings."""

    config = load_config()

    assert config.project.config_schema_version == 14
    assert config.rules.board_size == 16
    assert config.rules.score_occupied_cells is True
    assert config.rules.majority_award == "one-point"
    assert config.rules.action_size == 256
    assert config.rules.zobrist_seed == 7_640_891_576_956_012_809
    assert config.network.input_planes == 16
    assert config.network.residual_blocks == 10
    assert config.network.channels == 128
    assert config.network.score_normalizer == 256.0
    assert config.network.policy_size == 256
    assert config.mcts.training_simulations == 1600
    assert config.mcts.engine == "native"
    assert config.mcts.evaluation_simulations == 1600
    assert config.mcts.dirichlet_alpha == 0.15
    assert config.mcts.dirichlet_epsilon == 0.25
    assert config.mcts.virtual_loss == 3
    assert config.mcts.root_noise_enabled is True
    assert config.mcts.root_noise_only is True
    assert config.mcts.fpu_reduction == 0.0
    assert config.mcts.max_inference_batch_size == 32
    assert config.mcts.opening_temperature == 1.0
    assert config.mcts.endgame_temperature == 0.0
    assert config.mcts.exact_endgame_enabled is True
    assert config.mcts.exact_endgame_max_legal_moves == 8
    assert config.replay.capacity_positions == 200_000
    assert config.replay.persistence_enabled is True
    assert config.replay.persistence_file_name == "replay.sqlite3"
    assert config.replay.persistence_chunk_size == 1_024
    assert config.self_play.games_per_iteration == 32
    assert config.self_play.worker_processes == 16
    assert config.self_play.worker_torch_threads == 1
    assert config.self_play.inference_transport == "object"
    assert config.self_play.inference_max_batch_size == 256
    assert config.self_play.inference_batch_wait_seconds == 0.005
    assert config.self_play.inference_response_timeout_seconds == 120.0
    assert config.self_play.worker_shutdown_timeout_seconds == 10.0
    assert config.self_play.random_seed == 20_260_803
    assert config.training.optimizer == "adamw"
    assert config.training.learning_rate == 0.0003
    assert config.training.weight_decay == 0.0001
    assert config.training.adamw_beta1 == 0.9
    assert config.training.adamw_beta2 == 0.999
    assert config.training.adamw_epsilon == 1e-8
    assert config.training.adamw_amsgrad is False
    assert config.training.gradient_clip_norm == 1.0
    assert config.training.batch_size == 256
    assert config.training.batches_per_iteration == 64
    assert config.training.minimum_replay_positions == 256
    assert config.training.data_loader_shuffle is True
    assert config.training.data_loader_workers == 0
    assert config.training.data_loader_prefetch_factor == 2
    assert config.training.data_loader_pin_memory is True
    assert config.training.data_loader_drop_last is True
    assert config.training.symmetry_augmentation == "d4"
    assert config.training.amp_enabled is True
    assert config.training.amp_dtype == "float16"
    assert config.training.amp_initial_scale == 65_536.0
    assert config.training.amp_growth_factor == 2.0
    assert config.training.amp_backoff_factor == 0.5
    assert config.training.amp_growth_interval == 2_000
    assert config.training.amp_max_step_retries == 8
    assert config.training.scheduler == "warmup-cosine"
    assert config.training.scheduler_warmup_steps == 1_000
    assert config.training.scheduler_decay_steps == 100_000
    assert config.training.scheduler_minimum_learning_rate == 0.00003
    assert config.training.scheduler_restart.enabled is False
    assert config.training.scheduler_restart.checkpoint_id == "none"
    assert config.training.scheduler_restart.initial_learning_rate == 0.00003
    assert config.training.scheduler_restart.peak_learning_rate == 0.0001
    assert config.training.scheduler_restart.warmup_steps == 1_000
    assert config.training.scheduler_restart.decay_steps == 30_000
    assert config.training.scheduler_restart.minimum_learning_rate == 0.00003
    assert config.training.checkpoint_interval_iterations == 1
    assert config.training.tensorboard_enabled is True
    assert config.training.tensorboard_log_interval_batches == 1
    assert config.training.tensorboard_flush_seconds == 30
    assert config.training.tensorboard_max_queue == 10
    assert config.training.random_seed == 20_260_803
    assert config.training.policy_loss_weight == 1.0
    assert config.evaluation.game_count == 100
    assert config.evaluation.opening_pair_count == 50
    assert config.evaluation.opening_plies == 2
    assert config.evaluation.paired_colors is True
    assert config.evaluation.promotion_score == 0.55
    assert config.evaluation.draw_score == 0.5
    assert config.evaluation.promotion_confidence_lower_bound == 0.5
    assert config.evaluation.confidence_level == 0.95
    assert config.evaluation.bootstrap_samples == 10_000
    assert config.evaluation.elo_scale == 400.0
    assert config.evaluation.elo_prior_points == 0.5
    assert config.evaluation.worker_processes == 4
    assert config.evaluation.inference_max_batch_size == 256
    assert config.evaluation.amp_enabled is True
    assert config.benchmark.board_measurement_iterations == 10_000
    assert config.benchmark.scoring_measurement_iterations == 1_000
    assert config.benchmark.environment_measurement_iterations == 1_000
    assert config.benchmark.network_gpu_batch_sizes == (1, 32, 256)
    assert config.benchmark.mcts_warmup_iterations == 2
    assert config.benchmark.mcts_measurement_iterations == 5
    assert config.benchmark.mcts_benchmark_simulations == 800
    assert config.benchmark.replay_benchmark_positions == 4_096
    assert config.benchmark.replay_benchmark_batch_size == 256
    assert config.benchmark.self_play_benchmark_games == 4
    assert config.benchmark.self_play_benchmark_simulations == 64
    assert config.benchmark.trainer_benchmark_warmup_batches == 3
    assert config.benchmark.trainer_benchmark_measurement_batches == 10
    assert config.benchmark.evaluation_benchmark_games == 4
    assert config.benchmark.evaluation_benchmark_simulations == 32
    assert config.benchmark.native_warmup_iterations == 5
    assert config.benchmark.native_measurement_iterations == 20
    assert config.benchmark.native_feature_batch_size == 96
    assert config.benchmark.native_mcts_simulations == 800
    assert config.benchmark.monitoring_warmup_events == 100
    assert config.benchmark.monitoring_measurement_events == 1_000
    assert config.benchmark.transport_warmup_iterations == 10
    assert config.benchmark.transport_measurement_iterations == 100
    assert config.benchmark.transport_batch_size == 32
    assert config.paths.evaluation_directory.as_posix() == "evaluation-results"
    assert config.monitoring.bind_host == "0.0.0.0"
    assert config.monitoring.bind_port == 5555
    assert config.monitoring.training_device == "auto"
    assert config.monitoring.auto_resume is True
    assert config.monitoring.event_buffer_capacity == 4_096
    assert config.monitoring.event_replay_limit == 2_048
    assert config.monitoring.status_refresh_interval_seconds == 5.0
    assert config.monitoring.evaluation_enabled is True
    assert config.monitoring.evaluation_interval_iterations == 50
    assert config.monitoring.interactive_move_simulations_default == 1600
    assert config.monitoring.interactive_move_simulations_minimum == 200
    assert config.monitoring.interactive_move_simulations_maximum == 3200
    assert config.monitoring.interactive_move_simulations_step == 200
    assert config.monitoring.interactive_analysis_simulations == 1600


def test_compact_transport_profile_overrides_only_the_transport() -> None:
    """The Stage 12.2 benchmark profile is an explicit opt-in."""

    profile = Path("config/compact-inference.toml")
    config = load_config(profile_path=profile)

    assert config.self_play.inference_transport == "compact"
    assert config.self_play.worker_processes == 16
    assert config.mcts.training_simulations == 1600


def test_scheduler_restart_profile_targets_one_exact_checkpoint() -> None:
    """The recovery profile opts into one auditable scheduler-only restart."""

    config = load_config(
        profile_path=Path("config/compact-scheduler-restart-iteration-001870.toml")
    )

    assert config.self_play.inference_transport == "compact"
    assert config.training.scheduler_restart.enabled is True
    assert config.training.scheduler_restart.checkpoint_id == "iteration-001870"
    assert config.training.scheduler_restart.peak_learning_rate == 0.0001


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (
            {
                "training.scheduler_restart.enabled": True,
                "training.scheduler_restart.checkpoint_id": "none",
            },
            "must identify a checkpoint",
        ),
        (
            {
                "training.scheduler_restart.enabled": True,
                "training.scheduler_restart.checkpoint_id": "latest",
            },
            "immutable checkpoint ID",
        ),
        (
            {
                "training.scheduler_restart.initial_learning_rate": 0.0002,
                "training.scheduler_restart.peak_learning_rate": 0.0001,
            },
            "initial_learning_rate must not exceed",
        ),
        (
            {
                "training.scheduler_restart.initial_learning_rate": 0.00003,
                "training.scheduler_restart.minimum_learning_rate": 0.00004,
            },
            "minimum_learning_rate must not exceed",
        ),
        (
            {"training.scheduler_restart.peak_learning_rate": 0.0004},
            "must not exceed training.learning_rate",
        ),
    ],
)
def test_invalid_scheduler_restart_parameters_are_rejected(
    override: dict[str, object], message: str
) -> None:
    """Recovery cycles reject ambiguous identities and unsafe LR bounds."""

    with pytest.raises(ConfigError, match=message):
        load_config(overrides=override)


def test_profile_and_overrides_are_applied_in_precedence_order(
    tmp_path: Path,
) -> None:
    """Profile values override defaults and explicit values override profiles."""

    profile_path = tmp_path / "profile.toml"
    profile_path.write_text(
        "[training]\nlearning_rate = 0.001\n[logging]\nlevel = 'DEBUG'\n",
        encoding="utf-8",
    )

    config = load_config(
        profile_path=profile_path,
        overrides={"training.learning_rate": 0.0005},
    )

    assert config.training.learning_rate == 0.0005
    assert config.logging.level == "DEBUG"


def test_unknown_configuration_key_is_rejected(tmp_path: Path) -> None:
    """A misspelled configuration key fails instead of being silently ignored."""

    profile_path = tmp_path / "invalid.toml"
    profile_path.write_text("[mcts]\nunknown_setting = 1\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="Unexpected configuration keys"):
        load_config(profile_path=profile_path)


def test_unknown_configuration_schema_version_is_rejected() -> None:
    """A config cannot claim compatibility with an unsupported schema."""

    with pytest.raises(ConfigError, match=r"Unsupported.*config_schema_version"):
        load_config(overrides={"project.config_schema_version": 99})


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"monitoring.bind_port": 65_536}, "at most 65535"),
        ({"monitoring.event_replay_limit": 4_097}, "must not exceed"),
        ({"monitoring.status_refresh_interval_seconds": 0.0}, "positive"),
        (
            {
                "monitoring.evaluation_enabled": True,
                "monitoring.evaluation_interval_iterations": 0,
            },
            "must be positive",
        ),
        (
            {
                "monitoring.client_reconnect_delay_seconds": 2.0,
                "monitoring.client_reconnect_max_delay_seconds": 1.0,
            },
            "must be at least",
        ),
    ],
)
def test_invalid_monitoring_parameters_are_rejected(
    override: dict[str, object], message: str
) -> None:
    """Public server, recovery, and evaluation controls are schema-validated."""

    with pytest.raises(ConfigError, match=message):
        load_config(overrides=override)


def test_incompatible_ruleset_dimensions_are_rejected() -> None:
    """The board and policy shapes cannot be changed independently."""

    with pytest.raises(ConfigError, match="policy_size"):
        load_config(overrides={"rules.board_size": 15})


def test_all_training_objectives_cannot_be_disabled() -> None:
    """A valid training config must retain at least one learned objective."""

    with pytest.raises(ConfigError, match="At least one training loss"):
        load_config(
            overrides={
                "training.policy_loss_weight": 0.0,
                "training.win_loss_weight": 0.0,
                "training.black_score_loss_weight": 0.0,
                "training.white_score_loss_weight": 0.0,
            }
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"mcts.dirichlet_epsilon": 1.1}, "at most 1.0"),
        ({"mcts.virtual_loss": 0}, "greater than zero"),
        ({"mcts.fpu_reduction": -0.1}, "nonnegative"),
        ({"mcts.max_inference_batch_size": 0}, "greater than zero"),
        ({"mcts.root_noise_only": False}, "root-scoped"),
    ],
)
def test_invalid_mcts_parallelism_parameters_are_rejected(
    override: dict[str, object], message: str
) -> None:
    """Noise and virtual batch controls fail during unified config loading."""

    with pytest.raises(ConfigError, match=message):
        load_config(overrides=override)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"replay.persistence_file_name": "../replay.sqlite3"}, ".sqlite3 file"),
        ({"replay.persistence_chunk_size": 0}, "greater than zero"),
        ({"training.data_loader_workers": -1}, "zero or greater"),
        ({"training.data_loader_prefetch_factor": 0}, "greater than zero"),
        ({"training.batch_size": 200_001}, "must not exceed replay"),
        (
            {"benchmark.replay_benchmark_batch_size": 4_097},
            "must not exceed benchmark positions",
        ),
    ],
)
def test_invalid_replay_and_loader_parameters_are_rejected(
    override: dict[str, object], message: str
) -> None:
    """Stage 8 storage and loading controls fail at unified config loading."""

    with pytest.raises(ConfigError, match=message):
        load_config(overrides=override)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"self_play.games_per_iteration": 0}, "greater than zero"),
        ({"self_play.worker_processes": 0}, "greater than zero"),
        ({"self_play.worker_torch_threads": 0}, "greater than zero"),
        ({"self_play.inference_transport": "invalid"}, "must be one of"),
        ({"self_play.inference_batch_wait_seconds": 0.0}, "positive"),
        ({"self_play.inference_response_timeout_seconds": 0.0}, "positive"),
        ({"self_play.worker_shutdown_timeout_seconds": 0.0}, "positive"),
        ({"self_play.random_seed": -1}, "zero or greater"),
        (
            {"self_play.inference_max_batch_size": 31},
            "at least the MCTS batch size",
        ),
    ],
)
def test_invalid_self_play_parameters_are_rejected(
    override: dict[str, object], message: str
) -> None:
    """Worker, batching, timeout, and seed controls fail during config load."""

    with pytest.raises(ConfigError, match=message):
        load_config(overrides=override)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"training.weight_decay": -0.1}, "nonnegative"),
        ({"training.adamw_beta1": 1.0}, "less than 1.0"),
        ({"training.adamw_beta2": 1.1}, "less than 1.0"),
        ({"training.adamw_epsilon": 0.0}, "positive"),
        ({"training.gradient_clip_norm": 0.0}, "positive"),
        ({"training.batches_per_iteration": 0}, "greater than zero"),
        (
            {"training.minimum_replay_positions": 255},
            "at least training.batch_size",
        ),
        (
            {"training.minimum_replay_positions": 200_001},
            "must not exceed replay capacity",
        ),
        (
            {"training.scheduler_minimum_learning_rate": 0.0004},
            "must not exceed learning_rate",
        ),
        ({"training.scheduler_warmup_steps": 0}, "greater than zero"),
        ({"training.scheduler_decay_steps": 0}, "greater than zero"),
        ({"training.checkpoint_interval_iterations": -1}, "zero or greater"),
        ({"training.amp_max_step_retries": -1}, "zero or greater"),
        ({"training.amp_initial_scale": 0.0}, "positive"),
        ({"training.amp_growth_factor": 1.0}, "greater than 1.0"),
        ({"training.amp_backoff_factor": 1.0}, "less than 1.0"),
        ({"training.amp_growth_interval": 0}, "greater than zero"),
        ({"training.tensorboard_log_interval_batches": 0}, "greater than zero"),
        ({"training.tensorboard_flush_seconds": 0}, "greater than zero"),
        ({"training.tensorboard_max_queue": 0}, "greater than zero"),
        ({"training.random_seed": -1}, "zero or greater"),
    ],
)
def test_invalid_trainer_parameters_are_rejected(
    override: dict[str, object], message: str
) -> None:
    """Optimizer, scheduler, AMP, logging, and lifecycle controls are validated."""

    with pytest.raises(ConfigError, match=message):
        load_config(overrides=override)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"evaluation.game_count": 99}, "must be even"),
        ({"evaluation.game_count": 514}, "opening pairs"),
        ({"evaluation.opening_plies": 256}, "less than the action size"),
        ({"evaluation.paired_colors": False}, "paired-color"),
        ({"evaluation.draw_score": 0.25}, "must be 0.5"),
        (
            {"evaluation.promotion_confidence_lower_bound": 0.55},
            "less than promotion_score",
        ),
        ({"evaluation.inference_max_batch_size": 31}, "MCTS batch size"),
        ({"evaluation.bootstrap_samples": 0}, "greater than zero"),
    ],
)
def test_invalid_evaluation_parameters_are_rejected(
    override: dict[str, object], message: str
) -> None:
    """Arena pairing, confidence, and batching constraints fail at load time."""

    with pytest.raises(ConfigError, match=message):
        load_config(overrides=override)

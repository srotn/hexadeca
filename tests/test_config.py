"""Tests for staged configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import ConfigError, load_config


def test_default_config_matches_confirmed_project_parameters() -> None:
    """The default TOML contains the confirmed rules and training settings."""

    config = load_config()

    assert config.project.config_schema_version == 6
    assert config.rules.board_size == 16
    assert config.rules.score_occupied_cells is True
    assert config.rules.action_size == 256
    assert config.rules.zobrist_seed == 7_640_891_576_956_012_809
    assert config.network.input_planes == 16
    assert config.network.residual_blocks == 10
    assert config.network.channels == 128
    assert config.network.score_normalizer == 16_384.0
    assert config.network.policy_size == 256
    assert config.mcts.training_simulations == 800
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
    assert config.replay.capacity_positions == 200_000
    assert config.replay.persistence_enabled is True
    assert config.replay.persistence_file_name == "replay.sqlite3"
    assert config.replay.persistence_chunk_size == 1_024
    assert config.self_play.games_per_iteration == 32
    assert config.self_play.worker_processes == 4
    assert config.self_play.worker_torch_threads == 1
    assert config.self_play.inference_max_batch_size == 256
    assert config.self_play.inference_batch_wait_seconds == 0.002
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
    assert config.training.checkpoint_interval_iterations == 1
    assert config.training.tensorboard_enabled is True
    assert config.training.tensorboard_log_interval_batches == 1
    assert config.training.tensorboard_flush_seconds == 30
    assert config.training.tensorboard_max_queue == 10
    assert config.training.random_seed == 20_260_803
    assert config.training.policy_loss_weight == 1.0
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

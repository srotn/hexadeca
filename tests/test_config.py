"""Tests for staged configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from config import ConfigError, load_config


def test_default_config_matches_confirmed_project_parameters() -> None:
    """The default TOML contains the confirmed rules and training settings."""

    config = load_config()

    assert config.rules.board_size == 16
    assert config.rules.score_occupied_cells is True
    assert config.rules.action_size == 256
    assert config.network.input_planes == 16
    assert config.network.policy_size == 256
    assert config.mcts.training_simulations == 800
    assert config.mcts.evaluation_simulations == 1600
    assert config.replay.capacity_positions == 200_000
    assert config.training.batch_size == 256


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


def test_incompatible_ruleset_dimensions_are_rejected() -> None:
    """The board and policy shapes cannot be changed independently."""

    with pytest.raises(ConfigError, match="policy_size"):
        load_config(overrides={"rules.board_size": 15})

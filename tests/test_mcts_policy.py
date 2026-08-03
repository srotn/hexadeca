"""Tests for configured noise, temperature, and action sampling."""

from __future__ import annotations

import random

import pytest

from config import load_config
from game import Player
from mcts import (
    MctsNode,
    SearchMode,
    add_root_dirichlet_noise,
    configured_temperature,
    sample_action,
    visit_probabilities,
)


def test_dirichlet_noise_is_seeded_normalized_and_uses_configured_epsilon() -> None:
    """Equal roots and RNG seeds produce identical mixed legal priors."""

    config = load_config().mcts
    first = MctsNode(Player.BLACK, zobrist_hash=1, terminal=False)
    second = MctsNode(Player.BLACK, zobrist_hash=1, terminal=False)
    first.expand({0: 0.5, 2: 0.5})
    second.expand({0: 0.5, 2: 0.5})

    add_root_dirichlet_noise(first, config, random.Random(17))
    add_root_dirichlet_noise(second, config, random.Random(17))

    first_priors = tuple(edge.prior for edge in first.children.values())
    second_priors = tuple(edge.prior for edge in second.children.values())
    assert first_priors == second_priors
    assert first_priors != (0.5, 0.5)
    assert sum(first_priors) == pytest.approx(1.0)
    assert all(
        prior >= (1.0 - config.dirichlet_epsilon) * 0.5 for prior in first_priors
    )


def test_temperature_one_and_zero_have_exact_visit_semantics() -> None:
    """Tau one normalizes visits and tau zero uses the lower-action tie-break."""

    assert visit_probabilities((1, 0, 3), 1.0) == pytest.approx((0.25, 0.0, 0.75))
    assert visit_probabilities((4, 4, 1), 0.0) == (1.0, 0.0, 0.0)


def test_configured_temperature_uses_training_ply_boundary() -> None:
    """Only the configured opening plies use stochastic temperature."""

    config = load_config().mcts

    assert configured_temperature(config, SearchMode.TRAINING, 14) == 1.0
    assert configured_temperature(config, SearchMode.TRAINING, 15) == 0.0
    assert configured_temperature(config, SearchMode.EVALUATION, 0) == 0.0


def test_seeded_action_sampling_is_reproducible() -> None:
    """Callers own RNG state, enabling deterministic worker-level replay."""

    probabilities = (0.2, 0.3, 0.5)

    assert sample_action(probabilities, random.Random(9)) == sample_action(
        probabilities, random.Random(9)
    )

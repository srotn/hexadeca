"""Root noise, visit distributions, and action sampling for MCTS."""

from __future__ import annotations

import random
from enum import Enum
from math import isfinite

from config.schema import MctsConfig
from mcts.errors import NodeStateError
from mcts.node import MctsNode


class SearchMode(Enum):
    """Configured search behavior for self-play training or evaluation."""

    TRAINING = "training"
    EVALUATION = "evaluation"


def simulation_count(config: MctsConfig, mode: SearchMode) -> int:
    """Return the mode-specific number of configured simulations."""

    if mode is SearchMode.TRAINING:
        return config.training_simulations
    return config.evaluation_simulations


def add_root_dirichlet_noise(
    root: MctsNode,
    config: MctsConfig,
    random_source: random.Random,
) -> None:
    """Mix configured Dirichlet noise into all legal root priors in place."""

    if not config.root_noise_enabled:
        raise NodeStateError("Root noise is disabled by MCTS configuration")
    if not config.root_noise_only:
        raise NodeStateError("Only root-scoped MCTS noise is supported")
    if not root.expanded or not root.children:
        raise NodeStateError("Root must be expanded before adding Dirichlet noise")
    noise = [
        random_source.gammavariate(config.dirichlet_alpha, 1.0) for _ in root.children
    ]
    noise_sum = sum(noise)
    if not isfinite(noise_sum) or noise_sum <= 0:
        raise NodeStateError("Dirichlet sampling produced an invalid total")
    epsilon = config.dirichlet_epsilon
    retained = 1.0 - epsilon
    for edge, sample in zip(root.children.values(), noise, strict=True):
        edge.prior = retained * edge.prior + epsilon * sample / noise_sum


def visit_probabilities(
    visit_counts: tuple[int, ...], temperature: float
) -> tuple[float, ...]:
    """Convert visit counts to a stable distribution at nonnegative temperature."""

    if not visit_counts or any(
        type(count) is not int or count < 0 for count in visit_counts
    ):
        raise NodeStateError("Visit counts must be a non-empty nonnegative tuple")
    if not isfinite(temperature) or temperature < 0:
        raise NodeStateError("Temperature must be finite and nonnegative")
    maximum = max(visit_counts)
    if maximum == 0:
        raise NodeStateError("At least one action must have a visit")

    if temperature == 0:
        selected_action = visit_counts.index(maximum)
        return tuple(
            1.0 if action == selected_action else 0.0
            for action in range(len(visit_counts))
        )

    exponent = 1.0 / temperature
    weights = [
        (count / maximum) ** exponent if count > 0 else 0.0 for count in visit_counts
    ]
    total = sum(weights)
    if not isfinite(total) or total <= 0:
        raise NodeStateError("Temperature scaling produced an invalid distribution")
    return tuple(weight / total for weight in weights)


def configured_temperature(config: MctsConfig, mode: SearchMode, ply: int) -> float:
    """Return the configured temperature for a mode and zero-based ply."""

    if type(ply) is not int or ply < 0:
        raise NodeStateError("Ply must be a nonnegative integer")
    if mode is SearchMode.TRAINING and ply < config.stochastic_plies:
        return config.opening_temperature
    return config.endgame_temperature


def sample_action(
    probabilities: tuple[float, ...], random_source: random.Random
) -> int:
    """Sample a normalized action distribution with a stable final fallback."""

    if not probabilities:
        raise NodeStateError("Action probabilities must not be empty")
    if any(not isfinite(value) or value < 0 for value in probabilities):
        raise NodeStateError("Action probabilities must be finite and nonnegative")
    total = sum(probabilities)
    if not isfinite(total) or abs(total - 1.0) > 1e-6:
        raise NodeStateError("Action probabilities must sum to one")

    threshold = random_source.random()
    cumulative = 0.0
    last_nonzero = 0
    for action, probability in enumerate(probabilities):
        if probability > 0:
            last_nonzero = action
        cumulative += probability
        if threshold < cumulative:
            return action
    return last_nonzero

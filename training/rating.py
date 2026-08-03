"""Paired-bootstrap confidence bounds and finite performance Elo."""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import floor, isfinite, log10

from config.schema import EvaluationConfig
from training.arena import ArenaBatch
from training.errors import EvaluationStatisticsError


@dataclass(frozen=True, slots=True)
class ArenaStatistics:
    """Candidate result, uncertainty, Elo, and promotion decision."""

    games: int
    opening_pairs: int
    candidate_wins: int
    best_wins: int
    draws: int
    candidate_points: float
    score_rate: float
    confidence_level: float
    score_confidence_lower: float
    score_confidence_upper: float
    performance_elo: float
    performance_elo_lower: float
    performance_elo_upper: float
    promotion_score: float
    promotion_confidence_lower_bound: float
    promoted: bool


def calculate_arena_statistics(
    batch: ArenaBatch, config: EvaluationConfig
) -> ArenaStatistics:
    """Calculate a deterministic opening-pair bootstrap and promotion gate."""

    if len(batch.games) != config.game_count:
        raise EvaluationStatisticsError(
            "Arena game count does not match evaluation configuration"
        )
    if len(batch.openings) != config.opening_pair_count:
        raise EvaluationStatisticsError(
            "Arena opening count does not match evaluation configuration"
        )
    pair_points = tuple(
        batch.games[index].candidate_points + batch.games[index + 1].candidate_points
        for index in range(0, len(batch.games), 2)
    )
    if any(not isfinite(points) or not 0.0 <= points <= 2.0 for points in pair_points):
        raise EvaluationStatisticsError("Arena opening-pair points are invalid")

    candidate_points = sum(pair_points)
    score_rate = candidate_points / config.game_count
    lower, upper = paired_bootstrap_interval(pair_points, config)
    performance = performance_elo(score_rate, config.game_count, config)
    elo_lower = performance_elo(lower, config.game_count, config)
    elo_upper = performance_elo(upper, config.game_count, config)
    promoted = (
        score_rate >= config.promotion_score
        and lower > config.promotion_confidence_lower_bound
    )
    return ArenaStatistics(
        games=config.game_count,
        opening_pairs=config.opening_pair_count,
        candidate_wins=batch.candidate_wins,
        best_wins=batch.best_wins,
        draws=batch.draws,
        candidate_points=candidate_points,
        score_rate=score_rate,
        confidence_level=config.confidence_level,
        score_confidence_lower=lower,
        score_confidence_upper=upper,
        performance_elo=performance,
        performance_elo_lower=elo_lower,
        performance_elo_upper=elo_upper,
        promotion_score=config.promotion_score,
        promotion_confidence_lower_bound=(config.promotion_confidence_lower_bound),
        promoted=promoted,
    )


def paired_bootstrap_interval(
    pair_points: tuple[float, ...], config: EvaluationConfig
) -> tuple[float, float]:
    """Return a seeded percentile interval by resampling complete opening pairs."""

    if len(pair_points) != config.opening_pair_count or not pair_points:
        raise EvaluationStatisticsError(
            "Bootstrap input must contain every configured opening pair"
        )
    if any(not isfinite(points) or not 0.0 <= points <= 2.0 for points in pair_points):
        raise EvaluationStatisticsError("Bootstrap pair points must be within [0, 2]")

    random_source = random.Random(config.random_seed)
    samples = sorted(
        sum(random_source.choice(pair_points) for _ in pair_points) / config.game_count
        for _ in range(config.bootstrap_samples)
    )
    tail = (1.0 - config.confidence_level) / 2.0
    return _quantile(samples, tail), _quantile(samples, 1.0 - tail)


def performance_elo(
    score_rate: float, game_count: int, config: EvaluationConfig
) -> float:
    """Convert match score to finite pairwise Elo using a symmetric prior."""

    if not isfinite(score_rate) or not 0.0 <= score_rate <= 1.0:
        raise EvaluationStatisticsError("Arena score rate must be within [0, 1]")
    if type(game_count) is not int or game_count <= 0:
        raise EvaluationStatisticsError("Arena Elo game count must be positive")
    prior = config.elo_prior_points
    adjusted = (score_rate * game_count + prior) / (game_count + 2.0 * prior)
    return config.elo_scale * log10(adjusted / (1.0 - adjusted))


def _quantile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        raise EvaluationStatisticsError("Cannot calculate an empty quantile")
    position = probability * (len(sorted_values) - 1)
    lower_index = floor(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )

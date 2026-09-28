"""Decision-quality, calibration, and score-ranking metrics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import sqrt
from statistics import fmean
from typing import Any

VALUE_BAND_THRESHOLD = 1.0 / 3.0


def build_search_summary(
    dataset: Mapping[str, Any],
    candidate_positions: Sequence[Mapping[str, Any]],
    reference_positions: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Aggregate one search configuration, optionally against a reference."""

    labels = {
        position["position_id"]: _outcome_for_position(position)
        for position in dataset["positions"]
    }
    root_errors: list[float] = []
    brier_scores: list[float] = []
    for candidate in candidate_positions:
        outcome = labels[candidate["position_id"]]
        root_value = float(candidate["root_value"])
        root_errors.append(abs(root_value - outcome))
        brier_scores.append(((root_value + 1.0) / 2.0 - (outcome + 1.0) / 2.0) ** 2)
    summary: dict[str, Any] = {
        "positions": len(candidate_positions),
        "mean_absolute_root_value_error": _mean(root_errors),
        "root_value_brier_score": _mean(brier_scores),
        "mean_search_seconds": _mean(
            [float(position["elapsed_seconds"]) for position in candidate_positions]
        ),
        "total_inference_positions": sum(
            int(position["inference_positions"]) for position in candidate_positions
        ),
    }
    if reference_positions is None:
        return summary

    reference_by_id = {
        position["position_id"]: position for position in reference_positions
    }
    regrets: list[float] = []
    agreements = 0
    top_hits = {1: 0, 3: 0, 5: 0}
    missing_reference_q = 0
    win_to_draw = 0
    win_to_loss = 0
    low_prior_recoveries = 0
    low_prior_opportunities = 0
    for candidate in candidate_positions:
        reference = reference_by_id[candidate["position_id"]]
        reference_action = int(reference["selected_action"])
        selected_action = int(candidate["selected_action"])
        agreements += selected_action == reference_action
        policy_order = sorted(
            candidate["action_statistics"],
            key=lambda item: (-float(item["prior"]), int(item["action"])),
        )
        for size in top_hits:
            top_hits[size] += reference_action in {
                int(item["action"]) for item in policy_order[:size]
            }
        reference_q = {
            int(item["action"]): float(item["q_value"])
            for item in reference["action_statistics"]
            if int(item["visits"]) > 0
        }
        best_q = reference_q.get(reference_action)
        selected_q = reference_q.get(selected_action)
        if best_q is None or selected_q is None:
            missing_reference_q += 1
            continue
        regrets.append(max(0.0, best_q - selected_q))
        best_band = _value_band(best_q)
        selected_band = _value_band(selected_q)
        if best_band == "win" and selected_band == "draw_or_uncertain":
            win_to_draw += 1
        elif best_band == "win" and selected_band == "loss":
            win_to_loss += 1
        reference_prior = next(
            float(item["prior"])
            for item in reference["action_statistics"]
            if int(item["action"]) == reference_action
        )
        if reference_prior < 0.05:
            low_prior_opportunities += 1
            low_prior_recoveries += selected_action == reference_action

    count = len(candidate_positions)
    summary["reference_comparison"] = {
        "reference_positions": len(reference_positions),
        "selected_action_agreement": agreements / count if count else 0.0,
        "policy_reference_action_recall": {
            f"top_{size}": top_hits[size] / count if count else 0.0 for size in top_hits
        },
        "mean_reference_q_regret": _mean(regrets),
        "p95_reference_q_regret": _quantile(regrets, 0.95),
        "maximum_reference_q_regret": max(regrets, default=0.0),
        "reference_win_to_draw_or_uncertain": win_to_draw,
        "reference_win_to_loss": win_to_loss,
        "missing_selected_reference_q": missing_reference_q,
        "low_prior_reference_actions": low_prior_opportunities,
        "low_prior_reference_action_recovery_rate": (
            low_prior_recoveries / low_prior_opportunities
            if low_prior_opportunities
            else None
        ),
        "value_band_threshold": VALUE_BAND_THRESHOLD,
    }
    return summary


def build_score_summary(
    root_records: Sequence[Mapping[str, Any]],
    ranking_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate score calibration and candidate-action ranking evidence."""

    black_errors = [
        abs(float(item["predicted_black_score"]) - int(item["final_black_score"]))
        for item in root_records
    ]
    white_errors = [
        abs(float(item["predicted_white_score"]) - int(item["final_white_score"]))
        for item in root_records
    ]
    predicted_margins = [float(item["predicted_margin"]) for item in root_records]
    final_margins = [float(item["final_margin"]) for item in root_records]
    score_regrets = [
        float(item["score_head_reference_q_regret"]) for item in ranking_records
    ]
    policy_regrets = [
        float(item["policy_reference_q_regret"]) for item in ranking_records
    ]
    correlations = [
        float(item["score_q_correlation"])
        for item in ranking_records
        if item["score_q_correlation"] is not None
    ]
    count = len(ranking_records)
    return {
        "root_calibration": {
            "positions": len(root_records),
            "black_score_mae": _mean(black_errors),
            "white_score_mae": _mean(white_errors),
            "margin_mae": _mean(
                [
                    abs(predicted - actual)
                    for predicted, actual in zip(
                        predicted_margins, final_margins, strict=True
                    )
                ]
            ),
            "margin_correlation": pearson_correlation(predicted_margins, final_margins),
            "margin_sign_accuracy": _mean(
                [
                    float(_sign(predicted) == _sign(actual))
                    for predicted, actual in zip(
                        predicted_margins, final_margins, strict=True
                    )
                ]
            ),
        },
        "candidate_ranking": {
            "positions": count,
            "mean_candidates": _mean(
                [float(item["candidate_count"]) for item in ranking_records]
            ),
            "score_head_top_1_reference_agreement": _mean(
                [
                    float(item["score_selected_action"] == item["reference_action"])
                    for item in ranking_records
                ]
            ),
            "policy_top_1_reference_agreement": _mean(
                [
                    float(item["policy_selected_action"] == item["reference_action"])
                    for item in ranking_records
                ]
            ),
            "mean_score_q_correlation": _mean(correlations),
            "mean_score_head_reference_q_regret": _mean(score_regrets),
            "mean_policy_reference_q_regret": _mean(policy_regrets),
            "score_head_regret_improvement_over_policy": (
                _mean(policy_regrets) - _mean(score_regrets)
            ),
            "score_head_lower_regret_rate": _mean(
                [
                    float(score < policy)
                    for score, policy in zip(score_regrets, policy_regrets, strict=True)
                ]
            ),
        },
    }


def pearson_correlation(
    first: Sequence[float], second: Sequence[float]
) -> float | None:
    """Return Pearson's r, or ``None`` for insufficient/constant samples."""

    if len(first) != len(second) or len(first) < 2:
        return None
    first_mean = fmean(first)
    second_mean = fmean(second)
    covariance = sum(
        (left - first_mean) * (right - second_mean)
        for left, right in zip(first, second, strict=True)
    )
    first_variance = sum((value - first_mean) ** 2 for value in first)
    second_variance = sum((value - second_mean) ** 2 for value in second)
    denominator = sqrt(first_variance * second_variance)
    return covariance / denominator if denominator > 0 else None


def _outcome_for_position(position: Mapping[str, Any]) -> float:
    winner = position["final_winner"]
    if winner is None:
        return 0.0
    return 1.0 if winner == position["to_play"] else -1.0


def _value_band(value: float) -> str:
    if value >= VALUE_BAND_THRESHOLD:
        return "win"
    if value <= -VALUE_BAND_THRESHOLD:
        return "loss"
    return "draw_or_uncertain"


def _sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def _mean(values: Sequence[float]) -> float | None:
    return fmean(values) if values else None


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(probability * len(ordered)))
    return ordered[index]

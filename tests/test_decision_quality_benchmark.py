"""Offline decision-quality dataset, persistence, and metric tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
import json

import pytest

from benchmark.decision_quality.arena import _paired_statistics
from benchmark.decision_quality.metrics import (
    build_score_summary,
    build_search_summary,
)
from benchmark.decision_quality.runtime import generate_dataset, replay_state
from benchmark.decision_quality.schema import (
    DecisionQualityDataError,
    load_dataset,
)
from config import load_config
from game import GameState
from mcts import Evaluation


class UniformEvaluator:
    """Small deterministic evaluator used without a neural checkpoint."""

    def evaluate(self, states: Sequence[GameState]) -> tuple[Evaluation, ...]:
        return tuple(_uniform_evaluation(state) for state in states)


def test_dataset_generation_is_replayable_stratified_and_idempotent(tmp_path) -> None:
    """Generation persists complete games and balanced non-terminal positions."""

    base = load_config()
    config = replace(
        base,
        rules=replace(base.rules, board_size=6),
        mcts=replace(
            base.mcts,
            engine="python",
            evaluation_simulations=2,
            max_inference_batch_size=2,
            exact_endgame_enabled=False,
        ),
    )
    checkpoint = {"checkpoint_id": "test-model", "state_sha256": "a" * 64}
    destination = tmp_path / "dataset.json"

    first = generate_dataset(
        config,
        checkpoint,
        UniformEvaluator(),  # type: ignore[arg-type]
        output_path=destination,
        games=3,
        positions_per_game=3,
        opening_plies=1,
        simulations=2,
        c_puct=2.3,
        random_seed=17,
        restart=False,
    )
    second = generate_dataset(
        config,
        checkpoint,
        UniformEvaluator(),  # type: ignore[arg-type]
        output_path=destination,
        games=3,
        positions_per_game=3,
        opening_plies=1,
        simulations=2,
        c_puct=2.3,
        random_seed=17,
        restart=False,
    )

    assert first == second == load_dataset(destination)
    assert len(first["games"]) == 3
    assert len(first["positions"]) == 9
    assert {item["stratum"] for item in first["positions"]} == {
        "opening",
        "middle",
        "endgame",
    }
    for position in first["positions"]:
        state = replay_state(config, position["history"])
        assert state.zobrist_hash == position["zobrist_hash"]
        assert state.terminal is False


def test_dataset_loader_rejects_unbalanced_strata(tmp_path) -> None:
    """A changed position selection cannot silently masquerade as the fixed set."""

    base = load_config()
    config = replace(
        base,
        rules=replace(base.rules, board_size=6),
        mcts=replace(
            base.mcts,
            engine="python",
            evaluation_simulations=1,
            max_inference_batch_size=1,
            exact_endgame_enabled=False,
        ),
    )
    destination = tmp_path / "dataset.json"
    generate_dataset(
        config,
        {"checkpoint_id": "test-model", "state_sha256": "b" * 64},
        UniformEvaluator(),  # type: ignore[arg-type]
        output_path=destination,
        games=1,
        positions_per_game=3,
        opening_plies=1,
        simulations=1,
        c_puct=2.3,
        random_seed=9,
        restart=False,
    )
    raw = json.loads(destination.read_text(encoding="utf-8"))
    raw["positions"][0]["stratum"] = "middle"
    destination.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(DecisionQualityDataError, match="strata"):
        load_dataset(destination)


def test_search_summary_measures_reference_regret_and_policy_recall() -> None:
    """Reference disagreement produces the expected Q regret and severity count."""

    dataset = {
        "positions": [
            {
                "position_id": "p1",
                "to_play": "black",
                "final_winner": "black",
            }
        ]
    }
    candidate = [_search_position("p1", selected=2, root_value=0.2)]
    reference = [_search_position("p1", selected=1, root_value=0.8)]
    reference[0]["action_statistics"] = [
        {"action": 1, "prior": 0.04, "visits": 80, "q_value": 0.8},
        {"action": 2, "prior": 0.60, "visits": 20, "q_value": -0.5},
    ]

    summary = build_search_summary(dataset, candidate, reference)
    comparison = summary["reference_comparison"]

    assert comparison["selected_action_agreement"] == 0.0
    assert comparison["mean_reference_q_regret"] == pytest.approx(1.3)
    assert comparison["reference_win_to_loss"] == 1
    assert comparison["policy_reference_action_recall"]["top_1"] == 0.0
    assert comparison["policy_reference_action_recall"]["top_3"] == 1.0
    assert comparison["low_prior_reference_actions"] == 1


def test_score_summary_compares_score_ranking_against_policy() -> None:
    """Score-head ranking reports positive regret improvement when it chooses better."""

    summary = build_score_summary(
        [
            {
                "predicted_black_score": 10.0,
                "predicted_white_score": 8.0,
                "predicted_margin": 2.0,
                "final_black_score": 11,
                "final_white_score": 8,
                "final_margin": 3.0,
            },
            {
                "predicted_black_score": 7.0,
                "predicted_white_score": 9.0,
                "predicted_margin": -2.0,
                "final_black_score": 7,
                "final_white_score": 10,
                "final_margin": -3.0,
            },
        ],
        [
            {
                "candidate_count": 4,
                "reference_action": 3,
                "score_selected_action": 3,
                "policy_selected_action": 4,
                "score_q_correlation": 0.8,
                "score_head_reference_q_regret": 0.0,
                "policy_reference_q_regret": 0.4,
            }
        ],
    )

    ranking = summary["candidate_ranking"]
    assert ranking["score_head_top_1_reference_agreement"] == 1.0
    assert ranking["score_head_regret_improvement_over_policy"] == pytest.approx(0.4)
    assert ranking["score_head_lower_regret_rate"] == 1.0


def test_paired_arena_statistics_preserve_opening_pair_dependence() -> None:
    """Color-swapped games are summarized as pairs rather than IID samples."""

    class Game:
        def __init__(self, candidate_points: float) -> None:
            self.candidate_points = candidate_points

    summary = _paired_statistics(
        [Game(1.0), Game(1.0), Game(1.0), Game(0.0), Game(0.0), Game(0.0)]
    )

    assert summary == {
        "candidate_pair_wins": 1,
        "tied_pairs": 1,
        "candidate_pair_losses": 1,
        "score_rates": [1.0, 0.5, 0.0],
    }


def _uniform_evaluation(state: GameState) -> Evaluation:
    legal_count = sum(state.legal_mask)
    return Evaluation(
        policy=tuple(1.0 / legal_count if legal else 0.0 for legal in state.legal_mask),
        value=0.0,
    )


def _search_position(
    position_id: str, *, selected: int, root_value: float
) -> dict[str, object]:
    return {
        "position_id": position_id,
        "root_value": root_value,
        "elapsed_seconds": 1.0,
        "inference_positions": 10,
        "selected_action": selected,
        "action_statistics": [
            {"action": 1, "prior": 0.4, "visits": 5, "q_value": 0.1},
            {"action": 2, "prior": 0.6, "visits": 6, "q_value": 0.0},
        ],
    }

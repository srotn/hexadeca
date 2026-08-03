"""Stage 11 complete paired Arena benchmark contract tests."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

from benchmark.evaluation import run_evaluation_benchmark
from config import load_config
from network import NetworkSpecification


def test_evaluation_benchmark_reports_end_to_end_arena_throughput() -> None:
    """A tiny CPU benchmark still uses two models, workers, MCTS, and scoring."""

    config = load_config()
    rules = replace(config.rules, board_size=4)
    network = replace(
        config.network,
        residual_blocks=1,
        channels=4,
        value_hidden_features=8,
        policy_size=16,
        score_normalizer=16.0,
    )
    mcts = replace(config.mcts, max_inference_batch_size=2)
    evaluation = replace(
        config.evaluation,
        game_count=2,
        opening_plies=1,
        bootstrap_samples=100,
        worker_processes=1,
        inference_max_batch_size=4,
        inference_batch_wait_seconds=0.001,
    )
    benchmark_config = replace(
        config,
        rules=rules,
        network=network,
        mcts=mcts,
        evaluation=evaluation,
    )
    specification = NetworkSpecification.from_config(rules, network)

    report = run_evaluation_benchmark(
        benchmark_config,
        specification,
        game_count=2,
        simulations=2,
        include_cuda=False,
    )

    network_report = cast(dict[str, Any], report["network"])
    parameters = cast(dict[str, Any], report["parameters"])
    results = cast(dict[str, Any], report["results"])
    assert network_report["models_loaded"] == 2
    assert parameters["games"] == 2
    assert parameters["opening_pairs"] == 1
    assert parameters["device"] == "cpu"
    assert results["simulations"] > 0
    assert results["simulations_per_second"] > 0.0
    assert results["games_per_second"] > 0.0
    assert results["inference_positions"] > 0
    assert results["candidate_score_rate"] == 0.5

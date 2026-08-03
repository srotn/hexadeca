"""Stage 9 end-to-end self-play benchmark contract tests."""

from __future__ import annotations

from dataclasses import replace

from benchmark.self_play import run_self_play_benchmark
from config import load_config
from network import NetworkSpecification


def test_self_play_benchmark_reports_end_to_end_throughput() -> None:
    """A small real-network run reports exact games, positions, and searches."""

    config = load_config()
    mcts = replace(
        config.mcts,
        training_simulations=2,
        max_inference_batch_size=2,
    )
    self_play = replace(
        config.self_play,
        games_per_iteration=1,
        worker_processes=1,
        inference_max_batch_size=2,
        inference_batch_wait_seconds=0.001,
        inference_response_timeout_seconds=30.0,
        worker_shutdown_timeout_seconds=5.0,
    )
    specification = replace(
        NetworkSpecification.from_config(config.rules, config.network),
        residual_blocks=1,
        channels=4,
        value_hidden_features=8,
    )

    report = run_self_play_benchmark(
        config.rules,
        mcts,
        self_play,
        specification,
        games=1,
        simulations=2,
        random_seed=17,
        include_cuda=False,
    )

    self_play_report = report["self_play"]
    assert isinstance(self_play_report, dict)
    assert self_play_report["active_worker_processes"] == 1
    assert self_play_report["simulations_per_move"] == 2
    results = report["results"]
    assert isinstance(results, dict)
    assert results["games_completed"] == 1
    assert int(results["positions_generated"]) > 0
    assert results["simulations_completed"] == (int(results["positions_generated"]) * 2)
    assert float(results["games_per_second"]) > 0.0
    assert float(results["positions_per_second"]) > 0.0
    assert float(results["simulations_per_second"]) > 0.0
    assert int(results["central_inference_batches"]) > 0
    assert int(results["central_inference_positions"]) > 0
    assert 1.0 <= float(results["central_average_inference_batch_size"]) <= 2.0
    assert int(results["central_maximum_inference_batch_size"]) <= 2

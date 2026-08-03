"""Smoke tests for the Stage 7 reproducible search benchmark."""

from __future__ import annotations

from dataclasses import replace

from benchmark.mcts import run_mcts_benchmark
from config import load_config
from network import NetworkSpecification


def test_mcts_benchmark_reports_simulation_and_batch_throughput() -> None:
    """A minimal core run preserves exact search and inference units."""

    config = load_config()
    mcts_config = replace(config.mcts, inference_batch_size=4)
    specification = replace(
        NetworkSpecification.from_config(config.rules, config.network),
        residual_blocks=1,
        channels=4,
        value_hidden_features=8,
    )

    report = run_mcts_benchmark(
        config.rules,
        mcts_config,
        specification,
        warmup_iterations=0,
        measurement_iterations=1,
        benchmark_simulations=8,
        random_seed=3,
        include_cuda=False,
    )

    results = report["results"]
    assert isinstance(results, dict)
    core = results["python_uniform_core"]
    assert isinstance(core, dict)
    assert core["simulations"] == 8
    assert float(core["simulations_per_second"]) > 0
    assert core["maximum_inference_batch_size"] == 4

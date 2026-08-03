"""Stage 10 real optimizer benchmark contract tests."""

from __future__ import annotations

from dataclasses import replace

from benchmark.trainer import run_trainer_benchmark
from config import load_config
from network import NetworkSpecification


def test_trainer_benchmark_reports_update_and_position_throughput() -> None:
    """A small CPU run measures exact real-network optimizer work."""

    config = load_config()
    network = replace(
        config.network,
        residual_blocks=1,
        channels=4,
        value_hidden_features=8,
    )
    training = replace(
        config.training,
        batch_size=2,
        minimum_replay_positions=2,
        data_loader_pin_memory=False,
    )
    replay = replace(config.replay, capacity_positions=8)
    benchmark_config = replace(
        config,
        network=network,
        training=training,
        replay=replay,
    )
    specification = NetworkSpecification.from_config(
        benchmark_config.rules, benchmark_config.network
    )

    report = run_trainer_benchmark(
        benchmark_config,
        specification,
        warmup_batches=1,
        measurement_batches=2,
        random_seed=29,
        include_cuda=False,
    )

    results = report["results"]
    assert isinstance(results, dict)
    assert results["device"] == "cpu"
    assert results["precision"] == "float32"
    assert results["batches"] == 2
    assert results["positions"] == 4
    assert float(results["batches_per_second"]) > 0.0
    assert float(results["positions_per_second"]) > 0.0
    assert float(results["mean_total_loss"]) > 0.0
    assert float(results["mean_gradient_norm"]) > 0.0

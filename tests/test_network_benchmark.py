"""Smoke tests for the Stage 6 reproducible inference benchmark."""

from __future__ import annotations

from dataclasses import replace

from benchmark.network import run_network_benchmark
from config import load_config
from network import NetworkSpecification


def test_network_benchmark_reports_batch_and_position_throughput() -> None:
    """A minimal CPU run retains metadata and throughput units."""

    config = load_config()
    specification = replace(
        NetworkSpecification.from_config(config.rules, config.network),
        residual_blocks=1,
        channels=4,
        value_hidden_features=8,
    )

    report = run_network_benchmark(
        specification,
        warmup_iterations=0,
        measurement_iterations=1,
        cpu_batch_size=2,
        gpu_batch_sizes=(1,),
        random_seed=3,
        include_cuda=False,
    )

    results = report["results"]
    assert isinstance(results, dict)
    cpu = results["cpu_float32"]
    assert isinstance(cpu, dict)
    assert cpu["batch_size"] == 2
    assert float(cpu["positions_per_second"]) > 0
    assert report["network"]["feature_schema_id"] == "hexadeca-v1-16p"  # type: ignore[index]

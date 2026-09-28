"""Tests for the Stage 12.2 compact transport benchmark."""

from __future__ import annotations

from dataclasses import replace

from benchmark.transport import run_transport_benchmark
from config import load_config


def test_transport_benchmark_reports_equivalent_protocol_comparison() -> None:
    """The benchmark exposes request and response size/timing measurements."""

    config = load_config()
    benchmark = replace(
        config.benchmark,
        transport_warmup_iterations=0,
        transport_measurement_iterations=1,
        transport_batch_size=4,
    )

    report = run_transport_benchmark(
        config.rules, benchmark, random_seed=config.benchmark.random_seed
    )

    assert report["schema_version"] == 1
    request = report["request"]
    response = report["response"]
    assert isinstance(request, dict)
    assert isinstance(response, dict)
    assert request["compact_bytes"] < request["object_bytes"]
    assert response["compact_bytes"] < response["object_bytes"]

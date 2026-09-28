"""Smoke coverage for the Stage 13 monitoring benchmark report."""

from __future__ import annotations

from benchmark.monitoring import run_monitoring_benchmark
from config import load_config


def test_monitoring_benchmark_reports_publish_and_serialization_costs() -> None:
    """A minimal run records latency/throughput with a full board payload."""

    report = run_monitoring_benchmark(
        load_config().rules,
        load_config().mcts,
        warmup_events=0,
        measurement_events=2,
    )

    results = report["results"]
    assert isinstance(results, dict)
    assert float(results["event_publish"]["throughput_ops_per_second"]) > 0.0
    assert float(results["websocket_json_serialization"]["median_ns"]) > 0.0
    assert int(results["event_payload_bytes"]) > 256

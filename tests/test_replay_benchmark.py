"""Stage 8 benchmark report contract tests."""

from __future__ import annotations

from pathlib import Path

from benchmark.replay import run_replay_benchmark
from config import load_config
from network import NetworkSpecification


def test_replay_benchmark_reports_exact_storage_and_throughput_units(
    tmp_path: Path,
) -> None:
    """A small run reports every Stage 8 path in positions per second."""

    config = load_config()
    specification = NetworkSpecification.from_config(config.rules, config.network)

    report = run_replay_benchmark(
        config.rules,
        specification,
        config.replay,
        positions=8,
        batch_size=4,
        warmup_iterations=0,
        measurement_iterations=1,
        random_seed=7,
        working_directory=tmp_path,
    )

    replay = report["replay"]
    assert isinstance(replay, dict)
    assert replay["retained_positions"] == 8
    assert replay["database_bytes"] > 0
    results = report["results"]
    assert isinstance(results, dict)
    assert results["ring_batch_append"]["positions_per_operation"] == 4
    assert results["feature_target_collation"]["positions_per_second"] > 0
    assert results["sqlite_snapshot_save"]["positions_per_operation"] == 8
    assert results["sqlite_snapshot_load"]["positions_per_second"] > 0

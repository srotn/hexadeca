"""Smoke tests for the reproducible Stage 3 board benchmark."""

from __future__ import annotations

from typing import Any, cast

from benchmark.board_engine import run_board_benchmarks
from config import load_config


def test_board_benchmark_reports_all_stage_three_operations() -> None:
    """A short benchmark run reports configured metadata and all operations."""

    report = run_board_benchmarks(
        load_config().rules,
        warmup_iterations=1,
        measurement_iterations=3,
        position_plies=4,
        random_seed=7,
    )

    parameters = cast(dict[str, Any], report["parameters"])
    results = cast(dict[str, Any], report["results"])
    assert parameters["actual_position_plies"] == 4
    assert set(results) == {
        "apply_undo_cycle",
        "board_construction",
        "legal_action_generation",
    }
    assert all(result["iterations"] == 3 for result in results.values())
    assert all(result["median_ns"] > 0 for result in results.values())

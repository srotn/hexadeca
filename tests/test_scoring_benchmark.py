"""Smoke tests for the reproducible Stage 4 scoring benchmark."""

from __future__ import annotations

from typing import Any, cast

from benchmark.scoring import run_scoring_benchmark
from config import load_config


def test_scoring_benchmark_reports_terminal_position_and_timing() -> None:
    """A short scoring benchmark reports rule, result, and timing metadata."""

    report = run_scoring_benchmark(
        load_config().rules,
        warmup_iterations=1,
        measurement_iterations=3,
        random_seed=11,
    )

    position = cast(dict[str, Any], report["position"])
    results = cast(dict[str, Any], report["results"])
    timing = results["terminal_scoring"]
    assert position["plies"] > 0
    assert position["black_score"] >= 0
    assert position["white_score"] >= 0
    assert timing["iterations"] == 3
    assert timing["median_ns"] > 0

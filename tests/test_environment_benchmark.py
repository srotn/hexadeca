"""Smoke tests for the reproducible Stage 5 environment benchmark."""

from __future__ import annotations

from typing import Any, cast

from benchmark.environment import run_environment_benchmark
from config import load_config


def test_environment_benchmark_reports_full_terminal_replay() -> None:
    """A short benchmark reports one reproducible complete game and timing."""

    report = run_environment_benchmark(
        load_config().rules,
        warmup_iterations=1,
        measurement_iterations=3,
        random_seed=13,
    )

    game = cast(dict[str, Any], report["game"])
    results = cast(dict[str, Any], report["results"])
    timing = results["full_game_environment"]
    assert game["plies"] > 0
    assert game["winner"] in {"BLACK", "WHITE", "DRAW"}
    assert timing["iterations"] == 3
    assert timing["median_ns"] > 0

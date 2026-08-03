"""Shared low-overhead timing helper for reproducible project benchmarks."""

from __future__ import annotations

import gc
from collections.abc import Callable
from statistics import median
from time import perf_counter_ns


def measure_operation(
    operation: Callable[[], object], warmup_iterations: int, iterations: int
) -> dict[str, int | float]:
    """Measure an operation after warm-up and return stable timing statistics."""

    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be nonnegative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    for _ in range(warmup_iterations):
        operation()

    samples: list[int] = []
    garbage_collection_enabled = gc.isenabled()
    if garbage_collection_enabled:
        gc.disable()
    try:
        total_start = perf_counter_ns()
        for _ in range(iterations):
            sample_start = perf_counter_ns()
            operation()
            samples.append(perf_counter_ns() - sample_start)
        total_nanoseconds = perf_counter_ns() - total_start
    finally:
        if garbage_collection_enabled:
            gc.enable()

    sorted_samples = sorted(samples)
    percentile_index = max(0, (95 * len(sorted_samples) + 99) // 100 - 1)
    return {
        "iterations": iterations,
        "median_ns": float(median(sorted_samples)),
        "p95_ns": sorted_samples[percentile_index],
        "throughput_ops_per_second": iterations * 1_000_000_000 / total_nanoseconds,
    }

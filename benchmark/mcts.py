"""Reproducible Stage 7 MCTS core and CUDA neural-search benchmark."""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import torch

from benchmark.timing import measure_operation
from config import load_config
from config.schema import MctsConfig, RulesConfig
from game import GameEnvironment, GameState
from mcts import (
    Evaluation,
    MctsSearch,
    SearchMode,
    SearchResult,
    TorchBatchEvaluator,
)
from native import is_available
from network import NetworkSpecification, PolicyValueNetwork


class _UniformEvaluator:
    """Deterministic zero-value evaluator used to isolate the Python tree core."""

    def evaluate(self, states: Sequence[GameState]) -> tuple[Evaluation, ...]:
        return tuple(_uniform_evaluation(state) for state in states)


def run_mcts_benchmark(
    rules: RulesConfig,
    mcts_config: MctsConfig,
    specification: NetworkSpecification,
    *,
    warmup_iterations: int,
    measurement_iterations: int,
    benchmark_simulations: int,
    random_seed: int,
    include_cuda: bool = True,
) -> dict[str, object]:
    """Measure reference/native search and optional CUDA neural search."""

    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be nonnegative")
    if measurement_iterations <= 0 or benchmark_simulations <= 0:
        raise ValueError("Measurement iterations and simulations must be positive")

    benchmark_config = replace(
        mcts_config, evaluation_simulations=benchmark_simulations
    )
    root_state = GameEnvironment(rules).state
    results: dict[str, object] = {
        "python_uniform_core": _measure_search(
            MctsSearch(
                rules,
                replace(benchmark_config, engine="reference"),
                _UniformEvaluator(),
            ),
            root_state,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
        )
    }
    if is_available():
        results["native_uniform_core"] = _measure_search(
            MctsSearch(
                rules,
                replace(benchmark_config, engine="native"),
                _UniformEvaluator(),
            ),
            root_state,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
        )

    cuda_available = torch.cuda.is_available()
    if include_cuda and cuda_available:
        torch.manual_seed(random_seed)
        model = PolicyValueNetwork(specification)
        evaluator = TorchBatchEvaluator(
            model,
            specification,
            device="cuda",
            use_amp=True,
        )
        results["cuda_amp_neural_search"] = _measure_search(
            MctsSearch(rules, benchmark_config, evaluator),
            root_state,
            warmup_iterations=warmup_iterations,
            measurement_iterations=measurement_iterations,
        )

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "platform": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "system": platform.platform(),
            "processor": platform.processor() or "unavailable",
            "cpu_threads": torch.get_num_threads(),
            "cuda_available": cuda_available,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": (
                torch.cuda.get_device_name(0) if cuda_available else "unavailable"
            ),
        },
        "mcts": {
            "engine": benchmark_config.engine,
            "simulations": benchmark_simulations,
            "c_puct": benchmark_config.c_puct,
            "root_noise_enabled": benchmark_config.root_noise_enabled,
            "root_noise_only": benchmark_config.root_noise_only,
            "virtual_loss": benchmark_config.virtual_loss,
            "fpu_reduction": benchmark_config.fpu_reduction,
            "max_inference_batch_size": (benchmark_config.max_inference_batch_size),
        },
        "network": {
            **specification.to_dict(),
            "parameter_count": sum(
                parameter.numel()
                for parameter in PolicyValueNetwork(specification).parameters()
            ),
        },
        "parameters": {
            "warmup_iterations": warmup_iterations,
            "measurement_iterations": measurement_iterations,
            "random_seed": random_seed,
        },
        "results": results,
    }


def _measure_search(
    search: MctsSearch,
    root_state: GameState,
    *,
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict[str, int | float]:
    latest: list[SearchResult] = []

    def operation() -> object:
        result = search.run(root_state, SearchMode.EVALUATION)
        if latest:
            latest[0] = result
        else:
            latest.append(result)
        return result

    timing = measure_operation(operation, warmup_iterations, measurement_iterations)
    if not latest:
        raise RuntimeError("MCTS benchmark did not produce a search result")
    result = latest[0]
    searches_per_second = float(timing["throughput_ops_per_second"])
    average_batch_size = result.inference_positions / result.inference_batches
    return {
        **timing,
        "simulations": result.simulations,
        "simulations_per_second": searches_per_second * result.simulations,
        "inference_batches_per_search": result.inference_batches,
        "inference_positions_per_search": result.inference_positions,
        "average_inference_batch_size": average_batch_size,
        "maximum_inference_batch_size": result.maximum_batch_size,
    }


def _uniform_evaluation(state: GameState) -> Evaluation:
    legal_count = sum(state.legal_mask)
    probability = 1.0 / legal_count
    return Evaluation(
        policy=tuple(probability if legal else 0.0 for legal in state.legal_mask),
        value=0.0,
    )


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        help="Optional TOML profile layered over config/default.toml.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path; defaults to the configured benchmark directory.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Skip CUDA neural-search measurements.",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run configured MCTS benchmarks and persist their JSON report."""

    parsed = _parse_arguments(arguments)
    profile_path: Path | None = parsed.profile
    output_path: Path | None = parsed.output
    config = load_config(profile_path=profile_path)
    specification = NetworkSpecification.from_config(config.rules, config.network)
    report = run_mcts_benchmark(
        config.rules,
        config.mcts,
        specification,
        warmup_iterations=config.benchmark.mcts_warmup_iterations,
        measurement_iterations=config.benchmark.mcts_measurement_iterations,
        benchmark_simulations=config.benchmark.mcts_benchmark_simulations,
        random_seed=config.benchmark.random_seed,
        include_cuda=not parsed.cpu_only,
    )
    destination = output_path or (config.paths.benchmark_directory / "stage7-mcts.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

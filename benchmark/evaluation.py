"""Reproducible Stage 11 paired Arena throughput benchmark."""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import torch

from config import load_config
from config.schema import AppConfig
from mcts import TorchBatchEvaluator
from network import NetworkSpecification, PolicyValueNetwork
from training import ArenaCoordinator, calculate_arena_statistics


def run_evaluation_benchmark(
    config: AppConfig,
    specification: NetworkSpecification,
    *,
    game_count: int,
    simulations: int,
    include_cuda: bool = True,
) -> dict[str, object]:
    """Measure a complete equal-model paired Arena at a reduced search budget."""

    if game_count <= 0 or game_count % 2 != 0:
        raise ValueError("Evaluation benchmark game count must be positive and even")
    if simulations <= 0:
        raise ValueError("Evaluation benchmark simulations must be positive")
    configured = NetworkSpecification.from_config(config.rules, config.network)
    configured.ensure_compatible(specification)

    cuda_available = torch.cuda.is_available()
    use_cuda = include_cuda and cuda_available
    device = torch.device("cuda" if use_cuda else "cpu")
    mcts = replace(config.mcts, evaluation_simulations=simulations)
    evaluation = replace(
        config.evaluation,
        game_count=game_count,
        worker_processes=min(config.evaluation.worker_processes, game_count),
        bootstrap_samples=min(config.evaluation.bootstrap_samples, 1_000),
    )
    benchmark_config = replace(config, mcts=mcts, evaluation=evaluation)

    torch.manual_seed(config.benchmark.random_seed)
    candidate_model = PolicyValueNetwork(specification)
    best_model = PolicyValueNetwork(specification)
    best_model.load_state_dict(candidate_model.state_dict())
    candidate = TorchBatchEvaluator(
        candidate_model,
        specification,
        device=device,
        use_amp=use_cuda and evaluation.amp_enabled,
    )
    best = TorchBatchEvaluator(
        best_model,
        specification,
        device=device,
        use_amp=use_cuda and evaluation.amp_enabled,
    )
    arena = ArenaCoordinator(
        benchmark_config.rules,
        benchmark_config.mcts,
        benchmark_config.evaluation,
        candidate,
        best,
        candidate_identifier="benchmark-candidate",
        best_identifier="benchmark-best",
    ).run()
    statistics = calculate_arena_statistics(arena, benchmark_config.evaluation)
    positions = arena.candidate_inference_positions + arena.best_inference_positions
    inference_batches = arena.candidate_inference_batches + arena.best_inference_batches

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
        "network": {
            **specification.to_dict(),
            "parameter_count_per_model": sum(
                parameter.numel() for parameter in candidate_model.parameters()
            ),
            "models_loaded": 2,
        },
        "parameters": {
            "games": game_count,
            "opening_pairs": game_count // 2,
            "opening_plies": evaluation.opening_plies,
            "simulations_per_searched_move": simulations,
            "worker_processes": arena.worker_processes,
            "inference_capacity": evaluation.inference_max_batch_size,
            "device": str(device),
            "precision": "amp_float16"
            if use_cuda and evaluation.amp_enabled
            else "float32",
            "random_seed": evaluation.random_seed,
        },
        "results": {
            "elapsed_seconds": arena.elapsed_seconds,
            "games_per_second": game_count / arena.elapsed_seconds,
            "simulations": arena.total_simulations,
            "simulations_per_second": arena.total_simulations / arena.elapsed_seconds,
            "inference_batches": inference_batches,
            "inference_positions": positions,
            "average_inference_batch_size": positions / inference_batches,
            "maximum_inference_batch_size": arena.maximum_inference_batch_size,
            "candidate_points": arena.candidate_points,
            "candidate_score_rate": statistics.score_rate,
            "candidate_wins": arena.candidate_wins,
            "best_wins": arena.best_wins,
            "draws": arena.draws,
        },
    }


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
        help="Force CPU float32 inference even when CUDA is available.",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the configured Arena benchmark and persist its JSON report."""

    parsed = _parse_arguments(arguments)
    config = load_config(profile_path=parsed.profile)
    specification = NetworkSpecification.from_config(config.rules, config.network)
    report = run_evaluation_benchmark(
        config,
        specification,
        game_count=config.benchmark.evaluation_benchmark_games,
        simulations=config.benchmark.evaluation_benchmark_simulations,
        include_cuda=not parsed.cpu_only,
    )
    destination = parsed.output or (
        config.paths.benchmark_directory / "stage11-evaluation.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

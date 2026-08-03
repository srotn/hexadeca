"""Reproducible Stage 9 multiprocess self-play throughput benchmark."""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import torch

from config import load_config
from config.schema import MctsConfig, RulesConfig, SelfPlayConfig
from mcts import TorchBatchEvaluator
from network import NetworkSpecification, PolicyValueNetwork
from training import SELF_PLAY_SCHEMA_VERSION, SelfPlayCoordinator


def run_self_play_benchmark(
    rules: RulesConfig,
    mcts_config: MctsConfig,
    self_play_config: SelfPlayConfig,
    specification: NetworkSpecification,
    *,
    games: int,
    simulations: int,
    random_seed: int,
    include_cuda: bool = True,
) -> dict[str, object]:
    """Measure complete self-play using real centralized neural inference."""

    if games <= 0 or simulations <= 0:
        raise ValueError("Self-play benchmark games and simulations must be positive")
    if random_seed < 0:
        raise ValueError("Self-play benchmark seed must be nonnegative")

    cuda_available = torch.cuda.is_available()
    use_cuda = include_cuda and cuda_available
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(random_seed)
    model = PolicyValueNetwork(specification)
    evaluator = TorchBatchEvaluator(
        model,
        specification,
        device=device,
        use_amp=use_cuda,
    )
    benchmark_mcts = replace(mcts_config, training_simulations=simulations)
    coordinator = SelfPlayCoordinator(
        rules,
        benchmark_mcts,
        self_play_config,
        evaluator,
        model_identifier="stage9-benchmark-random-model",
    )

    started_at = perf_counter()
    batch = coordinator.run(game_count=games, master_seed=random_seed)
    elapsed_seconds = perf_counter() - started_at
    positions = len(batch.samples)
    completed_simulations = batch.total_simulations
    average_inference_batch_size = batch.inference_positions / batch.inference_batches

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
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "device": str(device),
            "amp_enabled": use_cuda,
        },
        "self_play": {
            "schema_version": SELF_PLAY_SCHEMA_VERSION,
            "games": games,
            "simulations_per_move": simulations,
            "configured_worker_processes": self_play_config.worker_processes,
            "active_worker_processes": batch.worker_processes,
            "worker_torch_threads": self_play_config.worker_torch_threads,
            "central_inference_capacity": (self_play_config.inference_max_batch_size),
            "mcts_inference_batch_size": benchmark_mcts.max_inference_batch_size,
            "random_seed": random_seed,
        },
        "results": {
            "elapsed_seconds": elapsed_seconds,
            "games_completed": games,
            "positions_generated": positions,
            "simulations_completed": completed_simulations,
            "games_per_second": games / elapsed_seconds,
            "positions_per_second": positions / elapsed_seconds,
            "simulations_per_second": completed_simulations / elapsed_seconds,
            "central_inference_batches": batch.inference_batches,
            "central_inference_positions": batch.inference_positions,
            "central_average_inference_batch_size": average_inference_batch_size,
            "central_maximum_inference_batch_size": (
                batch.maximum_inference_batch_size
            ),
            "worker_inference_requests": sum(
                game.inference_batches for game in batch.games
            ),
            "aggregate_search_seconds": sum(
                game.search_elapsed_seconds for game in batch.games
            ),
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
        help="Force CPU inference even when CUDA is available.",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the configured self-play benchmark and persist its JSON report."""

    parsed = _parse_arguments(arguments)
    profile_path: Path | None = parsed.profile
    output_path: Path | None = parsed.output
    config = load_config(profile_path=profile_path)
    specification = NetworkSpecification.from_config(config.rules, config.network)
    report = run_self_play_benchmark(
        config.rules,
        config.mcts,
        config.self_play,
        specification,
        games=config.benchmark.self_play_benchmark_games,
        simulations=config.benchmark.self_play_benchmark_simulations,
        random_seed=config.benchmark.random_seed,
        include_cuda=not parsed.cpu_only,
    )
    destination = output_path or (
        config.paths.benchmark_directory / "stage9-self-play.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

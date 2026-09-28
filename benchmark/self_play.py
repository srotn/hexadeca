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
from mcts import EvaluatorTiming, TorchBatchEvaluator
from network import NetworkSpecification, PolicyValueNetwork
from training import (
    SELF_PLAY_SCHEMA_VERSION,
    CheckpointManager,
    SelfPlayCoordinator,
    SelfPlayTiming,
)


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
    checkpoint_identifier: str | None = None,
    checkpoint_directory: Path | None = None,
    profile_timing: bool = False,
) -> dict[str, object]:
    """Measure complete self-play using real centralized neural inference."""

    if games <= 0 or simulations <= 0:
        raise ValueError("Self-play benchmark games and simulations must be positive")
    if random_seed < 0:
        raise ValueError("Self-play benchmark seed must be nonnegative")
    if checkpoint_identifier is not None and checkpoint_directory is None:
        raise ValueError("Checkpoint directory is required with checkpoint identifier")

    cuda_available = torch.cuda.is_available()
    use_cuda = include_cuda and cuda_available
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(random_seed)
    model = PolicyValueNetwork(specification)
    resolved_checkpoint_id: str | None = None
    if checkpoint_identifier is not None:
        if checkpoint_directory is None:
            raise ValueError("Checkpoint directory is required with checkpoint id")
        checkpoint_manager = CheckpointManager(checkpoint_directory, specification)
        metadata = checkpoint_manager.load(
            checkpoint_identifier,
            model=model,
            restore_rng=False,
            map_location="cpu",
        )
        resolved_checkpoint_id = metadata.checkpoint_id
    evaluator_timing = EvaluatorTiming() if profile_timing else None
    evaluator = TorchBatchEvaluator(
        model,
        specification,
        device=device,
        use_amp=use_cuda,
        timing=evaluator_timing,
    )
    benchmark_mcts = replace(mcts_config, training_simulations=simulations)
    self_play_timing = SelfPlayTiming() if profile_timing else None
    coordinator = SelfPlayCoordinator(
        rules,
        benchmark_mcts,
        self_play_config,
        evaluator,
        model_identifier=(
            "stage9-benchmark-random-model"
            if resolved_checkpoint_id is None
            else resolved_checkpoint_id
        ),
        timing=self_play_timing,
    )

    started_at = perf_counter()
    batch = coordinator.run(game_count=games, master_seed=random_seed)
    elapsed_seconds = perf_counter() - started_at
    positions = len(batch.samples)
    completed_simulations = batch.total_simulations
    average_inference_batch_size = batch.inference_positions / batch.inference_batches

    report: dict[str, object] = {
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
            "checkpoint_identifier": resolved_checkpoint_id,
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
    if evaluator_timing is not None and self_play_timing is not None:
        report["timing"] = _timing_report(
            evaluator_timing,
            self_play_timing,
            elapsed_seconds,
        )
    return report


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
    parser.add_argument(
        "--games",
        type=_positive_integer,
        help="Override the configured number of complete self-play games.",
    )
    parser.add_argument(
        "--simulations",
        type=_positive_integer,
        help="Override the configured MCTS simulations per move.",
    )
    parser.add_argument(
        "--workers",
        type=_positive_integer,
        help="Override the configured self-play worker process count.",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=_positive_integer,
        help="Override the configured centralized inference batch capacity.",
    )
    parser.add_argument(
        "--checkpoint",
        default="none",
        help=(
            "Checkpoint identifier to load. Use 'none' to benchmark a random "
            "model; defaults to 'none'."
        ),
    )
    parser.add_argument(
        "--profile-timing",
        action="store_true",
        help="Collect detailed worker, IPC, encoding, and CUDA timing data.",
    )
    return parser.parse_args(arguments)


def _positive_integer(raw: str) -> int:
    """Parse one strictly positive command-line integer."""

    try:
        value = int(raw)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Expected a positive integer, received {raw!r}"
        ) from error
    if value <= 0:
        raise argparse.ArgumentTypeError(
            f"Expected a positive integer, received {raw!r}"
        )
    return value


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the configured self-play benchmark and persist its JSON report."""

    parsed = _parse_arguments(arguments)
    profile_path: Path | None = parsed.profile
    output_path: Path | None = parsed.output
    config = load_config(profile_path=profile_path)
    games = (
        config.benchmark.self_play_benchmark_games
        if parsed.games is None
        else parsed.games
    )
    simulations = (
        config.benchmark.self_play_benchmark_simulations
        if parsed.simulations is None
        else parsed.simulations
    )
    self_play_config = replace(
        config.self_play,
        worker_processes=(
            config.self_play.worker_processes
            if parsed.workers is None
            else parsed.workers
        ),
        inference_max_batch_size=(
            config.self_play.inference_max_batch_size
            if parsed.inference_batch_size is None
            else parsed.inference_batch_size
        ),
    )
    specification = NetworkSpecification.from_config(config.rules, config.network)
    checkpoint_identifier = None if parsed.checkpoint == "none" else parsed.checkpoint
    report = run_self_play_benchmark(
        config.rules,
        config.mcts,
        self_play_config,
        specification,
        games=games,
        simulations=simulations,
        random_seed=config.benchmark.random_seed,
        include_cuda=not parsed.cpu_only,
        checkpoint_identifier=checkpoint_identifier,
        checkpoint_directory=config.paths.checkpoint_directory,
        profile_timing=parsed.profile_timing,
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


def _timing_report(
    evaluator: EvaluatorTiming,
    self_play: SelfPlayTiming,
    elapsed_seconds: float,
) -> dict[str, object]:
    """Build one explicit timing report for bottleneck diagnosis."""

    gpu_pipeline_seconds = (
        evaluator.gpu_model_seconds
        + evaluator.gpu_postprocess_seconds
        + evaluator.host_transfer_seconds
    )
    return {
        "measurement_note": (
            "Coordinator and evaluator values are wall-clock totals. Worker values "
            "are aggregate CPU time across processes and may exceed wall time."
        ),
        "wall_elapsed_seconds": elapsed_seconds,
        "evaluator": evaluator.to_dict(),
        "self_play": self_play.to_dict(),
        "derived": {
            "evaluator_gpu_pipeline_seconds": gpu_pipeline_seconds,
            "evaluator_gpu_pipeline_wall_fraction": (
                gpu_pipeline_seconds / elapsed_seconds
            ),
            "central_evaluation_wall_fraction": (
                self_play.coordinator_evaluation_seconds / elapsed_seconds
            ),
            "worker_roundtrip_aggregate_seconds": (
                self_play.worker_mcts.evaluator_roundtrip_seconds
            ),
            "worker_non_inference_mcts_aggregate_seconds": (
                self_play.worker_mcts.root_validation_seconds
                + self_play.worker_mcts.root_setup_seconds
                + self_play.worker_mcts.tree_selection_seconds
                + self_play.worker_mcts.leaf_snapshot_seconds
                + self_play.worker_mcts.tree_commit_seconds
                + self_play.worker_mcts.tree_export_seconds
                + self_play.worker_mcts.root_statistics_seconds
            ),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())

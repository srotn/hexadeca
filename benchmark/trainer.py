"""Reproducible Stage 10 optimizer and CUDA AMP training benchmark."""

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
from game import GameEnvironment, Player
from network import NetworkSpecification, PolicyValueNetwork
from training import ReplayBuffer, ReplaySample, Trainer, TrainingBatchMetrics


def run_trainer_benchmark(
    config: AppConfig,
    specification: NetworkSpecification,
    *,
    warmup_batches: int,
    measurement_batches: int,
    random_seed: int,
    include_cuda: bool = True,
) -> dict[str, object]:
    """Measure real model updates without checkpoint or metric-I/O overhead."""

    if warmup_batches < 0 or measurement_batches <= 0:
        raise ValueError("Trainer benchmark batch counts are invalid")
    if random_seed < 0:
        raise ValueError("Trainer benchmark seed must be nonnegative")
    configured = NetworkSpecification.from_config(config.rules, config.network)
    configured.ensure_compatible(specification)

    cuda_available = torch.cuda.is_available()
    use_cuda = include_cuda and cuda_available
    device = torch.device("cuda" if use_cuda else "cpu")
    benchmark_training = replace(
        config.training,
        batches_per_iteration=warmup_batches + measurement_batches,
        minimum_replay_positions=config.training.batch_size,
        data_loader_workers=0,
        data_loader_pin_memory=use_cuda,
        data_loader_drop_last=True,
        checkpoint_interval_iterations=0,
        tensorboard_enabled=False,
        random_seed=random_seed,
    )
    benchmark_config = replace(config, training=benchmark_training)
    torch.manual_seed(random_seed)
    model = PolicyValueNetwork(specification)
    replay = _benchmark_replay(benchmark_config, specification)
    trainer = Trainer(
        benchmark_config,
        specification,
        model,
        device=device,
    )
    batch_metrics: list[TrainingBatchMetrics] = []
    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    try:
        iteration_metrics = trainer.train_iteration(
            replay, batch_callback=batch_metrics.append
        )
    finally:
        trainer.close()

    measured = batch_metrics[warmup_batches:]
    if len(measured) != measurement_batches:
        raise RuntimeError("Trainer benchmark produced an unexpected batch count")
    elapsed_seconds = sum(batch.elapsed_seconds for batch in measured)
    positions = sum(batch.positions for batch in measured)
    amp_retries = sum(batch.amp_retries for batch in measured)
    result: dict[str, int | float | str | bool] = {
        "device": str(device),
        "precision": "amp_float16" if trainer.amp_enabled else "float32",
        "amp_enabled": trainer.amp_enabled,
        "elapsed_seconds": elapsed_seconds,
        "batches": measurement_batches,
        "positions": positions,
        "batches_per_second": measurement_batches / elapsed_seconds,
        "positions_per_second": positions / elapsed_seconds,
        "ending_learning_rate": iteration_metrics.ending_learning_rate,
        "mean_total_loss": sum(batch.total_loss for batch in measured)
        / measurement_batches,
        "mean_gradient_norm": sum(batch.gradient_norm for batch in measured)
        / measurement_batches,
        "amp_retries": amp_retries,
    }
    if use_cuda:
        result["cuda_peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)

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
        },
        "training": {
            "optimizer": benchmark_training.optimizer,
            "learning_rate": benchmark_training.learning_rate,
            "weight_decay": benchmark_training.weight_decay,
            "adamw_betas": [
                benchmark_training.adamw_beta1,
                benchmark_training.adamw_beta2,
            ],
            "adamw_epsilon": benchmark_training.adamw_epsilon,
            "gradient_clip_norm": benchmark_training.gradient_clip_norm,
            "batch_size": benchmark_training.batch_size,
            "scheduler": benchmark_training.scheduler,
            "scheduler_warmup_steps": benchmark_training.scheduler_warmup_steps,
            "scheduler_decay_steps": benchmark_training.scheduler_decay_steps,
            "scheduler_minimum_learning_rate": (
                benchmark_training.scheduler_minimum_learning_rate
            ),
            "amp_initial_scale": benchmark_training.amp_initial_scale,
            "amp_max_step_retries": benchmark_training.amp_max_step_retries,
        },
        "parameters": {
            "warmup_batches": warmup_batches,
            "measurement_batches": measurement_batches,
            "random_seed": random_seed,
            "timing_scope": "optimizer update excluding data collation and metric I/O",
        },
        "results": result,
    }


def _benchmark_replay(
    config: AppConfig, specification: NetworkSpecification
) -> ReplayBuffer:
    state = GameEnvironment(config.rules).state
    probability = 1.0 / state.action_size
    sample = ReplaySample.create(
        state,
        (probability,) * state.action_size,
        win=1.0 if state.to_play is Player.BLACK else 0.0,
        black_score=100,
        white_score=10,
    )
    buffer = ReplayBuffer(
        config.rules,
        specification,
        config.training.batch_size,
    )
    buffer.extend((sample,) * config.training.batch_size)
    return buffer


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
        help="Force CPU float32 training even when CUDA is available.",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the configured trainer benchmark and persist its JSON report."""

    parsed = _parse_arguments(arguments)
    profile_path: Path | None = parsed.profile
    output_path: Path | None = parsed.output
    config = load_config(profile_path=profile_path)
    specification = NetworkSpecification.from_config(config.rules, config.network)
    report = run_trainer_benchmark(
        config,
        specification,
        warmup_batches=config.benchmark.trainer_benchmark_warmup_batches,
        measurement_batches=config.benchmark.trainer_benchmark_measurement_batches,
        random_seed=config.benchmark.random_seed,
        include_cuda=not parsed.cpu_only,
    )
    destination = output_path or (
        config.paths.benchmark_directory / "stage10-trainer.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

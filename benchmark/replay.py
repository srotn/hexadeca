"""Reproducible Stage 8 replay, collation, and persistence benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import random
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from benchmark.timing import measure_operation
from config import load_config
from config.schema import ReplayConfig, RulesConfig
from game import GameEnvironment, Player
from network import NetworkSpecification
from training import (
    REPLAY_SAMPLE_SCHEMA_VERSION,
    REPLAY_STORE_SCHEMA_VERSION,
    ReplayBuffer,
    ReplayCollator,
    ReplaySample,
    SqliteReplayStore,
)


def run_replay_benchmark(
    rules: RulesConfig,
    specification: NetworkSpecification,
    replay_config: ReplayConfig,
    *,
    positions: int,
    batch_size: int,
    warmup_iterations: int,
    measurement_iterations: int,
    random_seed: int,
    working_directory: Path,
) -> dict[str, object]:
    """Measure all Stage 8 replay paths with exact position accounting."""

    if positions <= 0 or batch_size <= 0 or batch_size > positions:
        raise ValueError("Replay benchmark positions and batch size are invalid")
    if warmup_iterations < 0 or measurement_iterations <= 0:
        raise ValueError("Replay benchmark iteration counts are invalid")

    sample = _initial_sample(rules)
    buffer = ReplayBuffer(rules, specification, positions)
    buffer.extend((sample,) * positions)
    insertion_batch = (sample,) * batch_size
    random_source = random.Random(random_seed)
    collator = ReplayCollator(specification)
    collation_batch = list(buffer.snapshot()[:batch_size])
    store = SqliteReplayStore.from_config(replay_config)
    working_directory.mkdir(parents=True, exist_ok=True)
    database = working_directory / replay_config.persistence_file_name

    append_timing = measure_operation(
        lambda: buffer.extend(insertion_batch),
        warmup_iterations,
        measurement_iterations,
    )
    sample_timing = measure_operation(
        lambda: buffer.sample(batch_size, random_source),
        warmup_iterations,
        measurement_iterations,
    )
    collate_timing = measure_operation(
        lambda: collator(collation_batch),
        warmup_iterations,
        measurement_iterations,
    )
    snapshot = buffer.snapshot()
    save_timing = measure_operation(
        lambda: store.save(snapshot, database),
        warmup_iterations,
        measurement_iterations,
    )
    load_timing = measure_operation(
        lambda: store.load(
            database,
            rules,
            specification,
            capacity_positions=positions,
        ),
        warmup_iterations,
        measurement_iterations,
    )
    database_bytes = database.stat().st_size

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "platform": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "system": platform.platform(),
            "processor": platform.processor() or "unavailable",
            "cpu_threads": torch.get_num_threads(),
        },
        "replay": {
            "sample_schema_version": REPLAY_SAMPLE_SCHEMA_VERSION,
            "store_schema_version": REPLAY_STORE_SCHEMA_VERSION,
            "capacity_positions": positions,
            "retained_positions": len(snapshot),
            "policy_size": specification.policy_size,
            "persistence_chunk_size": replay_config.persistence_chunk_size,
            "database_bytes": database_bytes,
            "database_bytes_per_position": database_bytes / positions,
        },
        "parameters": {
            "batch_size": batch_size,
            "warmup_iterations": warmup_iterations,
            "measurement_iterations": measurement_iterations,
            "random_seed": random_seed,
        },
        "results": {
            "ring_batch_append": _with_position_throughput(append_timing, batch_size),
            "uniform_sample_without_replacement": _with_position_throughput(
                sample_timing, batch_size
            ),
            "feature_target_collation": _with_position_throughput(
                collate_timing, batch_size
            ),
            "sqlite_snapshot_save": _with_position_throughput(save_timing, positions),
            "sqlite_snapshot_load": _with_position_throughput(load_timing, positions),
        },
    }


def _initial_sample(rules: RulesConfig) -> ReplaySample:
    state = GameEnvironment(rules).state
    probability = 1.0 / state.action_size
    return ReplaySample.create(
        state,
        (probability,) * state.action_size,
        win=1.0 if state.to_play is Player.BLACK else 0.0,
        black_score=1,
        white_score=0,
    )


def _with_position_throughput(
    timing: dict[str, int | float], positions_per_operation: int
) -> dict[str, int | float]:
    return {
        **timing,
        "positions_per_operation": positions_per_operation,
        "positions_per_second": (
            float(timing["throughput_ops_per_second"]) * positions_per_operation
        ),
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
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run configured replay benchmarks and persist their JSON report."""

    parsed = _parse_arguments(arguments)
    profile_path: Path | None = parsed.profile
    output_path: Path | None = parsed.output
    config = load_config(profile_path=profile_path)
    specification = NetworkSpecification.from_config(config.rules, config.network)
    destination = output_path or (
        config.paths.benchmark_directory / "stage8-replay.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix="stage8-replay-", dir=destination.parent
    ) as temporary:
        report = run_replay_benchmark(
            config.rules,
            specification,
            config.replay,
            positions=config.benchmark.replay_benchmark_positions,
            batch_size=config.benchmark.replay_benchmark_batch_size,
            warmup_iterations=config.benchmark.replay_warmup_iterations,
            measurement_iterations=config.benchmark.replay_measurement_iterations,
            random_seed=config.benchmark.random_seed,
            working_directory=Path(temporary),
        )
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

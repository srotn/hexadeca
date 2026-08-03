"""Reproducible Stage 5 benchmark for the complete game environment."""

from __future__ import annotations

import argparse
import json
import platform
import random
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from benchmark.timing import measure_operation
from config import load_config
from config.schema import RulesConfig
from game import GameEnvironment, ScoreResult


def run_environment_benchmark(
    rules: RulesConfig,
    *,
    warmup_iterations: int,
    measurement_iterations: int,
    random_seed: int,
) -> dict[str, object]:
    """Benchmark reset, full legal replay, snapshots, and terminal scoring."""

    actions, expected_result = _generate_game(rules, random_seed)
    environment = GameEnvironment(rules)

    def replay_game() -> object:
        environment.reset()
        transition = None
        for action in actions:
            transition = environment.step(action)
        if transition is None or transition.result is None:
            raise RuntimeError("Benchmark action sequence did not terminate")
        return transition.result

    timing = measure_operation(replay_game, warmup_iterations, measurement_iterations)
    actual_result = environment.result()
    if actual_result != expected_result:
        raise RuntimeError("Environment replay produced a different terminal result")

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "platform": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.platform(),
            "processor": platform.processor() or "unavailable",
        },
        "rules": {
            "ruleset_id": rules.ruleset_id,
            "board_size": rules.board_size,
        },
        "game": {
            "plies": len(actions),
            "black_score": expected_result.black_score,
            "white_score": expected_result.white_score,
            "winner": (
                expected_result.winner.name
                if expected_result.winner is not None
                else "DRAW"
            ),
            "zobrist_hash": expected_result.zobrist_hash,
        },
        "parameters": {
            "warmup_iterations": warmup_iterations,
            "measurement_iterations": measurement_iterations,
            "random_seed": random_seed,
        },
        "results": {"full_game_environment": timing},
    }


def _generate_game(
    rules: RulesConfig, random_seed: int
) -> tuple[tuple[int, ...], ScoreResult]:
    environment = GameEnvironment(rules)
    random_source = random.Random(random_seed)
    actions: list[int] = []
    while not environment.terminal():
        action = random_source.choice(environment.legal_moves())
        actions.append(action)
        environment.step(action)
    return tuple(actions), environment.result()


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
    """Run the configured environment benchmark and persist its JSON report."""

    parsed = _parse_arguments(arguments)
    profile_path: Path | None = parsed.profile
    output_path: Path | None = parsed.output
    config = load_config(profile_path=profile_path)
    report = run_environment_benchmark(
        config.rules,
        warmup_iterations=config.benchmark.environment_warmup_iterations,
        measurement_iterations=config.benchmark.environment_measurement_iterations,
        random_seed=config.benchmark.random_seed,
    )
    destination = output_path or (
        config.paths.benchmark_directory / "stage5-environment.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

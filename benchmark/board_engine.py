"""Reproducible Stage 3 benchmarks for the Python board reference engine."""

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
from game import Board


def run_board_benchmarks(
    rules: RulesConfig,
    *,
    warmup_iterations: int,
    measurement_iterations: int,
    position_plies: int,
    random_seed: int,
) -> dict[str, object]:
    """Benchmark construction, legal generation, and apply/undo operations."""

    if warmup_iterations < 0:
        raise ValueError("warmup_iterations must be nonnegative")
    if measurement_iterations <= 0:
        raise ValueError("measurement_iterations must be positive")
    if position_plies < 0:
        raise ValueError("position_plies must be nonnegative")

    board = _build_position(rules, position_plies, random_seed)
    if board.is_terminal:
        board.undo()
    action = board.legal_actions()[0]
    state_before_cycle = (
        board.cells,
        board.to_play,
        board.history,
        board.legal_count,
        board.zobrist_hash,
    )

    def construct_board() -> object:
        return Board(rules)

    def generate_legal_actions() -> object:
        return board.legal_actions()

    def apply_undo_cycle() -> object:
        board.apply(action)
        board.undo()
        return None

    results = {
        "board_construction": measure_operation(
            construct_board, warmup_iterations, measurement_iterations
        ),
        "legal_action_generation": measure_operation(
            generate_legal_actions, warmup_iterations, measurement_iterations
        ),
        "apply_undo_cycle": measure_operation(
            apply_undo_cycle, warmup_iterations, measurement_iterations
        ),
    }

    state_after_cycle = (
        board.cells,
        board.to_play,
        board.history,
        board.legal_count,
        board.zobrist_hash,
    )
    if state_after_cycle != state_before_cycle:
        raise RuntimeError("Apply/undo benchmark mutated its reference position")

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
            "neighborhood_radius": rules.neighborhood_radius,
        },
        "parameters": {
            "warmup_iterations": warmup_iterations,
            "measurement_iterations": measurement_iterations,
            "requested_position_plies": position_plies,
            "actual_position_plies": board.ply,
            "random_seed": random_seed,
        },
        "results": results,
    }


def _build_position(rules: RulesConfig, target_plies: int, seed: int) -> Board:
    board = Board(rules)
    random_source = random.Random(seed)
    while board.ply < target_plies and not board.is_terminal:
        legal_actions = board.legal_actions()
        board.apply(random_source.choice(legal_actions))
    return board


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
    """Run configured board benchmarks and persist their JSON report."""

    parsed = _parse_arguments(arguments)
    profile_path: Path | None = parsed.profile
    output_path: Path | None = parsed.output
    config = load_config(profile_path=profile_path)
    report = run_board_benchmarks(
        config.rules,
        warmup_iterations=config.benchmark.board_warmup_iterations,
        measurement_iterations=config.benchmark.board_measurement_iterations,
        position_plies=config.benchmark.board_position_plies,
        random_seed=config.benchmark.random_seed,
    )
    destination = output_path or (
        config.paths.benchmark_directory / "stage3-board.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Reproducible Stage 4 benchmark for exact terminal scoring."""

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
from game import Board, score_terminal


def run_scoring_benchmark(
    rules: RulesConfig,
    *,
    warmup_iterations: int,
    measurement_iterations: int,
    random_seed: int,
) -> dict[str, object]:
    """Benchmark full terminal scoring on one deterministic completed game."""

    board = _build_terminal_board(rules, random_seed)
    state_before = (board.cells, board.history, board.zobrist_hash)
    expected_result = score_terminal(board)

    def score_position() -> object:
        return score_terminal(board)

    timing = measure_operation(
        score_position, warmup_iterations, measurement_iterations
    )
    if (board.cells, board.history, board.zobrist_hash) != state_before:
        raise RuntimeError("Scoring benchmark mutated its terminal board")

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
            "distance_metric": rules.distance_metric,
            "tie_break": rules.tie_break,
            "majority_award": rules.majority_award,
            "score_occupied_cells": rules.score_occupied_cells,
        },
        "position": {
            "plies": board.ply,
            "zobrist_hash": board.zobrist_hash,
            "black_score": expected_result.black_score,
            "white_score": expected_result.white_score,
        },
        "parameters": {
            "warmup_iterations": warmup_iterations,
            "measurement_iterations": measurement_iterations,
            "random_seed": random_seed,
        },
        "results": {"terminal_scoring": timing},
    }


def _build_terminal_board(rules: RulesConfig, seed: int) -> Board:
    board = Board(rules)
    random_source = random.Random(seed)
    while not board.is_terminal:
        board.apply(random_source.choice(board.legal_actions()))
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
    """Run the configured scoring benchmark and persist its JSON report."""

    parsed = _parse_arguments(arguments)
    profile_path: Path | None = parsed.profile
    output_path: Path | None = parsed.output
    config = load_config(profile_path=profile_path)
    report = run_scoring_benchmark(
        config.rules,
        warmup_iterations=config.benchmark.scoring_warmup_iterations,
        measurement_iterations=config.benchmark.scoring_measurement_iterations,
        random_seed=config.benchmark.random_seed,
    )
    destination = output_path or (
        config.paths.benchmark_directory / "stage4-scoring.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

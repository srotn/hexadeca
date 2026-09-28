"""Paired Arena validation for a single decision-quality search change."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from benchmark.decision_quality.runtime import load_checkpoint_evaluator
from benchmark.decision_quality.schema import write_json_atomic
from config import load_config
from game import Player
from training import ArenaCoordinator, ArenaProgress, calculate_arena_statistics


def main(arguments: Sequence[str] | None = None) -> int:
    """Run paired 100-game before/after Arena and persist all game records."""

    parsed = _parse_arguments(arguments)
    config = load_config(profile_path=parsed.profile)
    device = _resolve_device(parsed.device)
    checkpoint, changed_evaluator = load_checkpoint_evaluator(
        config, parsed.checkpoint, device=device
    )
    _, baseline_evaluator = load_checkpoint_evaluator(
        config, parsed.checkpoint, device=device
    )
    baseline_mcts = config.mcts
    changed_mcts = replace(config.mcts, training_simulations=parsed.changed_simulations)
    # Arena always uses evaluation_simulations. Make the validation budget explicit
    # and keep both players on one identical network; only the selected change differs.
    baseline_mcts = replace(
        baseline_mcts,
        evaluation_simulations=parsed.baseline_simulations,
        root_noise_enabled=False,
    )
    changed_mcts = replace(
        changed_mcts,
        evaluation_simulations=parsed.changed_simulations,
        root_noise_enabled=False,
    )
    evaluation = replace(
        config.evaluation,
        game_count=parsed.games,
        opening_plies=parsed.opening_plies,
        worker_processes=min(parsed.workers, parsed.games),
        random_seed=parsed.seed,
        inference_max_batch_size=max(
            parsed.inference_batch_size,
            baseline_mcts.max_inference_batch_size,
            changed_mcts.max_inference_batch_size,
        ),
    )
    completed_games: set[int] = set()

    def publish_progress(progress: ArenaProgress) -> None:
        if progress.state.terminal and progress.game_index not in completed_games:
            completed_games.add(progress.game_index)
            print(
                f"Arena progress: {len(completed_games)}/{parsed.games} games",
                flush=True,
            )

    arena = ArenaCoordinator(
        config.rules,
        baseline_mcts,
        evaluation,
        changed_evaluator,
        baseline_evaluator,
        candidate_identifier=f"changed-s{parsed.changed_simulations}",
        best_identifier=f"baseline-s{parsed.baseline_simulations}",
        candidate_mcts=changed_mcts,
        best_mcts=baseline_mcts,
    ).run(progress_callback=publish_progress)
    statistics = calculate_arena_statistics(arena, evaluation)
    paired_statistics = _paired_statistics(arena.games)
    report = {
        "schema_version": 1,
        "kind": "decision_quality_arena",
        "generated_at": datetime.now(UTC).isoformat(),
        "checkpoint": checkpoint,
        "change": {
            "baseline_simulations_per_move": parsed.baseline_simulations,
            "changed_simulations_per_move": parsed.changed_simulations,
            "c_puct": config.mcts.c_puct,
            "root_noise_enabled": False,
        },
        "parameters": {
            "games": parsed.games,
            "opening_pairs": parsed.games // 2,
            "opening_plies": parsed.opening_plies,
            "seed": parsed.seed,
            "device": str(device),
            "workers": parsed.workers,
            "inference_batch_size": parsed.inference_batch_size,
        },
        "statistics": {
            **asdict(statistics),
            "candidate_score_rate": statistics.score_rate,
            "total_simulations": arena.total_simulations,
            "mean_candidate_margin": _mean_candidate_margin(arena.games),
            "paired": paired_statistics,
        },
        "throughput": {
            "elapsed_seconds": arena.elapsed_seconds,
            "games_per_second": len(arena.games) / arena.elapsed_seconds,
            "candidate_inference_batches": arena.candidate_inference_batches,
            "candidate_inference_positions": arena.candidate_inference_positions,
            "best_inference_batches": arena.best_inference_batches,
            "best_inference_positions": arena.best_inference_positions,
            "maximum_inference_batch_size": arena.maximum_inference_batch_size,
        },
        "games": [
            {
                "game_index": game.game_index,
                "opening_index": game.opening_index,
                "candidate_player": game.candidate_player.name.lower(),
                "winner": game.winner.name.lower() if game.winner else None,
                "candidate_points": game.candidate_points,
                "black_score": game.black_score,
                "white_score": game.white_score,
                "actions": list(game.actions),
                "opening_actions": list(game.opening_actions),
                "total_simulations": game.total_simulations,
            }
            for game in arena.games
        ],
    }
    write_json_atomic(parsed.output.resolve(), report)
    print(json.dumps(report["statistics"], indent=2, sort_keys=True))
    return 0


def _mean_candidate_margin(games: Sequence[Any]) -> float:
    margins = [
        float(game.black_score - game.white_score)
        * (1.0 if game.candidate_player is Player.BLACK else -1.0)
        for game in games
    ]
    return sum(margins) / len(margins) if margins else 0.0


def _paired_statistics(games: Sequence[Any]) -> dict[str, Any]:
    """Summarize color-swapped opening pairs without treating games as IID."""

    pair_points = [
        float(games[index].candidate_points + games[index + 1].candidate_points)
        for index in range(0, len(games), 2)
    ]
    return {
        "candidate_pair_wins": sum(points > 1.0 for points in pair_points),
        "tied_pairs": sum(points == 1.0 for points in pair_points),
        "candidate_pair_losses": sum(points < 1.0 for points in pair_points),
        "score_rates": [points / 2.0 for points in pair_points],
    }


def _parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--checkpoint", default="iteration-001360")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--opening-plies", type=int, default=2)
    parser.add_argument("--baseline-simulations", type=int, default=800)
    parser.add_argument("--changed-simulations", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parsed = parser.parse_args(arguments)
    if parsed.games <= 0 or parsed.games % 2 or parsed.opening_plies <= 0:
        parser.error("games must be positive and even; opening-plies must be positive")
    if parsed.baseline_simulations <= 0 or parsed.changed_simulations <= 0:
        parser.error("simulation counts must be positive")
    if parsed.workers <= 0 or parsed.inference_batch_size <= 0:
        parser.error("workers and inference-batch-size must be positive")
    return parsed


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


if __name__ == "__main__":
    raise SystemExit(main())

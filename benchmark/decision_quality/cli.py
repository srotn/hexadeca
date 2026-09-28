"""Command-line interface for the fixed-position decision-quality study."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from benchmark.decision_quality.runtime import (
    evaluate_score_heads,
    evaluate_search,
    generate_dataset,
    load_checkpoint_evaluator,
)
from benchmark.decision_quality.schema import write_json_atomic
from config import load_config

DEFAULT_SEED = 20260814


def main(arguments: Sequence[str] | None = None) -> int:
    """Dispatch one diagnostic operation and print its final summary."""

    parsed = _parse_arguments(arguments)
    config = load_config(profile_path=parsed.profile)
    device = _resolve_device(parsed.device)
    checkpoint, evaluator = load_checkpoint_evaluator(
        config, parsed.checkpoint, device=device
    )
    if parsed.command == "generate":
        report = generate_dataset(
            config,
            checkpoint,
            evaluator,
            output_path=parsed.output,
            games=parsed.games,
            positions_per_game=parsed.positions_per_game,
            opening_plies=parsed.opening_plies,
            simulations=parsed.simulations,
            c_puct=parsed.c_puct,
            random_seed=parsed.seed,
            restart=parsed.restart,
        )
        summary: Any = {
            "games": len(report["games"]),
            "positions": len(report["positions"]),
            "output": str(parsed.output.resolve()),
        }
    elif parsed.command == "search":
        report = evaluate_search(
            config,
            checkpoint,
            evaluator,
            dataset_path=parsed.dataset,
            output_path=parsed.output,
            simulations=parsed.simulations,
            c_puct=parsed.c_puct,
            device=device,
            reference_path=parsed.reference,
            save_every=parsed.save_every,
        )
        summary = report["summary"]
    elif parsed.command == "sweep":
        summary = _run_sweep(parsed, config, checkpoint, evaluator, device)
    else:
        report = evaluate_score_heads(
            config,
            checkpoint,
            evaluator,
            dataset_path=parsed.dataset,
            reference_path=parsed.reference,
            output_path=parsed.output,
            candidate_count=parsed.candidates,
            inference_batch_size=parsed.batch_size,
        )
        summary = report["summary"]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _run_sweep(
    parsed: argparse.Namespace,
    config: Any,
    checkpoint: dict[str, str],
    evaluator: Any,
    device: torch.device,
) -> dict[str, Any]:
    output_directory = parsed.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    reference_path = output_directory / _search_file_name(
        parsed.reference_simulations, parsed.reference_c_puct
    )
    reference = evaluate_search(
        config,
        checkpoint,
        evaluator,
        dataset_path=parsed.dataset,
        output_path=reference_path,
        simulations=parsed.reference_simulations,
        c_puct=parsed.reference_c_puct,
        device=device,
        reference_path=None,
        save_every=parsed.save_every,
    )
    configurations = []
    for simulations in parsed.simulations:
        for c_puct in parsed.c_pucts:
            output_path = output_directory / _search_file_name(simulations, c_puct)
            if output_path == reference_path:
                report = reference
            else:
                report = evaluate_search(
                    config,
                    checkpoint,
                    evaluator,
                    dataset_path=parsed.dataset,
                    output_path=output_path,
                    simulations=simulations,
                    c_puct=c_puct,
                    device=device,
                    reference_path=reference_path,
                    save_every=parsed.save_every,
                )
            configurations.append(
                {
                    "simulations": simulations,
                    "c_puct": c_puct,
                    "report": output_path.name,
                    "summary": report["summary"],
                }
            )
    summary = {
        "schema_version": 1,
        "checkpoint": checkpoint,
        "dataset": str(parsed.dataset.resolve()),
        "reference": {
            "simulations": parsed.reference_simulations,
            "c_puct": parsed.reference_c_puct,
            "report": reference_path.name,
        },
        "configurations": configurations,
    }
    write_json_atomic(output_directory / "ablation-summary.json", summary)
    return summary


def _parse_arguments(arguments: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--checkpoint", default="iteration-001360")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Generate the fixed dataset")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--games", type=int, default=50)
    generate.add_argument("--positions-per-game", type=int, default=6)
    generate.add_argument("--opening-plies", type=int, default=2)
    generate.add_argument("--simulations", type=int, default=800)
    generate.add_argument("--c-puct", type=float, default=2.3)
    generate.add_argument("--seed", type=int, default=DEFAULT_SEED)
    generate.add_argument("--restart", action="store_true")

    search = subparsers.add_parser("search", help="Evaluate one search setting")
    _add_search_arguments(search)

    sweep = subparsers.add_parser("sweep", help="Run the sims/c_puct ablation")
    sweep.add_argument("--dataset", type=Path, required=True)
    sweep.add_argument("--output-directory", type=Path, required=True)
    sweep.add_argument(
        "--simulations", type=int, nargs="+", default=[200, 800, 1600, 3200, 6400]
    )
    sweep.add_argument("--c-pucts", type=float, nargs="+", default=[1.5, 2.0, 2.3, 2.8])
    sweep.add_argument("--reference-simulations", type=int, default=6400)
    sweep.add_argument("--reference-c-puct", type=float, default=2.3)
    sweep.add_argument("--save-every", type=int, default=10)

    score = subparsers.add_parser("score", help="Evaluate score-head utility")
    score.add_argument("--dataset", type=Path, required=True)
    score.add_argument("--reference", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--candidates", type=int, default=8)
    score.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args(arguments)


def _add_search_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulations", type=int, required=True)
    parser.add_argument("--c-puct", type=float, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--save-every", type=int, default=10)


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def _search_file_name(simulations: int, c_puct: float) -> str:
    encoded_c_puct = format(c_puct, "g").replace(".", "p")
    return f"search-s{simulations}-c{encoded_c_puct}.json"

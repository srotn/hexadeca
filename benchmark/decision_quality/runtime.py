"""Checkpoint loading, state replay, and resumable offline diagnostics."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from benchmark.decision_quality.metrics import (
    build_score_summary,
    build_search_summary,
    pearson_correlation,
)
from benchmark.decision_quality.schema import (
    DATASET_SCHEMA_VERSION,
    REPORT_SCHEMA_VERSION,
    DecisionQualityDataError,
    file_sha256,
    load_dataset,
    load_search_report,
    write_json_atomic,
)
from config.schema import AppConfig
from game import GameEnvironment, GameState, Player
from mcts import MctsSearch, SearchMode, TorchBatchEvaluator
from network import NetworkSpecification, PolicyValueNetwork
from training import CheckpointManager, generate_openings


def load_checkpoint_evaluator(
    config: AppConfig,
    identifier: str,
    *,
    device: torch.device,
) -> tuple[dict[str, str], TorchBatchEvaluator]:
    """Load one checksum-validated checkpoint for inference-only diagnostics."""

    specification = NetworkSpecification.from_config(config.rules, config.network)
    manager = CheckpointManager(config.paths.checkpoint_directory, specification)
    model = PolicyValueNetwork(specification)
    metadata = manager.load(
        identifier,
        model=model,
        restore_rng=False,
        map_location="cpu",
    )
    evaluator = TorchBatchEvaluator(
        model,
        specification,
        device=device,
        use_amp=device.type == "cuda" and config.evaluation.amp_enabled,
    )
    return {
        "checkpoint_id": metadata.checkpoint_id,
        "state_sha256": metadata.state_sha256,
    }, evaluator


def generate_dataset(
    config: AppConfig,
    checkpoint: Mapping[str, str],
    evaluator: TorchBatchEvaluator,
    *,
    output_path: Path,
    games: int,
    positions_per_game: int,
    opening_plies: int,
    simulations: int,
    c_puct: float,
    random_seed: int,
    restart: bool,
) -> dict[str, Any]:
    """Generate fixed no-noise games and balanced diagnostic positions."""

    if games <= 0 or games > config.rules.action_size:
        raise ValueError("games must be in [1, action_size]")
    if positions_per_game <= 0 or positions_per_game % 3 != 0:
        raise ValueError("positions_per_game must be positive and divisible by three")
    if opening_plies <= 0 or simulations <= 0 or c_puct <= 0 or random_seed < 0:
        raise ValueError("generation parameters are outside their valid range")
    output_path = output_path.resolve()
    progress_path = output_path.with_name(f".{output_path.name}.progress.json")
    identity = {
        "checkpoint": dict(checkpoint),
        "generation": {
            "games": games,
            "opening_pairs": games,
            "opening_plies": opening_plies,
            "positions_per_game": positions_per_game,
            "random_seed": random_seed,
            "simulations": simulations,
            "c_puct": c_puct,
        },
    }
    if output_path.exists() and not restart:
        dataset = load_dataset(output_path)
        if (
            dataset["checkpoint"] != identity["checkpoint"]
            or dataset["generation"] != identity["generation"]
        ):
            raise DecisionQualityDataError(
                "Existing dataset has different settings; choose another output or use --restart"
            )
        return dataset
    records: list[dict[str, Any]] = []
    if progress_path.exists() and not restart:
        raw_progress = _read_progress(progress_path)
        if raw_progress["identity"] != identity:
            raise DecisionQualityDataError(
                "Existing generation progress has different settings; use --restart"
            )
        records = list(raw_progress["games"])
    elif restart:
        progress_path.unlink(missing_ok=True)

    openings = generate_openings(
        config.rules,
        pair_count=games,
        opening_plies=opening_plies,
        random_seed=random_seed,
    )
    mcts = replace(
        config.mcts,
        evaluation_simulations=simulations,
        c_puct=c_puct,
        root_noise_enabled=False,
        endgame_temperature=0.0,
    )
    search = MctsSearch(
        config.rules,
        mcts,
        evaluator,
        materialize_native_tree=False,
    )
    for game_index in range(len(records), games):
        environment = GameEnvironment(config.rules)
        actions: list[int] = []
        opening = openings[game_index]
        for action in opening.actions:
            environment.step(action)
            actions.append(action)
        random_source = random.Random(_derive_seed(random_seed, game_index))
        while not environment.terminal():
            result = search.run(
                environment.state,
                SearchMode.EVALUATION,
                random_source=random_source,
            )
            action = result.action_probabilities.index(max(result.action_probabilities))
            environment.step(action)
            actions.append(action)
        final = environment.result()
        records.append(
            {
                "game_index": game_index,
                "opening_index": opening.index,
                "opening_actions": list(opening.actions),
                "actions": actions,
                "black_score": final.black_score,
                "white_score": final.white_score,
                "winner": _player_name(final.winner),
            }
        )
        write_json_atomic(
            progress_path,
            {"schema_version": 1, "identity": identity, "games": records},
        )
        _print_event(
            "dataset_game_complete",
            game=game_index + 1,
            games=games,
            plies=len(actions),
        )

    positions: list[dict[str, Any]] = []
    for game in records:
        plies = _stratified_plies(
            len(game["actions"]),
            opening_plies=opening_plies,
            positions_per_game=positions_per_game,
        )
        for stratum, ply in plies:
            state = replay_state(config, game["actions"][:ply])
            positions.append(
                {
                    "position_id": f"g{game['game_index']:03d}-p{ply:03d}",
                    "game_index": game["game_index"],
                    "stratum": stratum,
                    "ply": ply,
                    "history": game["actions"][:ply],
                    "to_play": state.to_play.name.lower(),
                    "zobrist_hash": state.zobrist_hash,
                    "final_black_score": game["black_score"],
                    "final_white_score": game["white_score"],
                    "final_winner": game["winner"],
                }
            )
    dataset = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).isoformat(),
        **identity,
        "games": records,
        "positions": positions,
    }
    write_json_atomic(output_path, dataset)
    load_dataset(output_path)
    progress_path.unlink(missing_ok=True)
    return dataset


def evaluate_search(
    config: AppConfig,
    checkpoint: Mapping[str, str],
    evaluator: TorchBatchEvaluator,
    *,
    dataset_path: Path,
    output_path: Path,
    simulations: int,
    c_puct: float,
    device: torch.device,
    reference_path: Path | None,
    save_every: int,
) -> dict[str, Any]:
    """Run or resume one fixed-position MCTS configuration."""

    if simulations <= 0 or c_puct <= 0 or save_every <= 0:
        raise ValueError("Search parameters must be positive")
    dataset_path = dataset_path.resolve()
    output_path = output_path.resolve()
    dataset = load_dataset(dataset_path)
    if dataset["checkpoint"] != dict(checkpoint):
        raise DecisionQualityDataError("Dataset and loaded checkpoint disagree")
    reference = load_search_report(reference_path.resolve()) if reference_path else None
    if reference is not None:
        _validate_report_dataset(reference, dataset_path, checkpoint)
    identity = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "search",
        "dataset_sha256": file_sha256(dataset_path),
        "checkpoint": dict(checkpoint),
        "parameters": {
            "simulations": simulations,
            "c_puct": c_puct,
            "device": str(device),
        },
    }
    created_at = datetime.now(UTC).isoformat()
    records: list[dict[str, Any]] = []
    if output_path.exists():
        existing = load_search_report(output_path)
        for key, value in identity.items():
            if existing[key] != value:
                raise DecisionQualityDataError(
                    "Existing search report has different settings"
                )
        created_at = existing["created_at"]
        records = list(existing["positions"])
    completed_ids = {record["position_id"] for record in records}
    ordered_completed = [
        position["position_id"]
        for position in dataset["positions"]
        if position["position_id"] in completed_ids
    ]
    if ordered_completed != [record["position_id"] for record in records]:
        raise DecisionQualityDataError("Search resume records are out of dataset order")

    mcts = replace(
        config.mcts,
        evaluation_simulations=simulations,
        c_puct=c_puct,
        root_noise_enabled=False,
        endgame_temperature=0.0,
    )
    search = MctsSearch(
        config.rules,
        mcts,
        evaluator,
        materialize_native_tree=False,
    )
    pending_since_save = 0
    for position in dataset["positions"]:
        if position["position_id"] in completed_ids:
            continue
        state = replay_state(config, position["history"])
        if state.zobrist_hash != position["zobrist_hash"]:
            raise DecisionQualityDataError("Persisted position hash is inconsistent")
        started = perf_counter()
        result = search.run(state, SearchMode.EVALUATION)
        elapsed = perf_counter() - started
        selected_action = result.action_probabilities.index(
            max(result.action_probabilities)
        )
        records.append(
            {
                "position_id": position["position_id"],
                "stratum": position["stratum"],
                "ply": position["ply"],
                "to_play": position["to_play"],
                "selected_action": selected_action,
                "root_value": result.root_value,
                "simulations": result.simulations,
                "elapsed_seconds": elapsed,
                "inference_batches": result.inference_batches,
                "inference_positions": result.inference_positions,
                "solved_outcome": result.solved_outcome,
                "action_statistics": _action_statistics(result),
            }
        )
        completed_ids.add(position["position_id"])
        pending_since_save += 1
        if pending_since_save >= save_every:
            _write_search_report(
                output_path,
                identity,
                created_at,
                records,
                dataset,
                reference,
                complete=False,
            )
            pending_since_save = 0
            _print_event(
                "search_progress",
                simulations=simulations,
                c_puct=c_puct,
                positions=len(records),
                total=len(dataset["positions"]),
            )
    report = _write_search_report(
        output_path,
        identity,
        created_at,
        records,
        dataset,
        reference,
        complete=True,
    )
    return report


def evaluate_score_heads(
    config: AppConfig,
    checkpoint: Mapping[str, str],
    evaluator: TorchBatchEvaluator,
    *,
    dataset_path: Path,
    reference_path: Path,
    output_path: Path,
    candidate_count: int,
    inference_batch_size: int,
) -> dict[str, Any]:
    """Measure score calibration and score-based candidate ranking offline."""

    if candidate_count <= 1 or inference_batch_size <= 0:
        raise ValueError("Score diagnostic batch and candidate counts are invalid")
    dataset_path = dataset_path.resolve()
    reference_path = reference_path.resolve()
    dataset = load_dataset(dataset_path)
    reference = load_search_report(reference_path)
    if dataset["checkpoint"] != dict(checkpoint):
        raise DecisionQualityDataError("Dataset and loaded checkpoint disagree")
    _validate_report_dataset(reference, dataset_path, checkpoint)
    reference_by_id = {item["position_id"]: item for item in reference["positions"]}
    root_states = [
        replay_state(config, item["history"]) for item in dataset["positions"]
    ]
    root_predictions = _predict_scores_batched(
        evaluator, root_states, inference_batch_size
    )
    root_records = []
    child_states: list[GameState] = []
    ranking_contexts: list[dict[str, Any]] = []
    for position, state, prediction in zip(
        dataset["positions"], root_states, root_predictions, strict=True
    ):
        final_margin = position["final_black_score"] - position["final_white_score"]
        root_records.append(
            {
                "position_id": position["position_id"],
                "predicted_black_score": prediction.black_score,
                "predicted_white_score": prediction.white_score,
                "predicted_margin": prediction.margin,
                "final_black_score": position["final_black_score"],
                "final_white_score": position["final_white_score"],
                "final_margin": final_margin,
            }
        )
        reference_position = reference_by_id[position["position_id"]]
        candidates = sorted(
            (
                item
                for item in reference_position["action_statistics"]
                if int(item["visits"]) > 0
            ),
            key=lambda item: (-int(item["visits"]), int(item["action"])),
        )[:candidate_count]
        if len(candidates) < 2:
            continue
        start = len(child_states)
        for candidate in candidates:
            environment = GameEnvironment(config.rules)
            for action in position["history"]:
                environment.step(action)
            environment.step(int(candidate["action"]))
            child_states.append(environment.state)
        ranking_contexts.append(
            {
                "position": position,
                "reference": reference_position,
                "candidates": candidates,
                "prediction_start": start,
                "prediction_stop": len(child_states),
                "to_play": state.to_play,
            }
        )
    child_predictions = _predict_scores_batched(
        evaluator, child_states, inference_batch_size
    )
    ranking_records = []
    for context in ranking_contexts:
        candidates = context["candidates"]
        predictions = child_predictions[
            context["prediction_start"] : context["prediction_stop"]
        ]
        utilities = [
            prediction.margin
            if context["to_play"] is Player.BLACK
            else -prediction.margin
            for prediction in predictions
        ]
        q_values = [float(candidate["q_value"]) for candidate in candidates]
        score_index = max(
            range(len(candidates)),
            key=lambda index: (utilities[index], -int(candidates[index]["action"])),
        )
        policy_index = max(
            range(len(candidates)),
            key=lambda index: (
                float(candidates[index]["prior"]),
                -int(candidates[index]["action"]),
            ),
        )
        reference_action = int(context["reference"]["selected_action"])
        reference_q_by_action = {
            int(candidate["action"]): float(candidate["q_value"])
            for candidate in candidates
        }
        best_q = reference_q_by_action.get(reference_action, max(q_values))
        score_action = int(candidates[score_index]["action"])
        policy_action = int(candidates[policy_index]["action"])
        ranking_records.append(
            {
                "position_id": context["position"]["position_id"],
                "candidate_count": len(candidates),
                "reference_action": reference_action,
                "score_selected_action": score_action,
                "policy_selected_action": policy_action,
                "score_q_correlation": pearson_correlation(utilities, q_values),
                "score_head_reference_q_regret": max(
                    0.0, best_q - reference_q_by_action[score_action]
                ),
                "policy_reference_q_regret": max(
                    0.0, best_q - reference_q_by_action[policy_action]
                ),
                "candidates": [
                    {
                        "action": int(candidate["action"]),
                        "reference_q": q_value,
                        "reference_visits": int(candidate["visits"]),
                        "policy_prior": float(candidate["prior"]),
                        "predicted_margin": prediction.margin,
                        "score_utility": utility,
                    }
                    for candidate, q_value, prediction, utility in zip(
                        candidates, q_values, predictions, utilities, strict=True
                    )
                ],
            }
        )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "kind": "score_heads",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": file_sha256(dataset_path),
        "reference_sha256": file_sha256(reference_path),
        "checkpoint": dict(checkpoint),
        "parameters": {
            "candidate_count": candidate_count,
            "inference_batch_size": inference_batch_size,
        },
        "root_positions": root_records,
        "ranking_positions": ranking_records,
        "summary": build_score_summary(root_records, ranking_records),
    }
    write_json_atomic(output_path, report)
    return report


def replay_state(config: AppConfig, actions: Sequence[int]) -> GameState:
    """Reconstruct one immutable state through the official environment."""

    environment = GameEnvironment(config.rules)
    for action in actions:
        environment.step(int(action))
    if environment.terminal():
        raise DecisionQualityDataError("Diagnostic positions must be non-terminal")
    return environment.state


def _action_statistics(result: Any) -> list[dict[str, Any]]:
    statistics = []
    for action, edge in sorted(result.root.children.items()):
        statistics.append(
            {
                "action": action,
                "prior": edge.prior,
                "visits": edge.visit_count,
                "q_value": edge.mean_value,
                "solved_value": edge.solved_value,
            }
        )
    return statistics


def _write_search_report(
    path: Path,
    identity: Mapping[str, Any],
    created_at: str,
    records: list[dict[str, Any]],
    dataset: Mapping[str, Any],
    reference: Mapping[str, Any] | None,
    *,
    complete: bool,
) -> dict[str, Any]:
    summary = None
    if complete:
        reference_positions = reference["positions"] if reference else None
        summary = build_search_summary(dataset, records, reference_positions)
    report = {
        **identity,
        "created_at": created_at,
        "updated_at": datetime.now(UTC).isoformat(),
        "positions": records,
        "summary": summary,
    }
    write_json_atomic(path, report)
    load_search_report(path)
    return report


def _validate_report_dataset(
    report: Mapping[str, Any],
    dataset_path: Path,
    checkpoint: Mapping[str, str],
) -> None:
    if report["dataset_sha256"] != file_sha256(dataset_path):
        raise DecisionQualityDataError("Reference report belongs to another dataset")
    if report["checkpoint"] != dict(checkpoint):
        raise DecisionQualityDataError("Reference report uses another checkpoint")
    if report["summary"] is None:
        raise DecisionQualityDataError("Reference report is incomplete")


def _stratified_plies(
    action_count: int,
    *,
    opening_plies: int,
    positions_per_game: int,
) -> list[tuple[str, int]]:
    available = action_count - opening_plies
    if available < positions_per_game + 1:
        raise DecisionQualityDataError("Generated game is too short to stratify")
    per_stratum = positions_per_game // 3
    strata = ("opening", "middle", "endgame")
    selected: list[tuple[str, int]] = []
    used: set[int] = set()
    for index in range(positions_per_game):
        fraction = (index + 1) / (positions_per_game + 1)
        ply = opening_plies + round(fraction * (available - 1))
        while ply in used and ply < action_count - 1:
            ply += 1
        if ply in used:
            ply = max(
                value
                for value in range(opening_plies, action_count)
                if value not in used
            )
        used.add(ply)
        selected.append((strata[min(2, index // per_stratum)], ply))
    return selected


def _predict_scores_batched(
    evaluator: TorchBatchEvaluator,
    states: Sequence[GameState],
    batch_size: int,
) -> list[Any]:
    predictions = []
    for start in range(0, len(states), batch_size):
        predictions.extend(evaluator.predict_scores(states[start : start + batch_size]))
    return predictions


def _read_progress(path: Path) -> dict[str, Any]:
    import json

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DecisionQualityDataError("Cannot read generation progress") from error
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "identity", "games"}:
        raise DecisionQualityDataError("Generation progress is malformed")
    if (
        raw["schema_version"] != 1
        or not isinstance(raw["identity"], dict)
        or not isinstance(raw["games"], list)
    ):
        raise DecisionQualityDataError("Generation progress values are malformed")
    return raw


def _derive_seed(master_seed: int, index: int) -> int:
    value = (master_seed + (index + 1) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & ((1 << 64) - 1)
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & ((1 << 64) - 1)
    return (value ^ (value >> 31)) & ((1 << 63) - 1)


def _player_name(player: Player | None) -> str | None:
    return player.name.lower() if player is not None else None


def _print_event(event: str, **values: Any) -> None:
    import json

    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)

"""Strict, versioned persistence for decision-quality benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

DATASET_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1


class DecisionQualityDataError(ValueError):
    """Raised when a persisted diagnostic artifact is malformed."""


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably replace one JSON artifact without exposing partial content."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.temporary")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        last_error: PermissionError | None = None
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                last_error = None
                break
            except PermissionError as error:
                last_error = error
                if attempt == 19:
                    raise
                time.sleep(0.05 * (attempt + 1))
        if last_error is not None:
            raise last_error
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_dataset(path: Path) -> dict[str, Any]:
    """Load and validate a fixed diagnostic dataset."""

    raw = _read_object(path, "decision-quality dataset")
    _require_exact_keys(
        raw,
        {
            "schema_version",
            "created_at",
            "checkpoint",
            "generation",
            "games",
            "positions",
        },
        "dataset",
    )
    if _integer(raw, "schema_version", "dataset") != DATASET_SCHEMA_VERSION:
        raise DecisionQualityDataError("Unsupported diagnostic dataset schema")
    checkpoint = _mapping(raw, "checkpoint", "dataset")
    _require_exact_keys(checkpoint, {"checkpoint_id", "state_sha256"}, "checkpoint")
    _nonempty_string(checkpoint, "checkpoint_id", "checkpoint")
    _sha256(checkpoint, "state_sha256", "checkpoint")
    generation = _mapping(raw, "generation", "dataset")
    _require_exact_keys(
        generation,
        {
            "c_puct",
            "games",
            "opening_pairs",
            "opening_plies",
            "positions_per_game",
            "random_seed",
            "simulations",
        },
        "generation",
    )
    game_count = _positive_integer(generation, "games", "generation")
    position_count = _positive_integer(generation, "positions_per_game", "generation")
    _positive_integer(generation, "opening_pairs", "generation")
    _positive_integer(generation, "opening_plies", "generation")
    _nonnegative_integer(generation, "random_seed", "generation")
    _positive_integer(generation, "simulations", "generation")
    _positive_number(generation, "c_puct", "generation")

    games = _list(raw, "games", "dataset")
    positions = _list(raw, "positions", "dataset")
    if len(games) != game_count:
        raise DecisionQualityDataError("Dataset game count is inconsistent")
    if len(positions) != game_count * position_count:
        raise DecisionQualityDataError("Dataset position count is inconsistent")
    game_indices: set[int] = set()
    for game in games:
        mapping = _object(game, "game")
        _require_exact_keys(
            mapping,
            {
                "actions",
                "black_score",
                "game_index",
                "opening_actions",
                "opening_index",
                "white_score",
                "winner",
            },
            "game",
        )
        game_index = _nonnegative_integer(mapping, "game_index", "game")
        game_indices.add(game_index)
        _nonnegative_integer(mapping, "opening_index", "game")
        _integer_list(mapping, "opening_actions", "game", nonempty=True)
        _integer_list(mapping, "actions", "game", nonempty=True)
        _nonnegative_integer(mapping, "black_score", "game")
        _nonnegative_integer(mapping, "white_score", "game")
        if mapping["winner"] not in {"black", "white", None}:
            raise DecisionQualityDataError("Game winner is invalid")
    if game_indices != set(range(game_count)):
        raise DecisionQualityDataError("Dataset game indices must be contiguous")

    position_ids: set[str] = set()
    stratum_counts = {"opening": 0, "middle": 0, "endgame": 0}
    for position in positions:
        mapping = _object(position, "position")
        _require_exact_keys(
            mapping,
            {
                "final_black_score",
                "final_white_score",
                "final_winner",
                "game_index",
                "history",
                "ply",
                "position_id",
                "stratum",
                "to_play",
                "zobrist_hash",
            },
            "position",
        )
        position_id = _nonempty_string(mapping, "position_id", "position")
        if position_id in position_ids:
            raise DecisionQualityDataError("Position IDs must be unique")
        position_ids.add(position_id)
        if _nonnegative_integer(mapping, "game_index", "position") not in game_indices:
            raise DecisionQualityDataError("Position references an unknown game")
        history = _integer_list(mapping, "history", "position", nonempty=True)
        if _nonnegative_integer(mapping, "ply", "position") != len(history):
            raise DecisionQualityDataError("Position ply and history disagree")
        stratum = mapping["stratum"]
        if stratum not in stratum_counts:
            raise DecisionQualityDataError("Position stratum is invalid")
        stratum_counts[stratum] += 1
        if mapping["to_play"] not in {"black", "white"}:
            raise DecisionQualityDataError("Position side to play is invalid")
        _nonnegative_integer(mapping, "zobrist_hash", "position")
        _nonnegative_integer(mapping, "final_black_score", "position")
        _nonnegative_integer(mapping, "final_white_score", "position")
        if mapping["final_winner"] not in {"black", "white", None}:
            raise DecisionQualityDataError("Position final winner is invalid")
    if len(set(stratum_counts.values())) != 1:
        raise DecisionQualityDataError("Dataset strata must be balanced")
    return raw


def load_search_report(path: Path) -> dict[str, Any]:
    """Load the stable identity and completed entries of one search report."""

    raw = _read_object(path, "decision-quality search report")
    _require_exact_keys(
        raw,
        {
            "schema_version",
            "kind",
            "created_at",
            "updated_at",
            "dataset_sha256",
            "checkpoint",
            "parameters",
            "positions",
            "summary",
        },
        "search_report",
    )
    if _integer(raw, "schema_version", "search_report") != REPORT_SCHEMA_VERSION:
        raise DecisionQualityDataError("Unsupported search report schema")
    if raw["kind"] != "search":
        raise DecisionQualityDataError("Artifact is not a search report")
    _sha256(raw, "dataset_sha256", "search_report")
    checkpoint = _mapping(raw, "checkpoint", "search_report")
    _require_exact_keys(checkpoint, {"checkpoint_id", "state_sha256"}, "checkpoint")
    _nonempty_string(checkpoint, "checkpoint_id", "checkpoint")
    _sha256(checkpoint, "state_sha256", "checkpoint")
    parameters = _mapping(raw, "parameters", "search_report")
    _require_exact_keys(
        parameters,
        {"c_puct", "device", "simulations"},
        "parameters",
    )
    _positive_integer(parameters, "simulations", "parameters")
    _positive_number(parameters, "c_puct", "parameters")
    _nonempty_string(parameters, "device", "parameters")
    positions = _list(raw, "positions", "search_report")
    seen: set[str] = set()
    for position in positions:
        mapping = _object(position, "search position")
        position_id = _nonempty_string(mapping, "position_id", "search position")
        if position_id in seen:
            raise DecisionQualityDataError("Search position IDs must be unique")
        seen.add(position_id)
    if raw["summary"] is not None and not isinstance(raw["summary"], dict):
        raise DecisionQualityDataError("Search summary must be an object or null")
    return raw


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one artifact as stored on disk."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, description: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DecisionQualityDataError(f"Cannot read {description}: {path}") from error
    if not isinstance(raw, dict):
        raise DecisionQualityDataError(f"{description} root must be an object")
    return raw


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], path: str
) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise DecisionQualityDataError(
            f"Invalid {path} fields; missing={missing}, unexpected={extra}"
        )


def _mapping(raw: Mapping[str, Any], key: str, path: str) -> dict[str, Any]:
    return _object(raw.get(key), f"{path}.{key}")


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DecisionQualityDataError(f"{path} must be an object")
    return value


def _list(raw: Mapping[str, Any], key: str, path: str) -> list[Any]:
    value = raw.get(key)
    if not isinstance(value, list):
        raise DecisionQualityDataError(f"{path}.{key} must be an array")
    return value


def _integer(raw: Mapping[str, Any], key: str, path: str) -> int:
    value = raw.get(key)
    if type(value) is not int:
        raise DecisionQualityDataError(f"{path}.{key} must be an integer")
    return value


def _positive_integer(raw: Mapping[str, Any], key: str, path: str) -> int:
    value = _integer(raw, key, path)
    if value <= 0:
        raise DecisionQualityDataError(f"{path}.{key} must be positive")
    return value


def _nonnegative_integer(raw: Mapping[str, Any], key: str, path: str) -> int:
    value = _integer(raw, key, path)
    if value < 0:
        raise DecisionQualityDataError(f"{path}.{key} must be nonnegative")
    return value


def _positive_number(raw: Mapping[str, Any], key: str, path: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise DecisionQualityDataError(f"{path}.{key} must be positive")
    return float(value)


def _nonempty_string(raw: Mapping[str, Any], key: str, path: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise DecisionQualityDataError(f"{path}.{key} must be a non-empty string")
    return value


def _sha256(raw: Mapping[str, Any], key: str, path: str) -> str:
    value = _nonempty_string(raw, key, path)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise DecisionQualityDataError(f"{path}.{key} must be a lowercase SHA-256")
    return value


def _integer_list(
    raw: Mapping[str, Any], key: str, path: str, *, nonempty: bool
) -> list[int]:
    value = _list(raw, key, path)
    if nonempty and not value:
        raise DecisionQualityDataError(f"{path}.{key} must not be empty")
    if any(type(item) is not int or item < 0 for item in value):
        raise DecisionQualityDataError(
            f"{path}.{key} must contain nonnegative integers"
        )
    return value

"""Versioned, transactional SQLite persistence for replay snapshots."""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from array import array
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import torch

from config.schema import ReplayConfig, RulesConfig
from game import GameEnvironment, GameState
from network import NetworkSpecification, NetworkSpecificationError
from training.errors import ReplayPersistenceError, ReplayValidationError
from training.replay import ReplayBuffer, ReplaySample, ReplaySnapshot, _decode_actions

REPLAY_STORE_SCHEMA_VERSION = 1
_METADATA_KEYS = frozenset(
    {
        "schema_version",
        "created_at",
        "source_capacity",
        "stored_positions",
        "total_positions_added",
        "network_specification",
    }
)


@dataclass(frozen=True, slots=True)
class ReplayLoadResult:
    """A restored ring plus an explicit capacity-truncation report."""

    buffer: ReplayBuffer
    stored_positions: int
    loaded_positions: int
    dropped_positions: int
    created_at: str


class SqliteReplayStore:
    """Persist immutable replay snapshots using a portable normalized schema."""

    __slots__ = ("_chunk_size",)

    def __init__(self, chunk_size: int) -> None:
        if type(chunk_size) is not int or chunk_size <= 0:
            raise ReplayPersistenceError(
                "Replay persistence chunk size must be positive"
            )
        self._chunk_size = chunk_size

    @classmethod
    def from_config(cls, config: ReplayConfig) -> SqliteReplayStore:
        """Construct the SQLite backend from unified persistence settings."""

        return cls(config.persistence_chunk_size)

    def save(self, snapshot: ReplaySnapshot, destination: Path) -> None:
        """Write a complete database and atomically publish it at destination."""

        destination = destination.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.is_dir():
            raise ReplayPersistenceError(
                f"Replay destination is a directory: {destination}"
            )
        temporary = destination.with_name(
            f".{destination.name}.{uuid4().hex}.temporary"
        )
        try:
            self._write_database(snapshot, temporary)
            with temporary.open("rb+") as replay_file:
                os.fsync(replay_file.fileno())
            os.replace(temporary, destination)
        except ReplayPersistenceError:
            temporary.unlink(missing_ok=True)
            raise
        except (OSError, sqlite3.Error, ValueError) as error:
            temporary.unlink(missing_ok=True)
            raise ReplayPersistenceError(
                f"Failed to persist replay snapshot: {destination}"
            ) from error

    def load(
        self,
        source: Path,
        rules: RulesConfig,
        specification: NetworkSpecification,
        *,
        capacity_positions: int,
    ) -> ReplayLoadResult:
        """Validate and restore newest positions into the requested capacity."""

        if type(capacity_positions) is not int or capacity_positions <= 0:
            raise ReplayPersistenceError("Replay restore capacity must be positive")
        source = source.resolve()
        if not source.is_file():
            raise ReplayPersistenceError(f"Replay database does not exist: {source}")

        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                f"{source.as_uri()}?mode=ro", uri=True, timeout=30.0
            )
            connection.execute("PRAGMA query_only = ON")
            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if integrity != ("ok",):
                raise ReplayPersistenceError("Replay database integrity check failed")
            metadata = self._read_metadata(connection)
            persisted_specification = NetworkSpecification.from_dict(
                _required_object(metadata, "network_specification")
            )
            specification.ensure_compatible(persisted_specification)

            stored_positions = _required_int(metadata, "stored_positions")
            if stored_positions < 0:
                raise ReplayPersistenceError(
                    "Replay stored_positions must be nonnegative"
                )
            source_capacity = _required_int(metadata, "source_capacity")
            total_positions_added = _required_int(metadata, "total_positions_added")
            if source_capacity <= 0 or stored_positions > source_capacity:
                raise ReplayPersistenceError(
                    "Replay source capacity is inconsistent with stored positions"
                )
            if total_positions_added < stored_positions:
                raise ReplayPersistenceError(
                    "Replay lifetime position count is inconsistent"
                )
            actual_positions = int(
                connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
            )
            if actual_positions != stored_positions:
                raise ReplayPersistenceError(
                    "Replay metadata position count does not match samples"
                )

            loaded_positions = min(stored_positions, capacity_positions)
            buffer = ReplayBuffer(
                rules,
                specification,
                capacity_positions,
                initial_total_positions_added=(
                    total_positions_added - loaded_positions
                ),
            )
            rows = connection.execute(
                """
                SELECT history, policy, win, black_score, white_score
                FROM (
                    SELECT sequence, history, policy, win, black_score, white_score
                    FROM samples
                    ORDER BY sequence DESC
                    LIMIT ?
                )
                ORDER BY sequence ASC
                """,
                (loaded_positions,),
            )
            self._restore_rows(rows, buffer, specification)
            if len(buffer) != loaded_positions:
                raise ReplayPersistenceError(
                    "Replay restore count does not match requested rows"
                )
            return ReplayLoadResult(
                buffer=buffer,
                stored_positions=stored_positions,
                loaded_positions=loaded_positions,
                dropped_positions=stored_positions - loaded_positions,
                created_at=_required_string(metadata, "created_at"),
            )
        except ReplayPersistenceError:
            raise
        except (
            sqlite3.Error,
            ValueError,
            TypeError,
            NetworkSpecificationError,
            ReplayValidationError,
        ) as error:
            raise ReplayPersistenceError(
                f"Failed to restore replay database: {source}"
            ) from error
        finally:
            if connection is not None:
                connection.close()

    def _write_database(self, snapshot: ReplaySnapshot, destination: Path) -> None:
        connection = sqlite3.connect(destination, timeout=30.0)
        try:
            connection.execute("PRAGMA journal_mode = DELETE")
            connection.execute("PRAGMA synchronous = FULL")
            connection.execute(f"PRAGMA user_version = {REPLAY_STORE_SCHEMA_VERSION}")
            with connection:
                connection.executescript(
                    """
                    CREATE TABLE metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL
                    ) STRICT;
                    CREATE TABLE samples (
                        sequence INTEGER PRIMARY KEY,
                        history BLOB NOT NULL,
                        policy BLOB NOT NULL,
                        win REAL NOT NULL,
                        black_score INTEGER NOT NULL,
                        white_score INTEGER NOT NULL
                    ) STRICT;
                    """
                )
                metadata: dict[str, object] = {
                    "schema_version": REPLAY_STORE_SCHEMA_VERSION,
                    "created_at": datetime.now(UTC).isoformat(),
                    "source_capacity": snapshot.source_capacity,
                    "stored_positions": len(snapshot),
                    "total_positions_added": snapshot.total_positions_added,
                    "network_specification": snapshot.specification.to_dict(),
                }
                connection.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    (
                        (key, json.dumps(value, sort_keys=True, separators=(",", ":")))
                        for key, value in metadata.items()
                    ),
                )
                row_buffer: list[tuple[int, bytes, bytes, float, int, int]] = []
                for sequence, row in enumerate(snapshot._compact_rows()):
                    history, policy, win, black_score, white_score = row
                    row_buffer.append(
                        (
                            sequence,
                            history,
                            _policy_to_bytes(policy),
                            win,
                            black_score,
                            white_score,
                        )
                    )
                    if len(row_buffer) == self._chunk_size:
                        self._insert_rows(connection, row_buffer)
                        row_buffer.clear()
                if row_buffer:
                    self._insert_rows(connection, row_buffer)

            integrity = connection.execute("PRAGMA integrity_check").fetchone()
            if integrity != ("ok",):
                raise ReplayPersistenceError(
                    "New replay database failed its integrity check"
                )
        finally:
            connection.close()

    @staticmethod
    def _insert_rows(
        connection: sqlite3.Connection,
        rows: list[tuple[int, bytes, bytes, float, int, int]],
    ) -> None:
        connection.executemany(
            """
            INSERT INTO samples(
                sequence, history, policy, win, black_score, white_score
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    @staticmethod
    def _read_metadata(connection: sqlite3.Connection) -> dict[str, object]:
        user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if user_version != REPLAY_STORE_SCHEMA_VERSION:
            raise ReplayPersistenceError(
                f"Unsupported replay store schema version: {user_version}"
            )
        raw_rows = connection.execute("SELECT key, value FROM metadata").fetchall()
        raw_metadata = {str(key): str(value) for key, value in raw_rows}
        if set(raw_metadata) != _METADATA_KEYS:
            raise ReplayPersistenceError(
                "Replay metadata fields are incomplete or unknown"
            )
        metadata: dict[str, object] = {}
        for key, value in raw_metadata.items():
            metadata[key] = json.loads(value)
        if _required_int(metadata, "schema_version") != REPLAY_STORE_SCHEMA_VERSION:
            raise ReplayPersistenceError(
                "Replay metadata schema version is unsupported"
            )
        return metadata

    def _restore_rows(
        self,
        rows: sqlite3.Cursor,
        buffer: ReplayBuffer,
        specification: NetworkSpecification,
    ) -> None:
        pending: list[ReplaySample] = []
        for raw_row in rows:
            history_raw, policy_raw, win_raw, black_raw, white_raw = raw_row
            if not isinstance(history_raw, bytes) or not isinstance(policy_raw, bytes):
                raise ReplayPersistenceError("Replay row blobs have invalid types")
            actions = _decode_actions(history_raw)
            state = _state_from_actions(actions, buffer)
            pending.append(
                ReplaySample.create(
                    state,
                    _policy_from_bytes(policy_raw, specification.policy_size),
                    win=_exact_number(win_raw, "win"),
                    black_score=_exact_database_int(black_raw, "black_score"),
                    white_score=_exact_database_int(white_raw, "white_score"),
                )
            )
            if len(pending) == self._chunk_size:
                buffer.extend(pending)
                pending.clear()
        if pending:
            buffer.extend(pending)


def _state_from_actions(actions: tuple[int, ...], buffer: ReplayBuffer) -> GameState:
    environment = GameEnvironment(buffer.rules)
    for action in actions:
        environment.step(action)
    return environment.state


def _policy_to_bytes(policy: torch.Tensor) -> bytes:
    values = array("f", (float(value) for value in policy.tolist()))
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def _policy_from_bytes(encoded: bytes, policy_size: int) -> tuple[float, ...]:
    if len(encoded) != policy_size * 4:
        raise ReplayPersistenceError("Replay policy blob has an invalid byte length")
    values = array("f")
    values.frombytes(encoded)
    if sys.byteorder != "little":
        values.byteswap()
    return tuple(float(value) for value in values)


def _required_int(metadata: dict[str, object], key: str) -> int:
    value = metadata.get(key)
    if type(value) is not int:
        raise ReplayPersistenceError(f"Replay metadata {key} must be an integer")
    return value


def _required_string(metadata: dict[str, object], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise ReplayPersistenceError(f"Replay metadata {key} must be a string")
    return value


def _required_object(metadata: dict[str, object], key: str) -> dict[str, object]:
    value = metadata.get(key)
    if not isinstance(value, dict) or any(not isinstance(item, str) for item in value):
        raise ReplayPersistenceError(f"Replay metadata {key} must be an object")
    return value


def _exact_number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ReplayPersistenceError(f"Replay row {name} must be numeric")
    return float(value)


def _exact_database_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise ReplayPersistenceError(f"Replay row {name} must be an integer")
    return value

"""Replay ring and versioned SQLite persistence tests."""

from __future__ import annotations

import random
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from config import load_config
from game import GameEnvironment, GameState, Player
from network import NetworkSpecification
from training import (
    ReplayBuffer,
    ReplayPersistenceError,
    ReplaySample,
    ReplayValidationError,
    SqliteReplayStore,
)


def _states(count: int) -> tuple[GameState, ...]:
    environment = GameEnvironment(load_config().rules)
    states: list[GameState] = []
    for _ in range(count):
        states.append(environment.state)
        environment.step(environment.legal_moves()[0])
    return tuple(states)


def _sample(state: GameState, marker: int = 0) -> ReplaySample:
    legal_count = sum(state.legal_mask)
    policy = tuple((1.0 / legal_count if legal else 0.0) for legal in state.legal_mask)
    black_score = 100 + marker
    white_score = 10
    win = 1.0 if state.to_play is Player.BLACK else 0.0
    return ReplaySample.create(
        state,
        policy,
        win=win,
        black_score=black_score,
        white_score=white_score,
    )


def _buffer(capacity: int) -> ReplayBuffer:
    config = load_config()
    specification = NetworkSpecification.from_config(config.rules, config.network)
    return ReplayBuffer(config.rules, specification, capacity)


def test_ring_retains_newest_positions_and_samples_deterministically() -> None:
    """Capacity is position-based and chronological snapshots survive wrapping."""

    buffer = _buffer(3)
    buffer.extend(
        tuple(_sample(state, index) for index, state in enumerate(_states(5)))
    )

    snapshot = buffer.snapshot()
    assert [sample.state.ply for sample in snapshot] == [2, 3, 4]
    assert [sample.black_score for sample in snapshot] == [102, 103, 104]
    assert snapshot.source_capacity == 3
    assert snapshot.total_positions_added == 5
    assert buffer.generation == 1
    assert buffer.total_positions_added == 5

    first = buffer.sample(2, random.Random(19))
    second = buffer.sample(2, random.Random(19))
    assert [sample.state.ply for sample in first] == [
        sample.state.ply for sample in second
    ]
    assert len({sample.state.ply for sample in first}) == 2


def test_mixed_sampling_is_deterministic_and_uses_distinct_positions() -> None:
    """The optional sampler is reproducible without changing ring contents."""

    buffer = _buffer(8)
    buffer.extend(
        tuple(_sample(state, index) for index, state in enumerate(_states(8)))
    )

    first = buffer.sample_mixed(
        6,
        random.Random(23),
        uniform_fraction=0.5,
        recent_fraction=0.25,
        hard_fraction=0.25,
        recent_window_fraction=0.5,
    )
    second = buffer.sample_mixed(
        6,
        random.Random(23),
        uniform_fraction=0.5,
        recent_fraction=0.25,
        hard_fraction=0.25,
        recent_window_fraction=0.5,
    )

    assert [sample.state.ply for sample in first] == [
        sample.state.ply for sample in second
    ]
    assert len({sample.state.ply for sample in first}) == 6
    assert len(buffer) == 8


def test_invalid_multi_sample_insert_is_atomic() -> None:
    """One invalid entry prevents every mutation in its insertion transaction."""

    buffer = _buffer(4)
    state = _states(1)[0]
    valid = _sample(state)
    buffer.append(valid)
    invalid = replace(valid, win=0.0)
    before = buffer.snapshot()

    with pytest.raises(ReplayValidationError, match="win target"):
        buffer.extend((valid, invalid))

    after = buffer.snapshot()
    assert len(after) == len(before) == 1
    assert after[0] == before[0]
    assert buffer.generation == 1
    assert buffer.total_positions_added == 1
    with pytest.raises(ReplayValidationError, match="must not exceed"):
        buffer.sample(2, random.Random(1))


def test_sqlite_round_trip_reports_capacity_truncation(tmp_path: Path) -> None:
    """Restoring into a smaller ring explicitly keeps only newest positions."""

    config = load_config()
    specification = NetworkSpecification.from_config(config.rules, config.network)
    buffer = ReplayBuffer(config.rules, specification, 5)
    buffer.extend(
        tuple(_sample(state, index) for index, state in enumerate(_states(5)))
    )
    store = SqliteReplayStore(chunk_size=2)
    database = tmp_path / "replay.sqlite3"

    store.save(buffer.snapshot(), database)
    result = store.load(database, config.rules, specification, capacity_positions=3)

    assert result.stored_positions == 5
    assert result.loaded_positions == 3
    assert result.dropped_positions == 2
    assert [sample.state.ply for sample in result.buffer.snapshot()] == [2, 3, 4]
    assert result.buffer.total_positions_added == 5
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert connection.execute("PRAGMA user_version").fetchone() == (1,)


def test_sqlite_restore_rejects_corruption_and_incompatible_schema(
    tmp_path: Path,
) -> None:
    """Blob corruption and network drift fail before returning a usable ring."""

    config = load_config()
    specification = NetworkSpecification.from_config(config.rules, config.network)
    buffer = ReplayBuffer(config.rules, specification, 2)
    buffer.append(_sample(_states(1)[0]))
    store = SqliteReplayStore(chunk_size=1)
    database = tmp_path / "replay.sqlite3"
    store.save(buffer.snapshot(), database)

    incompatible = replace(specification, channels=specification.channels + 1)
    with pytest.raises(ReplayPersistenceError, match="restore"):
        store.load(database, config.rules, incompatible, capacity_positions=2)

    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE samples SET policy = ?", (b"broken",))
        connection.commit()
    with pytest.raises(ReplayPersistenceError, match="byte length"):
        store.load(database, config.rules, specification, capacity_positions=2)

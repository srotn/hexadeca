"""Validated position replay samples and a bounded thread-safe ring buffer."""

from __future__ import annotations

import random
import struct
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import isclose, isfinite
from threading import RLock
from typing import overload

import torch

from config.schema import ReplayConfig, RulesConfig
from game import GameEnvironment, GameState, Player
from network import NetworkSpecification
from training.errors import ReplayValidationError

REPLAY_SAMPLE_SCHEMA_VERSION = 1
_MAX_STORABLE_ACTION = (1 << 16) - 1


@dataclass(frozen=True, slots=True)
class ReplaySample:
    """One finalized AlphaZero target bound to one non-terminal position."""

    state: GameState
    policy: tuple[float, ...]
    win: float
    black_score: int
    white_score: int

    @classmethod
    def create(
        cls,
        state: GameState,
        policy: Sequence[float],
        *,
        win: float,
        black_score: int,
        white_score: int,
    ) -> ReplaySample:
        """Normalize a policy sequence into the immutable sample contract."""

        return cls(
            state=state,
            policy=tuple(float(value) for value in policy),
            win=float(win),
            black_score=black_score,
            white_score=white_score,
        )


@dataclass(frozen=True, slots=True)
class _EncodedSample:
    history: bytes
    policy: torch.Tensor
    win: float
    black_score: int
    white_score: int


class ReplaySnapshot(Sequence[ReplaySample]):
    """An immutable chronological copy isolated from subsequent ring writes."""

    __slots__ = (
        "_black_scores",
        "_histories",
        "_policy_targets",
        "_rules",
        "_source_capacity",
        "_specification",
        "_total_positions_added",
        "_white_scores",
        "_win_targets",
    )

    def __init__(
        self,
        rules: RulesConfig,
        specification: NetworkSpecification,
        source_capacity: int,
        total_positions_added: int,
        histories: tuple[bytes, ...],
        policy_targets: torch.Tensor,
        win_targets: torch.Tensor,
        black_scores: torch.Tensor,
        white_scores: torch.Tensor,
    ) -> None:
        self._rules = rules
        self._specification = specification
        self._source_capacity = source_capacity
        self._total_positions_added = total_positions_added
        self._histories = histories
        self._policy_targets = policy_targets
        self._win_targets = win_targets
        self._black_scores = black_scores
        self._white_scores = white_scores

    @property
    def rules(self) -> RulesConfig:
        """Return the immutable rules used to reconstruct every position."""

        return self._rules

    @property
    def specification(self) -> NetworkSpecification:
        """Return the feature and action schema shared by every sample."""

        return self._specification

    @property
    def source_capacity(self) -> int:
        """Return the ring capacity from which this snapshot was copied."""

        return self._source_capacity

    @property
    def total_positions_added(self) -> int:
        """Return the source ring's lifetime accepted-position count."""

        return self._total_positions_added

    def __len__(self) -> int:
        return len(self._histories)

    @overload
    def __getitem__(self, index: int) -> ReplaySample: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[ReplaySample, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> ReplaySample | tuple[ReplaySample, ...]:
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(len(self))))
        resolved_index = _resolve_index(index, len(self))
        state = _reconstruct_state(
            _decode_actions(self._histories[resolved_index]), self._rules
        )
        return ReplaySample(
            state=state,
            policy=tuple(
                float(value) for value in self._policy_targets[resolved_index].tolist()
            ),
            win=float(self._win_targets[resolved_index].item()),
            black_score=int(self._black_scores[resolved_index].item()),
            white_score=int(self._white_scores[resolved_index].item()),
        )

    def _compact_rows(
        self,
    ) -> Iterator[tuple[bytes, torch.Tensor, float, int, int]]:
        """Yield read-only compact rows for the persistence implementation."""

        for index, history in enumerate(self._histories):
            yield (
                history,
                self._policy_targets[index],
                float(self._win_targets[index].item()),
                int(self._black_scores[index].item()),
                int(self._white_scores[index].item()),
            )


class ReplayBuffer:
    """A bounded chronological ring with atomic multi-sample insertion."""

    __slots__ = (
        "_black_scores",
        "_capacity",
        "_generation",
        "_histories",
        "_lock",
        "_next_index",
        "_policy_targets",
        "_rules",
        "_size",
        "_specification",
        "_total_positions_added",
        "_white_scores",
        "_win_targets",
    )

    def __init__(
        self,
        rules: RulesConfig,
        specification: NetworkSpecification,
        capacity_positions: int,
        *,
        initial_total_positions_added: int = 0,
    ) -> None:
        """Allocate a schema-bound ring with columnar target storage."""

        specification.validate()
        if type(capacity_positions) is not int or capacity_positions <= 0:
            raise ReplayValidationError("Replay capacity must be a positive integer")
        if (
            type(initial_total_positions_added) is not int
            or initial_total_positions_added < 0
        ):
            raise ReplayValidationError(
                "Initial replay position count must be nonnegative"
            )
        if specification.ruleset_id != rules.ruleset_id:
            raise ReplayValidationError(
                "Replay ruleset must match the network specification"
            )
        if specification.board_size != rules.board_size:
            raise ReplayValidationError(
                "Replay board size must match the network specification"
            )
        if specification.policy_size > _MAX_STORABLE_ACTION + 1:
            raise ReplayValidationError(
                "Replay action encoding supports at most 65536 actions"
            )

        self._rules = rules
        self._specification = specification
        self._capacity = capacity_positions
        self._histories: list[bytes | None] = [None] * capacity_positions
        self._policy_targets = torch.empty(
            (capacity_positions, specification.policy_size), dtype=torch.float32
        )
        self._win_targets = torch.empty(capacity_positions, dtype=torch.float32)
        self._black_scores = torch.empty(capacity_positions, dtype=torch.int64)
        self._white_scores = torch.empty(capacity_positions, dtype=torch.int64)
        self._size = 0
        self._next_index = 0
        self._generation = 0
        self._total_positions_added = initial_total_positions_added
        self._lock = RLock()

    @classmethod
    def from_config(
        cls,
        rules: RulesConfig,
        specification: NetworkSpecification,
        config: ReplayConfig,
    ) -> ReplayBuffer:
        """Construct a replay ring from the unified capacity configuration."""

        return cls(rules, specification, config.capacity_positions)

    @property
    def capacity(self) -> int:
        """Return the maximum number of retained positions."""

        return self._capacity

    @property
    def rules(self) -> RulesConfig:
        """Return the immutable rules accepted by this buffer."""

        return self._rules

    @property
    def specification(self) -> NetworkSpecification:
        """Return the immutable feature and action schema."""

        return self._specification

    @property
    def generation(self) -> int:
        """Return the count of successful non-empty insertion transactions."""

        with self._lock:
            return self._generation

    @property
    def total_positions_added(self) -> int:
        """Return positions accepted over the lifetime of this ring."""

        with self._lock:
            return self._total_positions_added

    def __len__(self) -> int:
        with self._lock:
            return self._size

    def append(self, sample: ReplaySample) -> None:
        """Validate and append one finalized position."""

        self.extend((sample,))

    def extend(self, samples: Sequence[ReplaySample]) -> None:
        """Validate every sample before atomically committing the full sequence."""

        encoded = tuple(self._validate_and_encode(sample) for sample in samples)
        if not encoded:
            return

        with self._lock:
            for sample in encoded:
                index = self._next_index
                self._histories[index] = sample.history
                self._policy_targets[index].copy_(sample.policy)
                self._win_targets[index] = sample.win
                self._black_scores[index] = sample.black_score
                self._white_scores[index] = sample.white_score
                self._next_index = (index + 1) % self._capacity
                self._size = min(self._size + 1, self._capacity)
            self._generation += 1
            self._total_positions_added += len(encoded)

    def snapshot(self) -> ReplaySnapshot:
        """Copy retained positions in oldest-to-newest order."""

        with self._lock:
            physical_indices = self._chronological_indices()
            histories = tuple(
                self._required_history(index) for index in physical_indices
            )
            index_tensor = torch.tensor(physical_indices, dtype=torch.int64)
            return ReplaySnapshot(
                self._rules,
                self._specification,
                self._capacity,
                self._total_positions_added,
                histories,
                self._policy_targets.index_select(0, index_tensor),
                self._win_targets.index_select(0, index_tensor),
                self._black_scores.index_select(0, index_tensor),
                self._white_scores.index_select(0, index_tensor),
            )

    def sample(
        self, count: int, random_source: random.Random
    ) -> tuple[ReplaySample, ...]:
        """Uniformly sample distinct retained positions using an explicit RNG."""

        if type(count) is not int or count <= 0:
            raise ReplayValidationError("Replay sample count must be positive")
        if not isinstance(random_source, random.Random):
            raise ReplayValidationError("Replay sampling requires random.Random")

        with self._lock:
            if count > self._size:
                raise ReplayValidationError(
                    "Replay sample count must not exceed retained positions"
                )
            chronological = self._chronological_indices()
            selected = random_source.sample(chronological, count)
            histories = tuple(self._required_history(index) for index in selected)
            index_tensor = torch.tensor(selected, dtype=torch.int64)
            snapshot = ReplaySnapshot(
                self._rules,
                self._specification,
                count,
                count,
                histories,
                self._policy_targets.index_select(0, index_tensor),
                self._win_targets.index_select(0, index_tensor),
                self._black_scores.index_select(0, index_tensor),
                self._white_scores.index_select(0, index_tensor),
            )
        return tuple(snapshot)

    def _chronological_indices(self) -> list[int]:
        start = (self._next_index - self._size) % self._capacity
        return [(start + offset) % self._capacity for offset in range(self._size)]

    def _required_history(self, index: int) -> bytes:
        history = self._histories[index]
        if history is None:
            raise RuntimeError("Replay ring invariant violated: missing history")
        return history

    def _validate_and_encode(self, sample: ReplaySample) -> _EncodedSample:
        if not isinstance(sample, ReplaySample):
            raise ReplayValidationError("Replay entries must be ReplaySample values")
        state = sample.state
        if not isinstance(state, GameState):
            raise ReplayValidationError("Replay state must be a GameState")
        if state.ruleset_id != self._rules.ruleset_id:
            raise ReplayValidationError("Replay state ruleset is incompatible")
        if state.board_size != self._specification.board_size:
            raise ReplayValidationError("Replay state board size is incompatible")
        if state.action_size != self._specification.policy_size:
            raise ReplayValidationError("Replay state action size is incompatible")
        if state.terminal:
            raise ReplayValidationError("Replay positions must be non-terminal")
        if not any(state.legal_mask):
            raise ReplayValidationError("Replay positions require a legal action")

        actions = tuple(move.action for move in state.history)
        try:
            reconstructed = _reconstruct_state(actions, self._rules)
        except Exception as error:
            raise ReplayValidationError(
                "Replay state history cannot be reconstructed legally"
            ) from error
        if reconstructed != state:
            raise ReplayValidationError(
                "Replay state does not match its reconstructed history"
            )

        if len(sample.policy) != self._specification.policy_size:
            raise ReplayValidationError("Replay policy length must match action size")
        if any(
            type(value) not in {int, float} or not isfinite(value) or value < 0.0
            for value in sample.policy
        ):
            raise ReplayValidationError("Replay policy must be finite and nonnegative")
        if any(
            value != 0.0
            for value, legal in zip(sample.policy, state.legal_mask, strict=True)
            if not legal
        ):
            raise ReplayValidationError(
                "Replay policy must assign zero mass to illegal actions"
            )
        if not isclose(sum(sample.policy), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ReplayValidationError("Replay policy must sum to one")

        if (
            type(sample.win) not in {int, float}
            or not isfinite(sample.win)
            or sample.win not in {0.0, 0.5, 1.0}
        ):
            raise ReplayValidationError("Replay win target must be 0, 0.5, or 1")
        for name, score in (
            ("black_score", sample.black_score),
            ("white_score", sample.white_score),
        ):
            if (
                type(score) is not int
                or not 0 <= score <= self._specification.score_normalizer
            ):
                raise ReplayValidationError(
                    f"Replay {name} must be an integer within the score normalizer"
                )

        expected_win = _win_target(
            state.to_play, sample.black_score, sample.white_score
        )
        if sample.win != expected_win:
            raise ReplayValidationError(
                "Replay win target disagrees with side to play and final scores"
            )

        policy = torch.tensor(sample.policy, dtype=torch.float32)
        if not torch.isclose(
            policy.sum(), torch.tensor(1.0, dtype=torch.float32), atol=1e-6, rtol=0.0
        ).item():
            raise ReplayValidationError(
                "Replay policy loses normalization in float32 storage"
            )
        return _EncodedSample(
            history=_encode_actions(actions),
            policy=policy,
            win=sample.win,
            black_score=sample.black_score,
            white_score=sample.white_score,
        )


def _win_target(to_play: Player, black_score: int, white_score: int) -> float:
    if black_score == white_score:
        return 0.5
    winner = Player.BLACK if black_score > white_score else Player.WHITE
    return 1.0 if winner is to_play else 0.0


def _reconstruct_state(actions: tuple[int, ...], rules: RulesConfig) -> GameState:
    environment = GameEnvironment(rules)
    for action in actions:
        environment.step(action)
    return environment.state


def _encode_actions(actions: tuple[int, ...]) -> bytes:
    if any(
        type(action) is not int or not 0 <= action <= _MAX_STORABLE_ACTION
        for action in actions
    ):
        raise ReplayValidationError("Replay history contains an unsupported action")
    return struct.pack(f"<{len(actions)}H", *actions)


def _decode_actions(encoded: bytes) -> tuple[int, ...]:
    if len(encoded) % 2 != 0:
        raise ReplayValidationError("Replay history byte length is invalid")
    action_count = len(encoded) // 2
    return struct.unpack(f"<{action_count}H", encoded)


def _resolve_index(index: int, length: int) -> int:
    if type(index) is not int:
        raise TypeError("Replay snapshot indices must be integers or slices")
    resolved = index + length if index < 0 else index
    if not 0 <= resolved < length:
        raise IndexError("Replay snapshot index out of range")
    return resolved

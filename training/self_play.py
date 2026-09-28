"""Deterministic self-play games with spawn workers and centralized inference."""

from __future__ import annotations

import os
import queue
import random
import traceback
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from math import isclose, isfinite
from multiprocessing.process import BaseProcess
from threading import Thread
from time import monotonic, perf_counter
from typing import Any

import torch

from config.schema import MctsConfig, RulesConfig, SelfPlayConfig
from game import GameEnvironment, GameState, Player
from mcts import BatchEvaluator, Evaluation, MctsSearch, MctsTiming, SearchMode
from training.errors import (
    SelfPlayInferenceError,
    SelfPlayValidationError,
    SelfPlayWorkerError,
)
from training.inference_transport import (
    InferenceTransportError,
    combine_state_batches,
    pack_evaluations,
    pack_state_batch,
    unpack_evaluations,
    unpack_state_batch,
)
from training.replay import ReplayBuffer, ReplaySample

SELF_PLAY_SCHEMA_VERSION = 1
_UINT64_MASK = (1 << 64) - 1
_SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15
_SPLITMIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
_SPLITMIX_MULTIPLIER_2 = 0x94D049BB133111EB


@dataclass(frozen=True, slots=True)
class SelfPlayProgress:
    """One completed move suitable for monitoring without owning game logic."""

    game_index: int
    worker_index: int
    process_id: int
    model_identifier: str
    seed: int
    state: GameState
    action: int
    player: Player
    simulations: int
    root_value: float
    inference_batches: int
    inference_positions: int
    maximum_inference_batch_size: int
    search_elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class SelfPlayGame:
    """One finalized game and its all-or-nothing Replay sample sequence."""

    schema_version: int
    game_index: int
    worker_index: int
    process_id: int
    model_identifier: str
    seed: int
    actions: tuple[int, ...]
    samples: tuple[ReplaySample, ...]
    black_score: int
    white_score: int
    winner: Player | None
    total_simulations: int
    inference_batches: int
    inference_positions: int
    maximum_inference_batch_size: int
    search_elapsed_seconds: float

    @property
    def plies(self) -> int:
        """Return the number of placements and generated positions."""

        return len(self.actions)


@dataclass(frozen=True, slots=True)
class SelfPlayBatch:
    """Deterministically ordered games plus centralized inference statistics."""

    schema_version: int
    model_identifier: str
    master_seed: int
    games: tuple[SelfPlayGame, ...]
    worker_processes: int
    inference_batches: int
    inference_positions: int
    maximum_inference_batch_size: int
    elapsed_seconds: float

    @property
    def samples(self) -> tuple[ReplaySample, ...]:
        """Flatten samples in stable game-index and ply order."""

        return tuple(sample for game in self.games for sample in game.samples)

    @property
    def total_simulations(self) -> int:
        """Return the exact sum of completed MCTS simulations."""

        return sum(game.total_simulations for game in self.games)

    def commit(self, replay_buffer: ReplayBuffer) -> int:
        """Atomically append the complete batch and return its position count."""

        samples = self.samples
        replay_buffer.extend(samples)
        return len(samples)


@dataclass(slots=True)
class RemoteEvaluatorTiming:
    """Optional worker-side timing for one remote evaluator lifetime."""

    request_encoding_seconds: float = 0.0
    request_queue_seconds: float = 0.0
    response_wait_seconds: float = 0.0
    response_decoding_seconds: float = 0.0
    requests: int = 0
    positions: int = 0
    gpu_mcts_requests: int = 0

    def add(self, other: RemoteEvaluatorTiming) -> None:
        """Accumulate independent timing from one worker process."""

        self.request_encoding_seconds += other.request_encoding_seconds
        self.request_queue_seconds += other.request_queue_seconds
        self.response_wait_seconds += other.response_wait_seconds
        self.response_decoding_seconds += other.response_decoding_seconds
        self.requests += other.requests
        self.positions += other.positions
        self.gpu_mcts_requests += other.gpu_mcts_requests

    def to_dict(self) -> dict[str, float | int]:
        """Return JSON-compatible aggregate worker remote timings."""

        return {
            "request_encoding_seconds": self.request_encoding_seconds,
            "request_queue_seconds": self.request_queue_seconds,
            "response_wait_seconds": self.response_wait_seconds,
            "response_decoding_seconds": self.response_decoding_seconds,
            "requests": self.requests,
            "positions": self.positions,
            "gpu_mcts_requests": self.gpu_mcts_requests,
        }


@dataclass(slots=True)
class SelfPlayTiming:
    """Optional coordinator and aggregate-worker timing for one batch."""

    coordinator_request_wait_seconds: float = 0.0
    coordinator_batch_aggregation_seconds: float = 0.0
    coordinator_cpu_prepare_wait_seconds: float = 0.0
    coordinator_prepared_batches: int = 0
    coordinator_prepared_positions: int = 0
    coordinator_evaluation_seconds: float = 0.0
    coordinator_response_dispatch_seconds: float = 0.0
    worker_mcts: MctsTiming = field(default_factory=MctsTiming)
    worker_remote_evaluator: RemoteEvaluatorTiming = field(
        default_factory=RemoteEvaluatorTiming
    )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible timing with separate process ownership."""

        return {
            "coordinator": {
                "request_wait_seconds": self.coordinator_request_wait_seconds,
                "batch_aggregation_seconds": self.coordinator_batch_aggregation_seconds,
                "cpu_prepare_wait_seconds": self.coordinator_cpu_prepare_wait_seconds,
                "prepared_batches": self.coordinator_prepared_batches,
                "prepared_positions": self.coordinator_prepared_positions,
                "evaluation_seconds": self.coordinator_evaluation_seconds,
                "response_dispatch_seconds": self.coordinator_response_dispatch_seconds,
            },
            "worker_mcts": self.worker_mcts.to_dict(),
            "worker_remote_evaluator": self.worker_remote_evaluator.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class _PendingTarget:
    state: GameState
    policy: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class _GameJob:
    game_index: int
    seed: int


@dataclass(frozen=True, slots=True)
class _InferenceRequest:
    worker_index: int
    requester_index: int
    request_index: int
    state_count: int
    transport: str
    payload: tuple[GameState, ...] | bytes


@dataclass(frozen=True, slots=True)
class _InferenceResponse:
    request_index: int
    transport: str
    payload: tuple[Evaluation, ...] | bytes | None
    error: str | None


@dataclass(frozen=True, slots=True)
class _GpuMctsRequest:
    worker_index: int
    requester_index: int
    request_index: int
    visit_counts: tuple[int, ...]
    temperature: float
    puct: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class _GpuMctsResponse:
    request_index: int
    probabilities: tuple[float, ...] | None
    error: str | None


@dataclass(frozen=True, slots=True)
class _PreparedInferenceBatch:
    requests: tuple[_InferenceRequest, ...]
    future: Future[Any]
    position_count: int


@dataclass(frozen=True, slots=True)
class _WorkerFailure:
    worker_index: int
    process_id: int
    message: str
    traceback_text: str


@dataclass(frozen=True, slots=True)
class _WorkerDone:
    worker_index: int
    process_id: int
    mcts_timing: MctsTiming | None = None
    remote_evaluator_timing: RemoteEvaluatorTiming | None = None


ProgressCallback = Callable[[SelfPlayProgress], None]
# Called in the coordinator process as soon as a validated game arrives.
# Keeping this hook optional preserves the original all-at-once API while
# allowing an asynchronous Replay producer to consume games incrementally.
CompletedGameCallback = Callable[[SelfPlayGame], None]


class _RemoteGpuMctsOps:
    """Worker proxy for authoritative visit-policy normalization on CUDA."""

    __slots__ = ("_evaluator",)

    def __init__(self, evaluator: _RemoteBatchEvaluator) -> None:
        self._evaluator = evaluator

    def puct_scores(
        self,
        *,
        priors: Sequence[float],
        visit_counts: Sequence[int],
        value_sums: Sequence[float],
        virtual_visit_counts: Sequence[int],
        virtual_value_sums: Sequence[float],
        solved_values: Sequence[float] | None,
        parent_visit_count: int,
        parent_value_sum: float,
        c_puct: float,
        fpu_reduction: float,
    ) -> tuple[float, ...]:
        # Native C++ selection is authoritative.  Sending this diagnostic-only
        # vector over IPC once per move stalls useful leaf inference, so the
        # production proxy intentionally leaves it on the CPU hot path.
        del (
            priors,
            visit_counts,
            value_sums,
            virtual_visit_counts,
            virtual_value_sums,
            solved_values,
            parent_visit_count,
            parent_value_sum,
            c_puct,
            fpu_reduction,
        )
        return ()

    def visit_probabilities(
        self, visit_counts: Sequence[int], temperature: float
    ) -> tuple[float, ...]:
        return self._evaluator._exchange_gpu_mcts(
            tuple(int(value) for value in visit_counts),
            float(temperature),
            None,
        )


class _RemoteBatchEvaluator:
    """Synchronous worker facade over the coordinator's inference queues."""

    __slots__ = (
        "_request_index",
        "_request_queue",
        "_requester_index",
        "_response_queue",
        "_timeout_seconds",
        "_timing",
        "_transport",
        "_worker_index",
        "_gpu_mcts_ops",
    )

    def __init__(
        self,
        worker_index: int,
        requester_index: int,
        request_queue: Any,
        response_queue: Any,
        timeout_seconds: float,
        transport: str,
        timing: RemoteEvaluatorTiming | None = None,
        gpu_mcts_enabled: bool = False,
    ) -> None:
        self._worker_index = worker_index
        self._requester_index = requester_index
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._timeout_seconds = timeout_seconds
        self._transport = transport
        self._timing = timing
        self._request_index = 0
        self._gpu_mcts_ops = _RemoteGpuMctsOps(self) if gpu_mcts_enabled else None

    @property
    def gpu_mcts_ops(self) -> _RemoteGpuMctsOps | None:
        """Expose the central CUDA root-operations proxy to MctsSearch."""

        return self._gpu_mcts_ops

    def evaluate(self, states: Sequence[GameState]) -> tuple[Evaluation, ...]:
        if not states:
            raise SelfPlayInferenceError("Worker cannot request empty inference")
        action_size = states[0].action_size
        if any(state.action_size != action_size for state in states):
            raise SelfPlayInferenceError("Inference states disagree on action size")
        encoding_started = perf_counter() if self._timing is not None else 0.0
        payload: tuple[GameState, ...] | bytes
        if self._transport == "compact":
            try:
                payload = pack_state_batch(states)
            except InferenceTransportError as error:
                raise SelfPlayInferenceError(str(error)) from error
        else:
            payload = tuple(states)
        if self._timing is not None:
            self._timing.request_encoding_seconds += perf_counter() - encoding_started
        return self._exchange(payload, state_count=len(states), action_size=action_size)

    def evaluate_native_packed(
        self, payload: bytes, state_count: int, action_size: int
    ) -> tuple[Evaluation, ...]:
        """Forward a NativeSearchTree leaf packet without Python states."""

        if self._transport != "compact":
            raise SelfPlayInferenceError(
                "Native packed leaves require compact inference transport"
            )
        if not isinstance(payload, bytes) or not payload:
            raise SelfPlayInferenceError("Native compact inference payload is empty")
        if state_count <= 0 or action_size <= 0:
            raise SelfPlayInferenceError(
                "Native compact inference dimensions are invalid"
            )
        return self._exchange(payload, state_count=state_count, action_size=action_size)

    @property
    def supports_native_packed(self) -> bool:
        """Return whether this worker can forward C++ leaf packets directly."""

        return self._transport == "compact"

    def _exchange(
        self,
        payload: tuple[GameState, ...] | bytes,
        *,
        state_count: int,
        action_size: int,
    ) -> tuple[Evaluation, ...]:
        request = _InferenceRequest(
            worker_index=self._worker_index,
            requester_index=self._requester_index,
            request_index=self._request_index,
            state_count=state_count,
            transport=self._transport,
            payload=payload,
        )
        try:
            queue_started = perf_counter() if self._timing is not None else 0.0
            self._request_queue.put(request, timeout=self._timeout_seconds)
            if self._timing is not None:
                self._timing.request_queue_seconds += perf_counter() - queue_started
            response_wait_started = perf_counter() if self._timing is not None else 0.0
            raw_response = self._response_queue.get(timeout=self._timeout_seconds)
            if self._timing is not None:
                self._timing.response_wait_seconds += (
                    perf_counter() - response_wait_started
                )
        except queue.Full as error:
            raise SelfPlayInferenceError(
                "Central inference request queue is full"
            ) from error
        except queue.Empty as error:
            raise SelfPlayInferenceError(
                "Central inference response timed out"
            ) from error
        if not isinstance(raw_response, _InferenceResponse):
            raise SelfPlayInferenceError("Central inference response type is invalid")
        if raw_response.request_index != request.request_index:
            raise SelfPlayInferenceError("Central inference response order is invalid")
        if raw_response.transport != self._transport:
            raise SelfPlayInferenceError(
                "Central inference response transport is invalid"
            )
        self._request_index += 1
        if raw_response.error is not None:
            raise SelfPlayInferenceError(raw_response.error)
        if raw_response.payload is None:
            raise SelfPlayInferenceError("Central inference returned no evaluations")
        if self._transport == "compact":
            if not isinstance(raw_response.payload, bytes):
                raise SelfPlayInferenceError("Compact inference response is invalid")
            try:
                decoding_started = perf_counter() if self._timing is not None else 0.0
                evaluations = unpack_evaluations(
                    raw_response.payload, action_size=action_size
                )
                if self._timing is not None:
                    self._timing.response_decoding_seconds += (
                        perf_counter() - decoding_started
                    )
            except InferenceTransportError as error:
                raise SelfPlayInferenceError(str(error)) from error
        else:
            if not isinstance(raw_response.payload, tuple):
                raise SelfPlayInferenceError("Object inference response is invalid")
            evaluations = raw_response.payload
        if len(evaluations) != state_count:
            raise SelfPlayInferenceError("Central inference response length is invalid")
        if self._timing is not None:
            self._timing.requests += 1
            self._timing.positions += state_count
        return evaluations

    def _exchange_gpu_mcts(
        self,
        visit_counts: tuple[int, ...],
        temperature: float,
        puct: dict[str, Any] | None,
    ) -> tuple[float, ...]:
        request = _GpuMctsRequest(
            worker_index=self._worker_index,
            requester_index=self._requester_index,
            request_index=self._request_index,
            visit_counts=visit_counts,
            temperature=temperature,
            puct=puct,
        )
        try:
            queue_started = perf_counter() if self._timing is not None else 0.0
            self._request_queue.put(request, timeout=self._timeout_seconds)
            if self._timing is not None:
                self._timing.request_queue_seconds += perf_counter() - queue_started
            response_wait_started = perf_counter() if self._timing is not None else 0.0
            raw_response = self._response_queue.get(timeout=self._timeout_seconds)
            if self._timing is not None:
                self._timing.response_wait_seconds += (
                    perf_counter() - response_wait_started
                )
        except queue.Full as error:
            raise SelfPlayInferenceError("GPU MCTS request queue is full") from error
        except queue.Empty as error:
            raise SelfPlayInferenceError("GPU MCTS response timed out") from error
        if not isinstance(raw_response, _GpuMctsResponse):
            raise SelfPlayInferenceError("GPU MCTS response type is invalid")
        if raw_response.request_index != request.request_index:
            raise SelfPlayInferenceError("GPU MCTS response order is invalid")
        self._request_index += 1
        if raw_response.error is not None:
            raise SelfPlayInferenceError(raw_response.error)
        probabilities = raw_response.probabilities
        if probabilities is None or len(probabilities) != len(visit_counts):
            raise SelfPlayInferenceError("GPU MCTS response dimensions changed")
        if self._timing is not None:
            self._timing.gpu_mcts_requests += 1
        return probabilities


class SelfPlayCoordinator:
    """Run isolated search workers while retaining sole evaluator ownership."""

    __slots__ = (
        "_config",
        "_evaluator",
        "_games_per_worker",
        "_gpu_mcts_enabled",
        "_mcts",
        "_model_identifier",
        "_rules",
        "_timing",
    )

    def __init__(
        self,
        rules: RulesConfig,
        mcts: MctsConfig,
        config: SelfPlayConfig,
        evaluator: BatchEvaluator,
        *,
        model_identifier: str,
        games_per_worker: int = 1,
        timing: SelfPlayTiming | None = None,
    ) -> None:
        _validate_model_identifier(model_identifier)
        if type(games_per_worker) is not int or games_per_worker <= 0:
            raise SelfPlayValidationError("Games per worker must be positive")
        if config.inference_max_batch_size < mcts.max_inference_batch_size:
            raise SelfPlayValidationError(
                "Central inference batch must cover one complete MCTS request"
            )
        self._rules = rules
        self._mcts = mcts
        self._config = config
        self._evaluator = evaluator
        self._model_identifier = model_identifier
        self._timing = timing
        self._games_per_worker = games_per_worker
        self._gpu_mcts_enabled = bool(
            getattr(evaluator, "supports_remote_gpu_mcts", False)
        ) and callable(getattr(evaluator, "evaluate_gpu_mcts_root", None))

    def run(
        self,
        *,
        game_count: int | None = None,
        master_seed: int | None = None,
        progress_callback: ProgressCallback | None = None,
        completed_game_callback: CompletedGameCallback | None = None,
        stop_event: Any | None = None,
    ) -> SelfPlayBatch:
        """Generate a complete deterministic batch.

        When supplied, ``completed_game_callback`` receives each game after it
        has passed the same validation used by the returned batch.  It is
        deliberately a streaming notification; the returned value and the
        coordinator's failure/cleanup semantics remain unchanged.
        """

        resolved_game_count = (
            self._config.games_per_iteration if game_count is None else game_count
        )
        resolved_seed = self._config.random_seed if master_seed is None else master_seed
        if type(resolved_game_count) is not int or resolved_game_count <= 0:
            raise SelfPlayValidationError("Self-play game count must be positive")
        if type(resolved_seed) is not int or resolved_seed < 0:
            raise SelfPlayValidationError("Self-play master seed must be nonnegative")

        worker_count = min(self._config.worker_processes, resolved_game_count)
        assignments: list[list[_GameJob]] = [[] for _ in range(worker_count)]
        for game_index in range(resolved_game_count):
            assignments[game_index % worker_count].append(
                _GameJob(
                    game_index=game_index,
                    seed=derive_game_seed(resolved_seed, game_index),
                )
            )

        context = torch.multiprocessing.get_context("spawn")
        requester_count = worker_count * self._games_per_worker
        request_queue = context.Queue(maxsize=requester_count * 2)
        result_queue = context.Queue()
        response_queues = [context.Queue(maxsize=1) for _ in range(requester_count)]
        processes: list[BaseProcess] = []
        try:
            processes = self._start_workers(
                context,
                assignments,
                request_queue,
                response_queues,
                result_queue,
            )
            batch = self._run_event_loop(
                processes,
                request_queue,
                response_queues,
                result_queue,
                resolved_game_count,
                resolved_seed,
                progress_callback,
                completed_game_callback,
                stop_event,
            )
        except BaseException:
            _shutdown_processes(
                processes,
                timeout_seconds=self._config.worker_shutdown_timeout_seconds,
                terminate_first=True,
            )
            _close_queues(
                request_queue,
                result_queue,
                response_queues,
                cancel_pending=True,
            )
            raise

        forced = _shutdown_processes(
            processes,
            timeout_seconds=self._config.worker_shutdown_timeout_seconds,
            terminate_first=False,
        )
        _close_queues(
            request_queue,
            result_queue,
            response_queues,
            cancel_pending=False,
        )
        if forced:
            raise SelfPlayWorkerError(
                "Self-play workers did not exit within the configured timeout"
            )
        return batch

    def _start_workers(
        self,
        context: Any,
        assignments: list[list[_GameJob]],
        request_queue: Any,
        response_queues: list[Any],
        result_queue: Any,
    ) -> list[BaseProcess]:
        processes: list[BaseProcess] = []
        try:
            for worker_index, jobs in enumerate(assignments):
                first_requester = worker_index * self._games_per_worker
                process = context.Process(
                    target=_self_play_worker_main,
                    args=(
                        worker_index,
                        tuple(jobs),
                        self._rules,
                        self._mcts,
                        self._config,
                        self._model_identifier,
                        request_queue,
                        tuple(
                            response_queues[
                                first_requester : first_requester
                                + self._games_per_worker
                            ]
                        ),
                        result_queue,
                        self._games_per_worker,
                        self._timing is not None,
                        self._gpu_mcts_enabled,
                    ),
                    name=f"self-play-{worker_index}",
                )
                process.start()
                processes.append(process)
        except BaseException:
            _shutdown_processes(
                processes,
                timeout_seconds=self._config.worker_shutdown_timeout_seconds,
                terminate_first=True,
            )
            raise
        return processes

    def _run_event_loop(
        self,
        processes: list[BaseProcess],
        request_queue: Any,
        response_queues: list[Any],
        result_queue: Any,
        game_count: int,
        master_seed: int,
        progress_callback: ProgressCallback | None,
        completed_game_callback: CompletedGameCallback | None,
        stop_event: Any | None,
    ) -> SelfPlayBatch:
        started_at = perf_counter()
        active_workers = set(range(len(processes)))
        games: dict[int, SelfPlayGame] = {}
        deferred_requests: deque[_InferenceRequest | _GpuMctsRequest] = deque()
        pending_prepared: deque[_PreparedInferenceBatch] = deque()
        preparation_workers = max(
            0, int(getattr(self._evaluator, "cpu_preparation_workers", 0))
        )
        prefetch_batches = max(1, int(getattr(self._evaluator, "prefetch_batches", 1)))
        prepare_packed = getattr(self._evaluator, "prepare_packed_cpu", None)
        evaluate_prepared = getattr(self._evaluator, "evaluate_prepared_packed", None)
        preparation_enabled = (
            preparation_workers > 0
            and callable(prepare_packed)
            and callable(evaluate_prepared)
        )
        preparation_pool = (
            ThreadPoolExecutor(
                max_workers=preparation_workers,
                thread_name_prefix="self-play-prepare",
            )
            if preparation_enabled
            else None
        )
        inference_batches = 0
        inference_positions = 0
        maximum_batch_size = 0

        try:
            while active_workers or pending_prepared:
                if stop_event is not None and stop_event.is_set():
                    raise SelfPlayWorkerError("Self-play coordinator was stopped")
                for message in _drain_queue(result_queue):
                    self._handle_worker_message(
                        message,
                        active_workers,
                        games,
                        game_count,
                        progress_callback,
                        completed_game_callback,
                    )

                if pending_prepared and (
                    pending_prepared[0].future.done()
                    or len(pending_prepared) >= prefetch_batches
                    or not active_workers
                ):
                    prepared = pending_prepared.popleft()
                    self._serve_prepared_requests(prepared, response_queues)
                    inference_batches += 1
                    inference_positions += prepared.position_count
                    maximum_batch_size = max(
                        maximum_batch_size, prepared.position_count
                    )
                    continue

                request: object | None = (
                    deferred_requests.popleft() if deferred_requests else None
                )
                if request is None:
                    try:
                        request_wait_started = (
                            perf_counter() if self._timing is not None else 0.0
                        )
                        candidate = request_queue.get(
                            timeout=self._config.inference_batch_wait_seconds
                        )
                        if self._timing is not None:
                            self._timing.coordinator_request_wait_seconds += (
                                perf_counter() - request_wait_started
                            )
                    except queue.Empty:
                        if pending_prepared:
                            prepared = pending_prepared.popleft()
                            self._serve_prepared_requests(prepared, response_queues)
                            inference_batches += 1
                            inference_positions += prepared.position_count
                            maximum_batch_size = max(
                                maximum_batch_size, prepared.position_count
                            )
                        else:
                            self._raise_for_unreported_exit(processes, active_workers)
                        continue
                    request = candidate

                if isinstance(request, _GpuMctsRequest):
                    self._serve_gpu_mcts_request(
                        self._validate_gpu_mcts_request(request, active_workers),
                        response_queues,
                    )
                    continue
                request = self._validate_request(request, active_workers)

                requests = [request]
                position_count = request.state_count
                aggregation_started = (
                    perf_counter() if self._timing is not None else 0.0
                )
                deadline = monotonic() + self._config.inference_batch_wait_seconds
                while position_count < self._config.inference_max_batch_size:
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        break
                    try:
                        candidate = request_queue.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if isinstance(candidate, _GpuMctsRequest):
                        deferred_requests.append(
                            self._validate_gpu_mcts_request(candidate, active_workers)
                        )
                        continue
                    next_request = self._validate_request(candidate, active_workers)
                    if position_count + next_request.state_count > (
                        self._config.inference_max_batch_size
                    ):
                        deferred_requests.append(next_request)
                        break
                    requests.append(next_request)
                    position_count += next_request.state_count

                if self._timing is not None:
                    self._timing.coordinator_batch_aggregation_seconds += (
                        perf_counter() - aggregation_started
                    )
                if preparation_pool is not None and request.transport == "compact":
                    payload = self._combine_compact_requests(requests)
                    pending_prepared.append(
                        _PreparedInferenceBatch(
                            tuple(requests),
                            preparation_pool.submit(prepare_packed, payload),
                            position_count,
                        )
                    )
                    continue
                self._serve_requests(requests, response_queues)
                inference_batches += 1
                inference_positions += position_count
                maximum_batch_size = max(maximum_batch_size, position_count)
        finally:
            if preparation_pool is not None:
                preparation_pool.shutdown(wait=True, cancel_futures=True)

        for message in _drain_queue(result_queue):
            self._handle_worker_message(
                message,
                active_workers,
                games,
                game_count,
                progress_callback,
                completed_game_callback,
            )
        if len(games) != game_count:
            missing = sorted(set(range(game_count)) - set(games))
            raise SelfPlayWorkerError(
                f"Self-play batch is missing game results: {missing}"
            )
        ordered_games = tuple(games[index] for index in range(game_count))
        return SelfPlayBatch(
            schema_version=SELF_PLAY_SCHEMA_VERSION,
            model_identifier=self._model_identifier,
            master_seed=master_seed,
            games=ordered_games,
            worker_processes=len(processes),
            inference_batches=inference_batches,
            inference_positions=inference_positions,
            maximum_inference_batch_size=maximum_batch_size,
            elapsed_seconds=perf_counter() - started_at,
        )

    def _serve_requests(
        self,
        requests: list[_InferenceRequest],
        response_queues: list[Any],
    ) -> None:
        transport = requests[0].transport
        if any(request.transport != transport for request in requests):
            raise SelfPlayInferenceError("Mixed inference transports are invalid")
        try:
            evaluation_started = perf_counter() if self._timing is not None else 0.0
            evaluations = self._evaluate_requests(requests, transport)
            if self._timing is not None:
                self._timing.coordinator_evaluation_seconds += (
                    perf_counter() - evaluation_started
                )
        except Exception as error:
            self._raise_inference_failure(requests, response_queues, error)
        self._dispatch_evaluations(requests, evaluations, response_queues)

    def _combine_compact_requests(self, requests: Sequence[_InferenceRequest]) -> bytes:
        """Combine one homogeneous compact group before CPU preparation."""

        if not requests:
            raise SelfPlayInferenceError("Central inference request group is empty")
        if any(request.transport != "compact" for request in requests):
            raise SelfPlayInferenceError(
                "CPU-prepared inference requires compact requests"
            )
        payloads: list[bytes] = []
        for request in requests:
            if not isinstance(request.payload, bytes):
                raise SelfPlayInferenceError("Compact inference request is invalid")
            payloads.append(request.payload)
        try:
            return combine_state_batches(payloads)
        except InferenceTransportError as error:
            raise SelfPlayInferenceError(str(error)) from error

    def _serve_prepared_requests(
        self,
        prepared: _PreparedInferenceBatch,
        response_queues: list[Any],
    ) -> None:
        """Finish a CPU-prepared batch and dispatch its ordered responses."""

        evaluate = getattr(self._evaluator, "evaluate_prepared_packed", None)
        if not callable(evaluate):
            error = SelfPlayInferenceError("Prepared packed inference is unavailable")
            self._raise_inference_failure(prepared.requests, response_queues, error)
        try:
            wait_started = perf_counter() if self._timing is not None else 0.0
            host_batch = prepared.future.result()
            if self._timing is not None:
                self._timing.coordinator_cpu_prepare_wait_seconds += (
                    perf_counter() - wait_started
                )
                self._timing.coordinator_prepared_batches += 1
                self._timing.coordinator_prepared_positions += prepared.position_count
            evaluation_started = perf_counter() if self._timing is not None else 0.0
            evaluations = tuple(evaluate(host_batch))
            if self._timing is not None:
                self._timing.coordinator_evaluation_seconds += (
                    perf_counter() - evaluation_started
                )
        except Exception as error:
            self._raise_inference_failure(prepared.requests, response_queues, error)
        self._dispatch_evaluations(prepared.requests, evaluations, response_queues)

    def _raise_inference_failure(
        self,
        requests: Sequence[_InferenceRequest],
        response_queues: list[Any],
        error: BaseException,
    ) -> None:
        """Notify every blocked requester before failing the coordinator."""

        message = f"Central inference failed: {type(error).__name__}: {error}"
        for request in requests:
            _put_response(
                response_queues[request.requester_index],
                _InferenceResponse(
                    request.request_index, request.transport, None, message
                ),
                self._config.inference_response_timeout_seconds,
            )
        raise SelfPlayInferenceError(message) from error

    def _dispatch_evaluations(
        self,
        requests: Sequence[_InferenceRequest],
        evaluations: Sequence[Evaluation],
        response_queues: list[Any],
    ) -> None:
        """Split a combined evaluation batch back into requester order."""

        expected_count = sum(request.state_count for request in requests)
        if len(evaluations) != expected_count:
            error = SelfPlayInferenceError(
                "Central evaluator output count must match its input batch"
            )
            self._raise_inference_failure(requests, response_queues, error)
        transport = requests[0].transport
        if any(request.transport != transport for request in requests):
            error = SelfPlayInferenceError("Mixed inference transports are invalid")
            self._raise_inference_failure(requests, response_queues, error)

        offset = 0
        dispatch_seconds = 0.0
        for request in requests:
            request_count = request.state_count
            response_evaluations = evaluations[offset : offset + request_count]
            response_payload: tuple[Evaluation, ...] | bytes
            if transport == "compact":
                try:
                    response_payload = pack_evaluations(
                        response_evaluations, action_size=self._rules.action_size
                    )
                except InferenceTransportError as error:
                    raise SelfPlayInferenceError(str(error)) from error
            else:
                response_payload = response_evaluations
            response = _InferenceResponse(
                request_index=request.request_index,
                transport=transport,
                payload=response_payload,
                error=None,
            )
            dispatch_started = perf_counter() if self._timing is not None else 0.0
            _put_response(
                response_queues[request.requester_index],
                response,
                self._config.inference_response_timeout_seconds,
            )
            if self._timing is not None:
                dispatch_seconds += perf_counter() - dispatch_started
            offset += request_count
        if self._timing is not None:
            self._timing.coordinator_response_dispatch_seconds += dispatch_seconds

    def _serve_gpu_mcts_request(
        self,
        request: _GpuMctsRequest,
        response_queues: list[Any],
    ) -> None:
        try:
            evaluator = getattr(self._evaluator, "evaluate_gpu_mcts_root", None)
            if not callable(evaluator):
                raise SelfPlayInferenceError(
                    "Central CUDA MCTS evaluator is unavailable"
                )
            probabilities = tuple(
                evaluator(request.visit_counts, request.temperature, request.puct)
            )
            if len(probabilities) != len(request.visit_counts):
                raise SelfPlayInferenceError("Central CUDA MCTS output size changed")
            response = _GpuMctsResponse(request.request_index, probabilities, None)
        except Exception as error:
            message = f"Central CUDA MCTS failed: {type(error).__name__}: {error}"
            response = _GpuMctsResponse(request.request_index, None, message)
            _put_response(
                response_queues[request.requester_index],
                response,
                self._config.inference_response_timeout_seconds,
            )
            raise SelfPlayInferenceError(message) from error
        _put_response(
            response_queues[request.requester_index],
            response,
            self._config.inference_response_timeout_seconds,
        )

    def _evaluate_requests(
        self, requests: list[_InferenceRequest], transport: str
    ) -> tuple[Evaluation, ...]:
        """Evaluate one homogeneous request group in exact source order."""

        if transport == "object":
            states: list[GameState] = []
            for request in requests:
                if not isinstance(request.payload, tuple):
                    raise SelfPlayInferenceError("Object inference request is invalid")
                states.extend(request.payload)
            return tuple(self._evaluator.evaluate(states))
        if transport != "compact":
            raise SelfPlayInferenceError("Inference request transport is unsupported")

        payload_parts: list[bytes] = []
        for request in requests:
            if not isinstance(request.payload, bytes):
                raise SelfPlayInferenceError("Compact inference request is invalid")
            payload_parts.append(request.payload)
        try:
            payload = combine_state_batches(payload_parts)
        except InferenceTransportError as error:
            raise SelfPlayInferenceError(str(error)) from error
        evaluator = getattr(self._evaluator, "evaluate_packed", None)
        if callable(evaluator):
            return tuple(evaluator(payload))
        try:
            decoded_states = unpack_state_batch(
                payload,
                ruleset_id=self._rules.ruleset_id,
                board_size=self._rules.board_size,
            )
        except InferenceTransportError as error:
            raise SelfPlayInferenceError(str(error)) from error
        return tuple(self._evaluator.evaluate(decoded_states))

    def _validate_request(
        self, candidate: object, active_workers: set[int]
    ) -> _InferenceRequest:
        if not isinstance(candidate, _InferenceRequest):
            raise SelfPlayInferenceError("Worker inference request type is invalid")
        if candidate.worker_index not in active_workers:
            raise SelfPlayInferenceError(
                "Inference request came from an inactive worker"
            )
        if (
            candidate.requester_index < 0
            or candidate.requester_index // self._games_per_worker
            != candidate.worker_index
        ):
            raise SelfPlayInferenceError("Worker inference requester is invalid")
        if candidate.transport not in {"object", "compact"}:
            raise SelfPlayInferenceError("Inference request transport is invalid")
        if candidate.state_count <= 0:
            raise SelfPlayInferenceError("Worker inference request is empty")
        if candidate.state_count > self._config.inference_max_batch_size:
            raise SelfPlayInferenceError(
                "Worker inference request exceeds central batch capacity"
            )
        if candidate.transport == "object" and (
            not isinstance(candidate.payload, tuple)
            or len(candidate.payload) != candidate.state_count
        ):
            raise SelfPlayInferenceError("Object inference request payload is invalid")
        if candidate.transport == "compact" and not isinstance(
            candidate.payload, bytes
        ):
            raise SelfPlayInferenceError("Compact inference request payload is invalid")
        return candidate

    def _validate_gpu_mcts_request(
        self, candidate: object, active_workers: set[int]
    ) -> _GpuMctsRequest:
        if not isinstance(candidate, _GpuMctsRequest):
            raise SelfPlayInferenceError("GPU MCTS request type is invalid")
        if candidate.worker_index not in active_workers:
            raise SelfPlayInferenceError(
                "GPU MCTS request came from an inactive worker"
            )
        if (
            candidate.requester_index < 0
            or candidate.requester_index // self._games_per_worker
            != candidate.worker_index
        ):
            raise SelfPlayInferenceError("GPU MCTS requester is invalid")
        if len(candidate.visit_counts) != self._rules.action_size or any(
            value < 0 for value in candidate.visit_counts
        ):
            raise SelfPlayInferenceError("GPU MCTS visit counts are invalid")
        if not isfinite(candidate.temperature) or candidate.temperature < 0.0:
            raise SelfPlayInferenceError("GPU MCTS temperature is invalid")
        return candidate

    def _handle_worker_message(
        self,
        message: object,
        active_workers: set[int],
        games: dict[int, SelfPlayGame],
        game_count: int,
        progress_callback: ProgressCallback | None,
        completed_game_callback: CompletedGameCallback | None,
    ) -> None:
        if isinstance(message, SelfPlayProgress):
            if progress_callback is not None:
                progress_callback(message)
            return
        if isinstance(message, SelfPlayGame):
            _validate_completed_game(
                message,
                game_count,
                self._model_identifier,
                self._mcts.training_simulations,
                self._rules,
            )
            if message.game_index in games:
                raise SelfPlayWorkerError(
                    f"Duplicate self-play game result: {message.game_index}"
                )
            games[message.game_index] = message
            if completed_game_callback is not None:
                completed_game_callback(message)
            return
        if isinstance(message, _WorkerFailure):
            raise SelfPlayWorkerError(
                f"Worker {message.worker_index} (PID {message.process_id}) failed: "
                f"{message.message}\n{message.traceback_text}"
            )
        if isinstance(message, _WorkerDone):
            if message.worker_index not in active_workers:
                raise SelfPlayWorkerError(
                    f"Duplicate worker completion: {message.worker_index}"
                )
            active_workers.remove(message.worker_index)
            if self._timing is not None:
                if message.mcts_timing is not None:
                    self._timing.worker_mcts.add(message.mcts_timing)
                if message.remote_evaluator_timing is not None:
                    self._timing.worker_remote_evaluator.add(
                        message.remote_evaluator_timing
                    )
            return
        raise SelfPlayWorkerError(
            f"Unknown self-play worker message: {type(message).__name__}"
        )

    @staticmethod
    def _raise_for_unreported_exit(
        processes: list[BaseProcess], active_workers: set[int]
    ) -> None:
        for worker_index in sorted(active_workers):
            process = processes[worker_index]
            if process.exitcode is not None:
                raise SelfPlayWorkerError(
                    f"Worker {worker_index} exited with code {process.exitcode} "
                    "without a completion message"
                )


def play_self_play_game(
    rules: RulesConfig,
    mcts: MctsConfig,
    evaluator: BatchEvaluator,
    *,
    game_index: int,
    worker_index: int,
    seed: int,
    model_identifier: str,
    progress_callback: ProgressCallback | None = None,
    timing: MctsTiming | None = None,
) -> SelfPlayGame:
    """Play one game and finalize every position from its official result."""

    if type(game_index) is not int or game_index < 0:
        raise SelfPlayValidationError("Self-play game index must be nonnegative")
    if type(worker_index) is not int or worker_index < 0:
        raise SelfPlayValidationError("Self-play worker index must be nonnegative")
    if type(seed) is not int or seed < 0:
        raise SelfPlayValidationError("Self-play game seed must be nonnegative")
    _validate_model_identifier(model_identifier)

    random_source = random.Random(seed)
    environment = GameEnvironment(rules)
    search = MctsSearch(
        rules,
        mcts,
        evaluator,
        timing=timing,
        materialize_native_tree=False,
    )
    pending: list[_PendingTarget] = []
    actions: list[int] = []
    total_simulations = 0
    inference_batches = 0
    inference_positions = 0
    maximum_batch_size = 0
    search_elapsed_seconds = 0.0
    process_id = os.getpid()

    while not environment.terminal():
        state = environment.state
        search_started = perf_counter()
        search_result = search.run(
            state, SearchMode.TRAINING, random_source=random_source
        )
        search_elapsed = perf_counter() - search_started
        action = search_result.select_action(random_source)
        pending.append(_PendingTarget(state, search_result.policy_target))
        transition = environment.step(action)
        actions.append(action)
        total_simulations += search_result.simulations
        inference_batches += search_result.inference_batches
        inference_positions += search_result.inference_positions
        maximum_batch_size = max(maximum_batch_size, search_result.maximum_batch_size)
        search_elapsed_seconds += search_elapsed
        if progress_callback is not None:
            progress_callback(
                SelfPlayProgress(
                    game_index=game_index,
                    worker_index=worker_index,
                    process_id=process_id,
                    model_identifier=model_identifier,
                    seed=seed,
                    state=transition.state,
                    action=action,
                    player=transition.move.player,
                    simulations=search_result.simulations,
                    root_value=search_result.root_value,
                    inference_batches=search_result.inference_batches,
                    inference_positions=search_result.inference_positions,
                    maximum_inference_batch_size=search_result.maximum_batch_size,
                    search_elapsed_seconds=search_elapsed,
                )
            )

    result = environment.result()
    samples = tuple(
        ReplaySample.create(
            target.state,
            target.policy,
            win=_outcome_for_player(
                target.state.to_play, result.black_score, result.white_score
            ),
            black_score=result.black_score,
            white_score=result.white_score,
        )
        for target in pending
    )
    game = SelfPlayGame(
        schema_version=SELF_PLAY_SCHEMA_VERSION,
        game_index=game_index,
        worker_index=worker_index,
        process_id=process_id,
        model_identifier=model_identifier,
        seed=seed,
        actions=tuple(actions),
        samples=samples,
        black_score=result.black_score,
        white_score=result.white_score,
        winner=result.winner,
        total_simulations=total_simulations,
        inference_batches=inference_batches,
        inference_positions=inference_positions,
        maximum_inference_batch_size=maximum_batch_size,
        search_elapsed_seconds=search_elapsed_seconds,
    )
    _validate_completed_game(
        game,
        game_index + 1,
        model_identifier,
        mcts.training_simulations,
        rules,
    )
    return game


def derive_game_seed(master_seed: int, game_index: int) -> int:
    """Derive one stable uint64 seed independent of worker scheduling."""

    if type(master_seed) is not int or master_seed < 0:
        raise SelfPlayValidationError("Self-play master seed must be nonnegative")
    if type(game_index) is not int or game_index < 0:
        raise SelfPlayValidationError("Self-play game index must be nonnegative")
    value = (master_seed + (game_index + 1) * _SPLITMIX_INCREMENT) & _UINT64_MASK
    value = (value ^ (value >> 30)) * _SPLITMIX_MULTIPLIER_1 & _UINT64_MASK
    value = (value ^ (value >> 27)) * _SPLITMIX_MULTIPLIER_2 & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK


def _self_play_worker_main(
    worker_index: int,
    jobs: tuple[_GameJob, ...],
    rules: RulesConfig,
    mcts: MctsConfig,
    config: SelfPlayConfig,
    model_identifier: str,
    request_queue: Any,
    response_queues: tuple[Any, ...],
    result_queue: Any,
    games_per_worker: int = 1,
    timing_enabled: bool = False,
    gpu_mcts_enabled: bool = False,
) -> None:
    process_id = os.getpid()
    mcts_timing = MctsTiming() if timing_enabled else None
    remote_evaluator_timing = RemoteEvaluatorTiming() if timing_enabled else None
    try:
        torch.set_num_threads(config.worker_torch_threads)
        lane_count = min(games_per_worker, len(jobs))
        lane_jobs: list[list[_GameJob]] = [[] for _ in range(lane_count)]
        for job_index, job in enumerate(jobs):
            lane_jobs[job_index % lane_count].append(job)
        lane_mcts = [
            MctsTiming() if timing_enabled else None for _ in range(lane_count)
        ]
        lane_remote = [
            RemoteEvaluatorTiming() if timing_enabled else None
            for _ in range(lane_count)
        ]

        def run_lane(lane_index: int) -> None:
            try:
                evaluator = _RemoteBatchEvaluator(
                    worker_index,
                    worker_index * games_per_worker + lane_index,
                    request_queue,
                    response_queues[lane_index],
                    config.inference_response_timeout_seconds,
                    config.inference_transport,
                    lane_remote[lane_index],
                    gpu_mcts_enabled,
                )

                def publish_progress(progress: SelfPlayProgress) -> None:
                    result_queue.put(progress)

                for job in lane_jobs[lane_index]:
                    game = play_self_play_game(
                        rules,
                        mcts,
                        evaluator,
                        game_index=job.game_index,
                        worker_index=worker_index,
                        seed=job.seed,
                        model_identifier=model_identifier,
                        progress_callback=publish_progress,
                        timing=lane_mcts[lane_index],
                    )
                    result_queue.put(game)
            except BaseException as error:
                result_queue.put(
                    _WorkerFailure(
                        worker_index=worker_index,
                        process_id=process_id,
                        message=f"{type(error).__name__}: {error}",
                        traceback_text=traceback.format_exc(),
                    )
                )

        lanes = [
            Thread(
                target=run_lane,
                args=(lane_index,),
                name=f"self-play-{worker_index}-lane-{lane_index}",
            )
            for lane_index in range(lane_count)
        ]
        for lane in lanes:
            lane.start()
        for lane in lanes:
            lane.join()
        if timing_enabled:
            for timing in lane_mcts:
                if timing is not None and mcts_timing is not None:
                    mcts_timing.add(timing)
            for timing in lane_remote:
                if timing is not None and remote_evaluator_timing is not None:
                    remote_evaluator_timing.add(timing)
    except BaseException as error:
        result_queue.put(
            _WorkerFailure(
                worker_index=worker_index,
                process_id=process_id,
                message=f"{type(error).__name__}: {error}",
                traceback_text=traceback.format_exc(),
            )
        )
    finally:
        result_queue.put(
            _WorkerDone(
                worker_index,
                process_id,
                mcts_timing=mcts_timing,
                remote_evaluator_timing=remote_evaluator_timing,
            )
        )
        result_queue.close()
        result_queue.join_thread()
        request_queue.close()
        request_queue.join_thread()


def _validate_completed_game(
    game: SelfPlayGame,
    game_count: int,
    model_identifier: str,
    simulations_per_move: int,
    rules: RulesConfig,
) -> None:
    if game.schema_version != SELF_PLAY_SCHEMA_VERSION:
        raise SelfPlayValidationError("Self-play game schema version is unsupported")
    if not 0 <= game.game_index < game_count:
        raise SelfPlayValidationError("Self-play game index is outside the batch")
    if game.model_identifier != model_identifier:
        raise SelfPlayValidationError("Self-play game model identifier is inconsistent")
    if not game.actions or len(game.samples) != len(game.actions):
        raise SelfPlayValidationError(
            "Self-play game must contain one sample per action"
        )
    if any(
        sample.black_score != game.black_score or sample.white_score != game.white_score
        for sample in game.samples
    ):
        raise SelfPlayValidationError("Self-play samples disagree on final scores")
    if game.total_simulations != len(game.actions) * simulations_per_move:
        raise SelfPlayValidationError(
            "Self-play simulation count is inconsistent with configured search"
        )
    environment = GameEnvironment(rules)
    for sample, action in zip(game.samples, game.actions, strict=True):
        _validate_sample_target(
            sample,
            environment.state,
            game.black_score,
            game.white_score,
        )
        try:
            environment.step(action)
        except Exception as error:
            raise SelfPlayValidationError(
                "Self-play action sequence cannot be replayed legally"
            ) from error
    if not environment.terminal():
        raise SelfPlayValidationError("Self-play action sequence is not terminal")
    result = environment.result()
    if (
        result.black_score != game.black_score
        or result.white_score != game.white_score
        or result.winner is not game.winner
    ):
        raise SelfPlayValidationError(
            "Self-play result disagrees with authoritative terminal scoring"
        )
    expected_winner: Player | None
    if game.black_score > game.white_score:
        expected_winner = Player.BLACK
    elif game.white_score > game.black_score:
        expected_winner = Player.WHITE
    else:
        expected_winner = None
    if game.winner is not expected_winner:
        raise SelfPlayValidationError("Self-play winner disagrees with final scores")


def _validate_sample_target(
    sample: ReplaySample,
    expected_state: GameState,
    black_score: int,
    white_score: int,
) -> None:
    if sample.state != expected_state:
        raise SelfPlayValidationError(
            "Self-play sample state disagrees with authoritative game replay"
        )
    if len(sample.policy) != expected_state.action_size:
        raise SelfPlayValidationError("Self-play policy length is invalid")
    if any(
        not isfinite(probability) or probability < 0.0 for probability in sample.policy
    ):
        raise SelfPlayValidationError("Self-play policy must be finite and nonnegative")
    if any(
        probability != 0.0
        for probability, legal in zip(
            sample.policy, expected_state.legal_mask, strict=True
        )
        if not legal
    ):
        raise SelfPlayValidationError(
            "Self-play policy must assign zero mass to illegal actions"
        )
    if not isclose(sum(sample.policy), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise SelfPlayValidationError("Self-play policy must sum to one")
    expected_win = _outcome_for_player(expected_state.to_play, black_score, white_score)
    if sample.win != expected_win:
        raise SelfPlayValidationError(
            "Self-play outcome disagrees with side to play and final scores"
        )


def _outcome_for_player(player: Player, black_score: int, white_score: int) -> float:
    if black_score == white_score:
        return 0.5
    winner = Player.BLACK if black_score > white_score else Player.WHITE
    return 1.0 if player is winner else 0.0


def _validate_model_identifier(model_identifier: str) -> None:
    if not isinstance(model_identifier, str) or not model_identifier.strip():
        raise SelfPlayValidationError("Self-play model identifier must not be empty")


def _put_response(
    response_queue: Any, response: _InferenceResponse, timeout: float
) -> None:
    try:
        response_queue.put(response, timeout=timeout)
    except queue.Full as error:
        raise SelfPlayInferenceError(
            "Worker inference response queue is full"
        ) from error


def _drain_queue(source: Any) -> tuple[object, ...]:
    messages: list[object] = []
    while True:
        try:
            messages.append(source.get_nowait())
        except queue.Empty:
            return tuple(messages)


def _shutdown_processes(
    processes: list[BaseProcess], *, timeout_seconds: float, terminate_first: bool
) -> bool:
    if terminate_first:
        for process in processes:
            if process.is_alive():
                process.terminate()
    for process in processes:
        process.join(timeout=timeout_seconds)
    forced = False
    for process in processes:
        if process.is_alive():
            forced = True
            process.kill()
            process.join(timeout=timeout_seconds)
    return forced


def _close_queues(
    request_queue: Any,
    result_queue: Any,
    response_queues: list[Any],
    *,
    cancel_pending: bool,
) -> None:
    for queue_instance in (request_queue, result_queue, *response_queues):
        if cancel_pending:
            queue_instance.cancel_join_thread()
        queue_instance.close()
        if not cancel_pending:
            queue_instance.join_thread()

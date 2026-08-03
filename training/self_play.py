"""Deterministic self-play games with spawn workers and centralized inference."""

from __future__ import annotations

import os
import queue
import random
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isclose, isfinite
from multiprocessing.process import BaseProcess
from time import monotonic, perf_counter
from typing import Any

import torch

from config.schema import MctsConfig, RulesConfig, SelfPlayConfig
from game import GameEnvironment, GameState, Player
from mcts import BatchEvaluator, Evaluation, MctsSearch, SearchMode
from training.errors import (
    SelfPlayInferenceError,
    SelfPlayValidationError,
    SelfPlayWorkerError,
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
    request_index: int
    states: tuple[GameState, ...]


@dataclass(frozen=True, slots=True)
class _InferenceResponse:
    request_index: int
    evaluations: tuple[Evaluation, ...] | None
    error: str | None


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


ProgressCallback = Callable[[SelfPlayProgress], None]


class _RemoteBatchEvaluator:
    """Synchronous worker facade over the coordinator's inference queues."""

    __slots__ = (
        "_request_index",
        "_request_queue",
        "_response_queue",
        "_timeout_seconds",
        "_worker_index",
    )

    def __init__(
        self,
        worker_index: int,
        request_queue: Any,
        response_queue: Any,
        timeout_seconds: float,
    ) -> None:
        self._worker_index = worker_index
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._timeout_seconds = timeout_seconds
        self._request_index = 0

    def evaluate(self, states: Sequence[GameState]) -> tuple[Evaluation, ...]:
        if not states:
            raise SelfPlayInferenceError("Worker cannot request empty inference")
        request = _InferenceRequest(
            worker_index=self._worker_index,
            request_index=self._request_index,
            states=tuple(states),
        )
        try:
            self._request_queue.put(request, timeout=self._timeout_seconds)
            raw_response = self._response_queue.get(timeout=self._timeout_seconds)
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
        self._request_index += 1
        if raw_response.error is not None:
            raise SelfPlayInferenceError(raw_response.error)
        if raw_response.evaluations is None:
            raise SelfPlayInferenceError("Central inference returned no evaluations")
        return raw_response.evaluations


class SelfPlayCoordinator:
    """Run isolated search workers while retaining sole evaluator ownership."""

    __slots__ = ("_config", "_evaluator", "_mcts", "_model_identifier", "_rules")

    def __init__(
        self,
        rules: RulesConfig,
        mcts: MctsConfig,
        config: SelfPlayConfig,
        evaluator: BatchEvaluator,
        *,
        model_identifier: str,
    ) -> None:
        _validate_model_identifier(model_identifier)
        if config.inference_max_batch_size < mcts.max_inference_batch_size:
            raise SelfPlayValidationError(
                "Central inference batch must cover one complete MCTS request"
            )
        self._rules = rules
        self._mcts = mcts
        self._config = config
        self._evaluator = evaluator
        self._model_identifier = model_identifier

    def run(
        self,
        *,
        game_count: int | None = None,
        master_seed: int | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> SelfPlayBatch:
        """Generate a complete deterministic batch or raise without partial output."""

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
        request_queue = context.Queue(maxsize=worker_count * 2)
        result_queue = context.Queue()
        response_queues = [context.Queue(maxsize=1) for _ in range(worker_count)]
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
                        response_queues[worker_index],
                        result_queue,
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
    ) -> SelfPlayBatch:
        started_at = perf_counter()
        active_workers = set(range(len(processes)))
        games: dict[int, SelfPlayGame] = {}
        deferred_request: _InferenceRequest | None = None
        inference_batches = 0
        inference_positions = 0
        maximum_batch_size = 0

        while active_workers:
            for message in _drain_queue(result_queue):
                self._handle_worker_message(
                    message,
                    active_workers,
                    games,
                    game_count,
                    progress_callback,
                )

            request = deferred_request
            deferred_request = None
            if request is None:
                try:
                    candidate = request_queue.get(
                        timeout=self._config.inference_batch_wait_seconds
                    )
                except queue.Empty:
                    self._raise_for_unreported_exit(processes, active_workers)
                    continue
                request = self._validate_request(candidate, active_workers)

            requests = [request]
            position_count = len(request.states)
            deadline = monotonic() + self._config.inference_batch_wait_seconds
            while position_count < self._config.inference_max_batch_size:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                try:
                    candidate = request_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                next_request = self._validate_request(candidate, active_workers)
                if (
                    position_count + len(next_request.states)
                    > self._config.inference_max_batch_size
                ):
                    deferred_request = next_request
                    break
                requests.append(next_request)
                position_count += len(next_request.states)

            self._serve_requests(requests, response_queues)
            inference_batches += 1
            inference_positions += position_count
            maximum_batch_size = max(maximum_batch_size, position_count)

        for message in _drain_queue(result_queue):
            self._handle_worker_message(
                message,
                active_workers,
                games,
                game_count,
                progress_callback,
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
        states = tuple(state for request in requests for state in request.states)
        try:
            evaluations = tuple(self._evaluator.evaluate(states))
            if len(evaluations) != len(states):
                raise SelfPlayInferenceError(
                    "Central evaluator output count must match its input batch"
                )
        except Exception as error:
            message = f"Central inference failed: {type(error).__name__}: {error}"
            for request in requests:
                _put_response(
                    response_queues[request.worker_index],
                    _InferenceResponse(request.request_index, None, message),
                    self._config.inference_response_timeout_seconds,
                )
            raise SelfPlayInferenceError(message) from error

        offset = 0
        for request in requests:
            request_count = len(request.states)
            response = _InferenceResponse(
                request_index=request.request_index,
                evaluations=evaluations[offset : offset + request_count],
                error=None,
            )
            _put_response(
                response_queues[request.worker_index],
                response,
                self._config.inference_response_timeout_seconds,
            )
            offset += request_count

    def _validate_request(
        self, candidate: object, active_workers: set[int]
    ) -> _InferenceRequest:
        if not isinstance(candidate, _InferenceRequest):
            raise SelfPlayInferenceError("Worker inference request type is invalid")
        if candidate.worker_index not in active_workers:
            raise SelfPlayInferenceError(
                "Inference request came from an inactive worker"
            )
        if not candidate.states:
            raise SelfPlayInferenceError("Worker inference request is empty")
        if len(candidate.states) > self._config.inference_max_batch_size:
            raise SelfPlayInferenceError(
                "Worker inference request exceeds central batch capacity"
            )
        return candidate

    def _handle_worker_message(
        self,
        message: object,
        active_workers: set[int],
        games: dict[int, SelfPlayGame],
        game_count: int,
        progress_callback: ProgressCallback | None,
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
    search = MctsSearch(rules, mcts, evaluator)
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
    response_queue: Any,
    result_queue: Any,
) -> None:
    process_id = os.getpid()
    try:
        torch.set_num_threads(config.worker_torch_threads)
        evaluator = _RemoteBatchEvaluator(
            worker_index,
            request_queue,
            response_queue,
            config.inference_response_timeout_seconds,
        )

        def publish_progress(progress: SelfPlayProgress) -> None:
            result_queue.put(progress)

        for job in jobs:
            game = play_self_play_game(
                rules,
                mcts,
                evaluator,
                game_index=job.game_index,
                worker_index=worker_index,
                seed=job.seed,
                model_identifier=model_identifier,
                progress_callback=publish_progress,
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
    finally:
        result_queue.put(_WorkerDone(worker_index, process_id))
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

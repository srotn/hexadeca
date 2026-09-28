"""Paired-opening multiprocess Arena with centralized dual-model inference."""

from __future__ import annotations

import gc
import os
import queue
import random
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from multiprocessing.process import BaseProcess
from time import monotonic, perf_counter
from typing import Any

import torch

from config.schema import EvaluationConfig, MctsConfig, RulesConfig
from game import Board, GameEnvironment, GameState, Player
from mcts import BatchEvaluator, Evaluation, MctsSearch, SearchMode
from training.errors import (
    ArenaInferenceError,
    ArenaValidationError,
    ArenaWorkerError,
)

ARENA_SCHEMA_VERSION = 1


class ArenaModel(Enum):
    """Stable model roles used by worker inference requests."""

    CANDIDATE = "candidate"
    BEST = "best"


@dataclass(frozen=True, slots=True)
class ArenaOpening:
    """One unique, legal forced prefix shared by a color-swapped game pair."""

    index: int
    actions: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ArenaProgress:
    """One completed Arena move suitable for monitoring."""

    game_index: int
    opening_index: int
    worker_index: int
    process_id: int
    candidate_player: Player
    player: Player
    action: int
    state: GameState
    simulations: int
    root_value: float
    search_elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class ArenaGame:
    """One validated candidate-versus-best game."""

    schema_version: int
    game_index: int
    opening_index: int
    worker_index: int
    process_id: int
    candidate_player: Player
    seed: int
    opening_actions: tuple[int, ...]
    actions: tuple[int, ...]
    black_score: int
    white_score: int
    winner: Player | None
    candidate_points: float
    total_simulations: int
    search_elapsed_seconds: float

    @property
    def plies(self) -> int:
        """Return the full game length, including forced opening actions."""

        return len(self.actions)


@dataclass(frozen=True, slots=True)
class ArenaBatch:
    """All paired games plus centralized inference throughput counters."""

    schema_version: int
    candidate_identifier: str
    best_identifier: str
    random_seed: int
    openings: tuple[ArenaOpening, ...]
    games: tuple[ArenaGame, ...]
    worker_processes: int
    candidate_inference_batches: int
    candidate_inference_positions: int
    best_inference_batches: int
    best_inference_positions: int
    maximum_inference_batch_size: int
    elapsed_seconds: float

    @property
    def candidate_points(self) -> float:
        """Return total candidate match points across all games."""

        return sum(game.candidate_points for game in self.games)

    @property
    def candidate_wins(self) -> int:
        return sum(game.winner is game.candidate_player for game in self.games)

    @property
    def best_wins(self) -> int:
        return sum(
            game.winner is not None and game.winner is not game.candidate_player
            for game in self.games
        )

    @property
    def draws(self) -> int:
        return sum(game.winner is None for game in self.games)

    @property
    def total_simulations(self) -> int:
        return sum(game.total_simulations for game in self.games)


@dataclass(frozen=True, slots=True)
class _ArenaJob:
    game_index: int
    opening: ArenaOpening
    candidate_player: Player
    seed: int


@dataclass(frozen=True, slots=True)
class _InferenceRequest:
    worker_index: int
    request_index: int
    model: ArenaModel
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


ArenaProgressCallback = Callable[[ArenaProgress], None]
ArenaGameCallback = Callable[[ArenaGame], None]


class _RemoteArenaEvaluator:
    __slots__ = (
        "_model",
        "_request_index",
        "_request_queue",
        "_response_queue",
        "_timeout_seconds",
        "_worker_index",
    )

    def __init__(
        self,
        worker_index: int,
        model: ArenaModel,
        request_queue: Any,
        response_queue: Any,
        timeout_seconds: float,
    ) -> None:
        self._worker_index = worker_index
        self._model = model
        self._request_queue = request_queue
        self._response_queue = response_queue
        self._timeout_seconds = timeout_seconds
        self._request_index = 0

    def evaluate(self, states: Sequence[GameState]) -> tuple[Evaluation, ...]:
        if not states:
            raise ArenaInferenceError("Arena worker cannot request empty inference")
        request = _InferenceRequest(
            worker_index=self._worker_index,
            request_index=self._request_index,
            model=self._model,
            states=tuple(states),
        )
        try:
            self._request_queue.put(request, timeout=self._timeout_seconds)
            response = self._response_queue.get(timeout=self._timeout_seconds)
        except queue.Full as error:
            raise ArenaInferenceError(
                "Arena inference request queue is full"
            ) from error
        except queue.Empty as error:
            raise ArenaInferenceError("Arena inference response timed out") from error
        if not isinstance(response, _InferenceResponse):
            raise ArenaInferenceError("Arena inference response type is invalid")
        if response.request_index != request.request_index:
            raise ArenaInferenceError("Arena inference response order is invalid")
        self._request_index += 1
        if response.error is not None:
            raise ArenaInferenceError(response.error)
        if response.evaluations is None:
            raise ArenaInferenceError("Arena inference returned no evaluations")
        return response.evaluations


def generate_openings(
    rules: RulesConfig,
    *,
    pair_count: int,
    opening_plies: int,
    random_seed: int,
) -> tuple[ArenaOpening, ...]:
    """Generate unique legal openings with distinct first actions."""

    if type(pair_count) is not int or pair_count <= 0:
        raise ArenaValidationError("Arena opening pair count must be positive")
    if pair_count > rules.action_size:
        raise ArenaValidationError(
            "Arena opening pairs cannot exceed initial legal actions"
        )
    if type(opening_plies) is not int or opening_plies <= 0:
        raise ArenaValidationError("Arena opening plies must be positive")
    if type(random_seed) is not int or random_seed < 0:
        raise ArenaValidationError("Arena opening seed must be nonnegative")

    random_source = random.Random(random_seed)
    first_actions = list(range(rules.action_size))
    random_source.shuffle(first_actions)
    openings: list[ArenaOpening] = []
    for opening_index, first_action in enumerate(first_actions[:pair_count]):
        board = Board(rules)
        board.apply(first_action)
        for _ in range(1, opening_plies):
            legal = board.legal_actions()
            if not legal:
                raise ArenaValidationError(
                    "Configured opening length reaches a terminal position"
                )
            board.apply(random_source.choice(legal))
        openings.append(
            ArenaOpening(
                index=opening_index,
                actions=tuple(move.action for move in board.history),
            )
        )
    _validate_openings(tuple(openings), rules, pair_count, opening_plies)
    return tuple(openings)


def play_arena_game(
    rules: RulesConfig,
    mcts: MctsConfig,
    candidate_evaluator: BatchEvaluator,
    best_evaluator: BatchEvaluator,
    *,
    game_index: int,
    opening: ArenaOpening,
    candidate_player: Player,
    seed: int,
    draw_score: float,
    worker_index: int = 0,
    progress_callback: ArenaProgressCallback | None = None,
    candidate_mcts: MctsConfig | None = None,
    best_mcts: MctsConfig | None = None,
) -> ArenaGame:
    """Play one deterministic post-opening evaluation game."""

    if type(game_index) is not int or game_index < 0:
        raise ArenaValidationError("Arena game index must be nonnegative")
    if not isinstance(candidate_player, Player):
        raise ArenaValidationError("Arena candidate player must be Black or White")
    if type(seed) is not int or seed < 0:
        raise ArenaValidationError("Arena game seed must be nonnegative")
    if not isfinite(draw_score) or draw_score != 0.5:
        raise ArenaValidationError("Arena draw score must be 0.5")
    _validate_openings((opening,), rules, 1, len(opening.actions), exact_indices=False)

    environment = GameEnvironment(rules)
    actions: list[int] = []
    for action in opening.actions:
        environment.step(action)
        actions.append(action)
    if environment.terminal():
        raise ArenaValidationError("Arena opening must not be terminal")

    candidate_search_config = candidate_mcts or mcts
    best_search_config = best_mcts or mcts
    searches = {
        candidate_player: MctsSearch(
            rules, candidate_search_config, candidate_evaluator
        ),
        candidate_player.opponent: MctsSearch(
            rules, best_search_config, best_evaluator
        ),
    }
    random_source = random.Random(seed)
    total_simulations = 0
    search_elapsed_seconds = 0.0
    process_id = os.getpid()
    while not environment.terminal():
        state = environment.state
        started_at = perf_counter()
        search_result = searches[state.to_play].run(
            state, SearchMode.EVALUATION, random_source=random_source
        )
        elapsed = perf_counter() - started_at
        action = search_result.select_action(random_source)
        transition = environment.step(action)
        actions.append(action)
        total_simulations += search_result.simulations
        search_elapsed_seconds += elapsed
        if progress_callback is not None:
            progress_callback(
                ArenaProgress(
                    game_index=game_index,
                    opening_index=opening.index,
                    worker_index=worker_index,
                    process_id=process_id,
                    candidate_player=candidate_player,
                    player=transition.move.player,
                    action=action,
                    state=transition.state,
                    simulations=search_result.simulations,
                    root_value=search_result.root_value,
                    search_elapsed_seconds=elapsed,
                )
            )

    result = environment.result()
    candidate_points = (
        draw_score
        if result.winner is None
        else 1.0
        if result.winner is candidate_player
        else 0.0
    )
    game = ArenaGame(
        schema_version=ARENA_SCHEMA_VERSION,
        game_index=game_index,
        opening_index=opening.index,
        worker_index=worker_index,
        process_id=process_id,
        candidate_player=candidate_player,
        seed=seed,
        opening_actions=opening.actions,
        actions=tuple(actions),
        black_score=result.black_score,
        white_score=result.white_score,
        winner=result.winner,
        candidate_points=candidate_points,
        total_simulations=total_simulations,
        search_elapsed_seconds=search_elapsed_seconds,
    )
    _validate_game(
        game,
        rules,
        mcts,
        draw_score,
        candidate_mcts=candidate_search_config,
        best_mcts=best_search_config,
    )
    return game


class ArenaCoordinator:
    """Own both evaluators while isolated workers run paired MCTS games."""

    __slots__ = (
        "_best_evaluator",
        "_best_identifier",
        "_best_mcts",
        "_candidate_evaluator",
        "_candidate_identifier",
        "_candidate_mcts",
        "_config",
        "_mcts",
        "_rules",
    )

    def __init__(
        self,
        rules: RulesConfig,
        mcts: MctsConfig,
        config: EvaluationConfig,
        candidate_evaluator: BatchEvaluator,
        best_evaluator: BatchEvaluator,
        *,
        candidate_identifier: str,
        best_identifier: str,
        candidate_mcts: MctsConfig | None = None,
        best_mcts: MctsConfig | None = None,
    ) -> None:
        _validate_identifier(candidate_identifier, "candidate")
        _validate_identifier(best_identifier, "best")
        if candidate_identifier == best_identifier:
            raise ArenaValidationError("Arena models must have different identifiers")
        resolved_candidate_mcts = candidate_mcts or mcts
        resolved_best_mcts = best_mcts or mcts
        required_batch_size = max(
            resolved_candidate_mcts.max_inference_batch_size,
            resolved_best_mcts.max_inference_batch_size,
        )
        if config.inference_max_batch_size < required_batch_size:
            raise ArenaValidationError(
                "Arena inference capacity must cover one complete MCTS request"
            )
        if config.game_count <= 0 or config.game_count % 2 != 0:
            raise ArenaValidationError("Arena game count must be positive and even")
        if not config.paired_colors:
            raise ArenaValidationError("Arena requires paired color assignments")
        if config.opening_pair_count > rules.action_size:
            raise ArenaValidationError(
                "Arena opening pairs cannot exceed initial legal actions"
            )
        self._rules = rules
        self._mcts = mcts
        self._candidate_mcts = resolved_candidate_mcts
        self._best_mcts = resolved_best_mcts
        self._config = config
        self._candidate_evaluator = candidate_evaluator
        self._best_evaluator = best_evaluator
        self._candidate_identifier = candidate_identifier
        self._best_identifier = best_identifier

    def run(
        self,
        *,
        progress_callback: ArenaProgressCallback | None = None,
        game_callback: ArenaGameCallback | None = None,
        prior_games: Sequence[ArenaGame] = (),
    ) -> ArenaBatch:
        """Run the configured Arena, optionally retaining validated prior games.

        ``prior_games`` makes an interrupted external Arena report resumable.
        They are fully validated and included in the returned batch, while only
        their missing counterparts are assigned to workers. ``game_callback``
        is invoked in the parent process after each newly completed game has
        been validated, making it suitable for durable progress checkpoints.
        """

        openings = generate_openings(
            self._rules,
            pair_count=self._config.opening_pair_count,
            opening_plies=self._config.opening_plies,
            random_seed=self._config.random_seed,
        )
        jobs = _build_jobs(openings, self._config.random_seed)
        games = _validate_prior_games(
            prior_games,
            jobs,
            self._rules,
            self._mcts,
            self._config.draw_score,
            candidate_mcts=self._candidate_mcts,
            best_mcts=self._best_mcts,
        )
        pending_jobs = tuple(job for job in jobs if job.game_index not in games)
        if not pending_jobs:
            ordered_games = tuple(
                games[index] for index in range(self._config.game_count)
            )
            _validate_pairs(ordered_games, openings)
            return ArenaBatch(
                schema_version=ARENA_SCHEMA_VERSION,
                candidate_identifier=self._candidate_identifier,
                best_identifier=self._best_identifier,
                random_seed=self._config.random_seed,
                openings=openings,
                games=ordered_games,
                worker_processes=0,
                candidate_inference_batches=0,
                candidate_inference_positions=0,
                best_inference_batches=0,
                best_inference_positions=0,
                maximum_inference_batch_size=0,
                elapsed_seconds=0.0,
            )

        worker_count = min(self._config.worker_processes, len(pending_jobs))
        assignments: list[list[_ArenaJob]] = [[] for _ in range(worker_count)]
        for job in pending_jobs:
            assignments[job.game_index % worker_count].append(job)

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
            batch = self._event_loop(
                processes,
                request_queue,
                response_queues,
                result_queue,
                openings,
                progress_callback,
                game_callback,
                games,
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
            raise ArenaWorkerError(
                "Arena workers did not exit within the configured timeout"
            )
        return batch

    def _start_workers(
        self,
        context: Any,
        assignments: list[list[_ArenaJob]],
        request_queue: Any,
        response_queues: list[Any],
        result_queue: Any,
    ) -> list[BaseProcess]:
        processes: list[BaseProcess] = []
        try:
            for worker_index, jobs in enumerate(assignments):
                process = context.Process(
                    target=_arena_worker_main,
                    args=(
                        worker_index,
                        tuple(jobs),
                        self._rules,
                        self._mcts,
                        self._candidate_mcts,
                        self._best_mcts,
                        self._config,
                        request_queue,
                        response_queues[worker_index],
                        result_queue,
                    ),
                    name=f"arena-{worker_index}",
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

    def _event_loop(
        self,
        processes: list[BaseProcess],
        request_queue: Any,
        response_queues: list[Any],
        result_queue: Any,
        openings: tuple[ArenaOpening, ...],
        progress_callback: ArenaProgressCallback | None,
        game_callback: ArenaGameCallback | None,
        games: dict[int, ArenaGame],
    ) -> ArenaBatch:
        started_at = perf_counter()
        active_workers = set(range(len(processes)))
        deferred: _InferenceRequest | None = None
        batches = {ArenaModel.CANDIDATE: 0, ArenaModel.BEST: 0}
        positions = {ArenaModel.CANDIDATE: 0, ArenaModel.BEST: 0}
        maximum_batch_size = 0

        while active_workers:
            for message in _drain_queue(result_queue):
                self._handle_message(
                    message, active_workers, games, progress_callback, game_callback
                )
            request = deferred
            deferred = None
            if request is None:
                try:
                    raw = request_queue.get(
                        timeout=self._config.inference_batch_wait_seconds
                    )
                except queue.Empty:
                    for message in _drain_queue(result_queue):
                        self._handle_message(
                            message,
                            active_workers,
                            games,
                            progress_callback,
                            game_callback,
                        )
                    _raise_for_unreported_exit(processes, active_workers)
                    continue
                request = self._validate_request(raw, active_workers)

            requests = [request]
            position_count = len(request.states)
            deadline = monotonic() + self._config.inference_batch_wait_seconds
            while position_count < self._config.inference_max_batch_size:
                remaining = deadline - monotonic()
                if remaining <= 0:
                    break
                try:
                    raw = request_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                next_request = self._validate_request(raw, active_workers)
                if (
                    position_count + len(next_request.states)
                    > self._config.inference_max_batch_size
                ):
                    deferred = next_request
                    break
                requests.append(next_request)
                position_count += len(next_request.states)

            model_batches = self._serve_requests(requests, response_queues)
            for model, (
                served_batches,
                served_positions,
                largest_batch,
            ) in model_batches.items():
                batches[model] += served_batches
                positions[model] += served_positions
                maximum_batch_size = max(maximum_batch_size, largest_batch)

        for message in _drain_queue(result_queue):
            self._handle_message(
                message, active_workers, games, progress_callback, game_callback
            )
        if len(games) != self._config.game_count:
            missing = sorted(set(range(self._config.game_count)) - set(games))
            raise ArenaWorkerError(f"Arena batch is missing game results: {missing}")
        ordered_games = tuple(games[index] for index in range(self._config.game_count))
        _validate_pairs(ordered_games, openings)
        return ArenaBatch(
            schema_version=ARENA_SCHEMA_VERSION,
            candidate_identifier=self._candidate_identifier,
            best_identifier=self._best_identifier,
            random_seed=self._config.random_seed,
            openings=openings,
            games=ordered_games,
            worker_processes=len(processes),
            candidate_inference_batches=batches[ArenaModel.CANDIDATE],
            candidate_inference_positions=positions[ArenaModel.CANDIDATE],
            best_inference_batches=batches[ArenaModel.BEST],
            best_inference_positions=positions[ArenaModel.BEST],
            maximum_inference_batch_size=maximum_batch_size,
            elapsed_seconds=perf_counter() - started_at,
        )

    def _serve_requests(
        self,
        requests: list[_InferenceRequest],
        response_queues: list[Any],
    ) -> dict[ArenaModel, tuple[int, int, int]]:
        grouped = {
            model: [request for request in requests if request.model is model]
            for model in ArenaModel
        }
        served: dict[ArenaModel, tuple[int, int, int]] = {}
        for model, model_requests in grouped.items():
            if not model_requests:
                continue
            evaluator = (
                self._candidate_evaluator
                if model is ArenaModel.CANDIDATE
                else self._best_evaluator
            )
            states = tuple(
                state for request in model_requests for state in request.states
            )
            try:
                evaluations, evaluator_batches, largest_batch = (
                    self._evaluate_with_memory_fallback(evaluator, states)
                )
            except Exception as error:
                message = (
                    f"{model.value} inference failed: {type(error).__name__}: {error}"
                )
                for request in model_requests:
                    _put_response(
                        response_queues[request.worker_index],
                        _InferenceResponse(request.request_index, None, message),
                        self._config.inference_response_timeout_seconds,
                    )
                raise ArenaInferenceError(message) from error

            offset = 0
            for request in model_requests:
                count = len(request.states)
                _put_response(
                    response_queues[request.worker_index],
                    _InferenceResponse(
                        request.request_index,
                        evaluations[offset : offset + count],
                        None,
                    ),
                    self._config.inference_response_timeout_seconds,
                )
                offset += count
            served[model] = (evaluator_batches, len(states), largest_batch)
        return served

    @staticmethod
    def _evaluate_with_memory_fallback(
        evaluator: BatchEvaluator,
        states: tuple[GameState, ...],
    ) -> tuple[tuple[Evaluation, ...], int, int]:
        """Evaluate a batch, bisecting only after a host-memory failure."""

        try:
            evaluations = tuple(evaluator.evaluate(states))
        except MemoryError:
            if len(states) == 1:
                raise
            # The native encoder needs one contiguous NCHW NumPy array. A
            # fragmented Windows heap can reject a late long-Arena allocation
            # although its smaller equivalent batches still fit.
            gc.collect()
            midpoint = len(states) // 2
            left, left_batches, left_largest = (
                ArenaCoordinator._evaluate_with_memory_fallback(
                    evaluator, states[:midpoint]
                )
            )
            right, right_batches, right_largest = (
                ArenaCoordinator._evaluate_with_memory_fallback(
                    evaluator, states[midpoint:]
                )
            )
            return (
                left + right,
                left_batches + right_batches,
                max(left_largest, right_largest),
            )
        if len(evaluations) != len(states):
            raise ArenaInferenceError(
                "Arena evaluator output count must match its input batch"
            )
        return evaluations, 1, len(states)

    def _validate_request(
        self, raw: object, active_workers: set[int]
    ) -> _InferenceRequest:
        if not isinstance(raw, _InferenceRequest):
            raise ArenaInferenceError("Arena inference request type is invalid")
        if raw.worker_index not in active_workers:
            raise ArenaInferenceError("Arena request came from an inactive worker")
        if not isinstance(raw.model, ArenaModel):
            raise ArenaInferenceError("Arena request model role is invalid")
        if not raw.states:
            raise ArenaInferenceError("Arena inference request is empty")
        if len(raw.states) > self._config.inference_max_batch_size:
            raise ArenaInferenceError("Arena worker request exceeds batch capacity")
        return raw

    def _handle_message(
        self,
        message: object,
        active_workers: set[int],
        games: dict[int, ArenaGame],
        progress_callback: ArenaProgressCallback | None,
        game_callback: ArenaGameCallback | None,
    ) -> None:
        if isinstance(message, ArenaProgress):
            if progress_callback is not None:
                progress_callback(message)
            return
        if isinstance(message, ArenaGame):
            _validate_game(
                message,
                self._rules,
                self._mcts,
                self._config.draw_score,
                candidate_mcts=self._candidate_mcts,
                best_mcts=self._best_mcts,
            )
            if not 0 <= message.game_index < self._config.game_count:
                raise ArenaWorkerError("Arena game index is outside the batch")
            if message.game_index in games:
                raise ArenaWorkerError("Arena worker returned a duplicate game")
            games[message.game_index] = message
            if game_callback is not None:
                game_callback(message)
            return
        if isinstance(message, _WorkerFailure):
            raise ArenaWorkerError(
                f"Arena worker {message.worker_index} (PID {message.process_id}) "
                f"failed: {message.message}\n{message.traceback_text}"
            )
        if isinstance(message, _WorkerDone):
            if message.worker_index not in active_workers:
                raise ArenaWorkerError("Arena worker completed more than once")
            active_workers.remove(message.worker_index)
            return
        raise ArenaWorkerError(
            f"Unknown Arena worker message: {type(message).__name__}"
        )


def _build_jobs(
    openings: tuple[ArenaOpening, ...], random_seed: int
) -> tuple[_ArenaJob, ...]:
    jobs: list[_ArenaJob] = []
    for opening in openings:
        for offset, candidate_player in enumerate((Player.BLACK, Player.WHITE)):
            game_index = opening.index * 2 + offset
            jobs.append(
                _ArenaJob(
                    game_index=game_index,
                    opening=opening,
                    candidate_player=candidate_player,
                    seed=_derive_seed(random_seed, game_index),
                )
            )
    return tuple(jobs)


def _validate_prior_games(
    prior_games: Sequence[ArenaGame],
    jobs: tuple[_ArenaJob, ...],
    rules: RulesConfig,
    mcts: MctsConfig,
    draw_score: float,
    *,
    candidate_mcts: MctsConfig,
    best_mcts: MctsConfig,
) -> dict[int, ArenaGame]:
    """Validate externally persisted results before an Arena resume."""

    expected = {job.game_index: job for job in jobs}
    validated: dict[int, ArenaGame] = {}
    for game in prior_games:
        if not isinstance(game, ArenaGame):
            raise ArenaValidationError("Arena prior result has an invalid type")
        _validate_game(
            game,
            rules,
            mcts,
            draw_score,
            candidate_mcts=candidate_mcts,
            best_mcts=best_mcts,
        )
        job = expected.get(game.game_index)
        if job is None:
            raise ArenaValidationError("Arena prior result index is outside the batch")
        if (
            game.opening_index != job.opening.index
            or game.opening_actions != job.opening.actions
            or game.candidate_player is not job.candidate_player
            or game.seed != job.seed
        ):
            raise ArenaValidationError(
                "Arena prior result does not match this Arena configuration"
            )
        if game.game_index in validated:
            raise ArenaValidationError("Arena prior results contain a duplicate game")
        validated[game.game_index] = game
    return validated


def _arena_worker_main(
    worker_index: int,
    jobs: tuple[_ArenaJob, ...],
    rules: RulesConfig,
    mcts: MctsConfig,
    candidate_mcts: MctsConfig,
    best_mcts: MctsConfig,
    config: EvaluationConfig,
    request_queue: Any,
    response_queue: Any,
    result_queue: Any,
) -> None:
    process_id = os.getpid()
    try:
        torch.set_num_threads(config.worker_torch_threads)
        candidate = _RemoteArenaEvaluator(
            worker_index,
            ArenaModel.CANDIDATE,
            request_queue,
            response_queue,
            config.inference_response_timeout_seconds,
        )
        best = _RemoteArenaEvaluator(
            worker_index,
            ArenaModel.BEST,
            request_queue,
            response_queue,
            config.inference_response_timeout_seconds,
        )

        def publish(progress: ArenaProgress) -> None:
            result_queue.put(progress)

        for job in jobs:
            result_queue.put(
                play_arena_game(
                    rules,
                    mcts,
                    candidate,
                    best,
                    game_index=job.game_index,
                    opening=job.opening,
                    candidate_player=job.candidate_player,
                    seed=job.seed,
                    draw_score=config.draw_score,
                    worker_index=worker_index,
                    progress_callback=publish,
                    candidate_mcts=candidate_mcts,
                    best_mcts=best_mcts,
                )
            )
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


def _validate_openings(
    openings: tuple[ArenaOpening, ...],
    rules: RulesConfig,
    pair_count: int,
    opening_plies: int,
    *,
    exact_indices: bool = True,
) -> None:
    if len(openings) != pair_count:
        raise ArenaValidationError("Arena opening count is inconsistent")
    seen: set[tuple[int, ...]] = set()
    first_actions: set[int] = set()
    for expected_index, opening in enumerate(openings):
        if not isinstance(opening, ArenaOpening):
            raise ArenaValidationError("Arena opening type is invalid")
        if opening.index < 0:
            raise ArenaValidationError("Arena opening index must be nonnegative")
        if exact_indices and opening.index != expected_index:
            raise ArenaValidationError("Arena opening indices must be contiguous")
        if len(opening.actions) != opening_plies:
            raise ArenaValidationError("Arena opening length is inconsistent")
        if opening.actions in seen or opening.actions[0] in first_actions:
            raise ArenaValidationError("Arena openings must be unique by first action")
        board = Board(rules)
        try:
            for action in opening.actions:
                board.apply(action)
        except Exception as error:
            raise ArenaValidationError(
                "Arena opening is not legally replayable"
            ) from error
        if board.is_terminal:
            raise ArenaValidationError("Arena opening must leave a playable state")
        seen.add(opening.actions)
        first_actions.add(opening.actions[0])


def _validate_game(
    game: ArenaGame,
    rules: RulesConfig,
    mcts: MctsConfig,
    draw_score: float,
    *,
    candidate_mcts: MctsConfig | None = None,
    best_mcts: MctsConfig | None = None,
) -> None:
    if game.schema_version != ARENA_SCHEMA_VERSION:
        raise ArenaValidationError("Arena game schema version is unsupported")
    if game.game_index < 0 or game.opening_index < 0:
        raise ArenaValidationError("Arena game indices must be nonnegative")
    if game.candidate_player not in {Player.BLACK, Player.WHITE}:
        raise ArenaValidationError("Arena candidate player is invalid")
    if game.actions[: len(game.opening_actions)] != game.opening_actions:
        raise ArenaValidationError("Arena game does not preserve its opening prefix")
    environment = GameEnvironment(rules)
    try:
        for action in game.actions:
            environment.step(action)
    except Exception as error:
        raise ArenaValidationError("Arena game actions cannot be replayed") from error
    if not environment.terminal():
        raise ArenaValidationError("Arena game must terminate")
    result = environment.result()
    if (
        result.black_score != game.black_score
        or result.white_score != game.white_score
        or result.winner is not game.winner
    ):
        raise ArenaValidationError("Arena game disagrees with official scoring")
    expected_points = (
        draw_score
        if result.winner is None
        else 1.0
        if result.winner is game.candidate_player
        else 0.0
    )
    if game.candidate_points != expected_points:
        raise ArenaValidationError("Arena candidate points are inconsistent")
    candidate_search_config = candidate_mcts or mcts
    best_search_config = best_mcts or mcts
    searched_plies = len(game.actions) - len(game.opening_actions)
    expected_simulations = 0
    for offset in range(searched_plies):
        player = (
            Player.BLACK
            if (len(game.opening_actions) + offset) % 2 == 0
            else Player.WHITE
        )
        expected_simulations += (
            candidate_search_config.evaluation_simulations
            if player is game.candidate_player
            else best_search_config.evaluation_simulations
        )
    if game.total_simulations != expected_simulations:
        raise ArenaValidationError("Arena simulation count is inconsistent")
    if not isfinite(game.search_elapsed_seconds) or game.search_elapsed_seconds < 0:
        raise ArenaValidationError("Arena search duration is invalid")


def _validate_pairs(
    games: tuple[ArenaGame, ...], openings: tuple[ArenaOpening, ...]
) -> None:
    if len(games) != len(openings) * 2:
        raise ArenaValidationError("Arena paired game count is inconsistent")
    for opening in openings:
        first, second = games[opening.index * 2 : opening.index * 2 + 2]
        if (
            first.opening_index != opening.index
            or second.opening_index != opening.index
            or first.opening_actions != opening.actions
            or second.opening_actions != opening.actions
            or first.candidate_player is not Player.BLACK
            or second.candidate_player is not Player.WHITE
        ):
            raise ArenaValidationError("Arena color-swapped opening pair is invalid")


def _derive_seed(master_seed: int, game_index: int) -> int:
    value = (master_seed + (game_index + 1) * 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & ((1 << 64) - 1)
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & ((1 << 64) - 1)
    return (value ^ (value >> 31)) & ((1 << 64) - 1)


def _validate_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ArenaValidationError(f"Arena {name} identifier must not be empty")


def _put_response(
    queue_instance: Any, response: _InferenceResponse, timeout: float
) -> None:
    try:
        queue_instance.put(response, timeout=timeout)
    except queue.Full as error:
        raise ArenaInferenceError("Arena response queue is full") from error


def _drain_queue(source: Any) -> tuple[object, ...]:
    messages: list[object] = []
    while True:
        try:
            messages.append(source.get_nowait())
        except queue.Empty:
            return tuple(messages)


def _raise_for_unreported_exit(
    processes: list[BaseProcess], active_workers: set[int]
) -> None:
    for worker_index in sorted(active_workers):
        process = processes[worker_index]
        if process.exitcode is not None:
            raise ArenaWorkerError(
                f"Arena worker {worker_index} exited with code {process.exitcode} "
                "without a completion message"
            )


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

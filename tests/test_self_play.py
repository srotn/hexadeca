"""Single-game and centralized-inference multiprocess self-play tests."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import replace

import pytest

from config import load_config
from config.schema import MctsConfig, SelfPlayConfig
from game import GameEnvironment, GameState
from mcts import Evaluation, visit_probabilities
from network import NetworkSpecification
from training import (
    ReplayBuffer,
    SelfPlayCoordinator,
    SelfPlayInferenceError,
    SelfPlayProgress,
    derive_game_seed,
    play_self_play_game,
)


class UniformEvaluator:
    """Deterministic legal policy used to isolate self-play orchestration."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.process_ids: list[int] = []

    def evaluate(self, states: Sequence[GameState]) -> tuple[Evaluation, ...]:
        self.batch_sizes.append(len(states))
        self.process_ids.append(os.getpid())
        return tuple(
            Evaluation(
                policy=tuple(
                    1.0 / sum(state.legal_mask) if legal else 0.0
                    for legal in state.legal_mask
                ),
                value=0.0,
            )
            for state in states
        )


class FailingEvaluator:
    """Evaluator failure fixture for all-or-nothing coordinator behavior."""

    def evaluate(self, states: Sequence[GameState]) -> tuple[Evaluation, ...]:
        raise RuntimeError("inference unavailable")


class CentralGpuMctsEvaluator(UniformEvaluator):
    """CPU fixture for the production central CUDA root-operation contract."""

    supports_remote_gpu_mcts = True

    def __init__(self) -> None:
        super().__init__()
        self.root_requests: list[dict[str, object] | None] = []

    def evaluate_gpu_mcts_root(
        self,
        visit_counts: tuple[int, ...],
        temperature: float,
        puct: dict[str, object] | None,
    ) -> tuple[float, ...]:
        self.root_requests.append(puct)
        if puct is None:
            # The current production path centralizes visit-probability
            # normalization while Native C++ remains authoritative for PUCT
            # selection. No diagnostic edge vector is sent in that path.
            return visit_probabilities(visit_counts, temperature)
        puct_length = len(puct["visit_counts"])
        assert 0 < puct_length <= len(visit_counts)
        assert all(
            len(puct[name]) == puct_length
            for name in (
                "priors",
                "value_sums",
                "virtual_visit_counts",
                "virtual_value_sums",
                "solved_values",
            )
        )
        assert sum(puct["visit_counts"]) == sum(visit_counts)
        return visit_probabilities(visit_counts, temperature)


def _mcts_config() -> MctsConfig:
    return replace(
        load_config().mcts,
        training_simulations=4,
        max_inference_batch_size=4,
        exact_endgame_enabled=False,
    )


def _self_play_config(*, workers: int) -> SelfPlayConfig:
    return replace(
        load_config().self_play,
        games_per_iteration=4,
        worker_processes=workers,
        inference_transport="compact",
        inference_max_batch_size=8,
        inference_batch_wait_seconds=0.01,
        inference_response_timeout_seconds=30.0,
        worker_shutdown_timeout_seconds=5.0,
    )


def test_single_game_finalizes_official_targets_deterministically() -> None:
    """Every pre-move state receives the same final score and correct perspective."""

    config = load_config()
    mcts = _mcts_config()
    first_progress: list[SelfPlayProgress] = []
    first = play_self_play_game(
        config.rules,
        mcts,
        UniformEvaluator(),
        game_index=0,
        worker_index=0,
        seed=derive_game_seed(91, 0),
        model_identifier="candidate-1",
        progress_callback=first_progress.append,
    )
    second = play_self_play_game(
        config.rules,
        mcts,
        UniformEvaluator(),
        game_index=0,
        worker_index=0,
        seed=derive_game_seed(91, 0),
        model_identifier="candidate-1",
    )

    assert first.actions == second.actions
    assert [sample.policy for sample in first.samples] == [
        sample.policy for sample in second.samples
    ]
    assert first.plies == len(first.samples) == len(first_progress)
    assert first.total_simulations == first.plies * 4
    assert tuple(progress.action for progress in first_progress) == first.actions
    assert first_progress[-1].state.terminal is True
    assert all(sample.black_score == first.black_score for sample in first.samples)
    assert all(sample.white_score == first.white_score for sample in first.samples)
    assert all(
        sample.win
        == (
            0.5 if first.winner is None else float(sample.state.to_play is first.winner)
        )
        for sample in first.samples
    )

    environment = GameEnvironment(config.rules)
    for action in first.actions:
        environment.step(action)
    assert environment.terminal()
    assert environment.result().black_score == first.black_score
    assert environment.result().white_score == first.white_score


def test_multiprocess_workers_are_isolated_and_results_are_stably_ordered() -> None:
    """Search runs in child PIDs while all neural evaluation stays in the parent."""

    config = load_config()
    mcts = _mcts_config()
    evaluator = UniformEvaluator()
    progress: list[SelfPlayProgress] = []
    coordinator = SelfPlayCoordinator(
        config.rules,
        mcts,
        _self_play_config(workers=2),
        evaluator,
        model_identifier="candidate-2",
    )

    batch = coordinator.run(
        game_count=4,
        master_seed=1234,
        progress_callback=progress.append,
    )

    assert [game.game_index for game in batch.games] == [0, 1, 2, 3]
    assert [game.seed for game in batch.games] == [
        derive_game_seed(1234, index) for index in range(4)
    ]
    assert batch.worker_processes == 2
    assert {game.worker_index for game in batch.games} == {0, 1}
    assert all(game.process_id != os.getpid() for game in batch.games)
    assert set(evaluator.process_ids) == {os.getpid()}
    assert batch.inference_batches == len(evaluator.batch_sizes)
    assert batch.inference_positions == sum(evaluator.batch_sizes)
    assert batch.maximum_inference_batch_size == max(evaluator.batch_sizes)
    assert batch.maximum_inference_batch_size <= 8
    assert len(progress) == sum(game.plies for game in batch.games)
    assert batch.total_simulations == sum(game.plies for game in batch.games) * 4

    specification = NetworkSpecification.from_config(config.rules, config.network)
    replay = ReplayBuffer(config.rules, specification, len(batch.samples) + 1)
    assert batch.commit(replay) == len(batch.samples)
    assert len(replay) == len(batch.samples)
    assert [sample.state.history for sample in replay.snapshot()] == [
        sample.state.history for sample in batch.samples
    ]


def test_worker_root_mcts_arithmetic_executes_in_central_process() -> None:
    """Workers proxy PUCT and visit-policy work to the central device owner."""

    config = load_config()
    evaluator = CentralGpuMctsEvaluator()
    batch = SelfPlayCoordinator(
        config.rules,
        _mcts_config(),
        _self_play_config(workers=1),
        evaluator,
        model_identifier="candidate-central-gpu-mcts",
    ).run(game_count=1, master_seed=20260910)

    assert len(evaluator.root_requests) == batch.games[0].plies
    assert evaluator.process_ids and set(evaluator.process_ids) == {os.getpid()}


def test_game_seeds_and_outputs_do_not_depend_on_worker_count() -> None:
    """Changing process or lane parallelism cannot reassign game randomness."""

    config = load_config()
    mcts = _mcts_config()
    one_worker = SelfPlayCoordinator(
        config.rules,
        mcts,
        _self_play_config(workers=1),
        UniformEvaluator(),
        model_identifier="candidate-3",
    ).run(game_count=2, master_seed=777)
    two_workers = SelfPlayCoordinator(
        config.rules,
        mcts,
        _self_play_config(workers=2),
        UniformEvaluator(),
        model_identifier="candidate-3",
    ).run(game_count=2, master_seed=777)
    two_lanes = SelfPlayCoordinator(
        config.rules,
        mcts,
        _self_play_config(workers=1),
        UniformEvaluator(),
        model_identifier="candidate-3",
        games_per_worker=2,
    ).run(game_count=2, master_seed=777)

    assert [game.seed for game in one_worker.games] == [
        game.seed for game in two_workers.games
    ]
    assert [game.actions for game in one_worker.games] == [
        game.actions for game in two_workers.games
    ]
    assert [game.actions for game in one_worker.games] == [
        game.actions for game in two_lanes.games
    ]
    assert [sample.policy for sample in one_worker.samples] == [
        sample.policy for sample in two_workers.samples
    ]
    assert [sample.policy for sample in one_worker.samples] == [
        sample.policy for sample in two_lanes.samples
    ]
    assert two_lanes.worker_processes == 1
    assert len({game.process_id for game in two_lanes.games}) == 1
    assert two_lanes.maximum_inference_batch_size > mcts.max_inference_batch_size


def test_compact_and_object_transports_produce_identical_games() -> None:
    """Transport selection cannot alter MCTS ordering, targets, or results."""

    config = load_config()
    mcts = _mcts_config()
    compact = SelfPlayCoordinator(
        config.rules,
        mcts,
        _self_play_config(workers=1),
        UniformEvaluator(),
        model_identifier="candidate-transport",
    ).run(game_count=2, master_seed=314159)
    object_transport = SelfPlayCoordinator(
        config.rules,
        mcts,
        replace(_self_play_config(workers=1), inference_transport="object"),
        UniformEvaluator(),
        model_identifier="candidate-transport",
    ).run(game_count=2, master_seed=314159)

    assert [game.actions for game in compact.games] == [
        game.actions for game in object_transport.games
    ]
    assert [sample.policy for sample in compact.samples] == [
        sample.policy for sample in object_transport.samples
    ]


def test_central_inference_failure_aborts_the_complete_batch() -> None:
    """Evaluator failure is propagated and no partial SelfPlayBatch is returned."""

    config = load_config()
    coordinator = SelfPlayCoordinator(
        config.rules,
        _mcts_config(),
        _self_play_config(workers=2),
        FailingEvaluator(),
        model_identifier="candidate-4",
    )

    with pytest.raises(SelfPlayInferenceError, match="inference unavailable"):
        coordinator.run(game_count=2, master_seed=9)


def test_seed_derivation_is_stable_unique_and_validated() -> None:
    """SplitMix-derived streams are reproducible and reject invalid indices."""

    seeds = tuple(derive_game_seed(20_260_803, index) for index in range(100))

    assert seeds == tuple(derive_game_seed(20_260_803, index) for index in range(100))
    assert len(set(seeds)) == len(seeds)
    assert all(0 <= seed < 2**64 for seed in seeds)
    with pytest.raises(ValueError, match="nonnegative"):
        derive_game_seed(-1, 0)
    with pytest.raises(ValueError, match="nonnegative"):
        derive_game_seed(1, -1)

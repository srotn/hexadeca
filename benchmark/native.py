"""Reproducible Stage 12.1 native-versus-reference kernel benchmarks."""

from __future__ import annotations

import argparse
import json
import platform
import random
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import network.features as feature_module
from benchmark.timing import measure_operation
from config import load_config
from config.schema import MctsConfig, RulesConfig
from game import Board, GameEnvironment, GameState
from game.scoring import _collect_stones, _distance_table, _score_cell
from mcts import Evaluation, MctsSearch, SearchMode
from native import NativeBoard, is_available, score_cells
from network import NetworkSpecification


class _UniformEvaluator:
    """Return legal uniform policies to isolate native tree performance."""

    def evaluate(self, states: Sequence[GameState]) -> tuple[Evaluation, ...]:
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


def run_native_benchmark(
    rules: RulesConfig,
    mcts: MctsConfig,
    specification: NetworkSpecification,
    *,
    warmup_iterations: int,
    measurement_iterations: int,
    feature_batch_size: int,
    mcts_simulations: int,
    random_seed: int,
) -> dict[str, object]:
    """Measure equivalent reference and C++ kernels on fixed valid positions."""

    if not is_available() or NativeBoard is None or score_cells is None:
        raise RuntimeError("Stage 12.1 benchmark requires the native extension")
    if feature_batch_size <= 0 or mcts_simulations <= 0:
        raise ValueError("Native benchmark sizes must be positive")

    position = _build_position(rules, target_plies=32, seed=random_seed)
    native_board = NativeBoard(
        rules.board_size, rules.neighborhood_radius, rules.zobrist_seed
    )
    for move in position.history:
        native_board.apply(move.action)
    action = position.legal_actions()[0]
    terminal = _build_terminal_board(rules, random_seed)
    states = _feature_states(rules, feature_batch_size, random_seed)
    root_environment = GameEnvironment(rules)
    for move in position.history:
        root_environment.step(move.action)
    root_state = root_environment.state
    benchmark_mcts = replace(
        mcts,
        evaluation_simulations=mcts_simulations,
        max_inference_batch_size=min(mcts.max_inference_batch_size, 32),
    )

    def reference_apply_undo() -> None:
        position.apply(action)
        position.undo()

    def native_apply_undo() -> None:
        native_board.apply(action)
        native_board.undo()

    def reference_score() -> tuple[object, ...]:
        stones = _collect_stones(terminal.cells)
        distances = _distance_table(terminal.size)
        return tuple(
            _score_cell(action_index, terminal.size, stones, distances)
            for action_index in range(terminal.action_size)
        )

    def native_score() -> object:
        return score_cells(terminal.cells, terminal.size)

    def reference_features() -> object:
        return feature_module._encode_reference_batch(
            states, specification, device="cpu"
        )

    def native_features() -> object:
        return feature_module.encode_batch(states, specification, device="cpu")

    results: dict[str, object] = {
        "board_apply_undo": {
            "reference": measure_operation(
                reference_apply_undo, warmup_iterations, measurement_iterations
            ),
            "native": measure_operation(
                native_apply_undo, warmup_iterations, measurement_iterations
            ),
        },
        "terminal_scoring": {
            "reference": measure_operation(
                reference_score, warmup_iterations, measurement_iterations
            ),
            "native": measure_operation(
                native_score, warmup_iterations, measurement_iterations
            ),
        },
        "feature_encoding": {
            "reference": measure_operation(
                reference_features, warmup_iterations, measurement_iterations
            ),
            "native": measure_operation(
                native_features, warmup_iterations, measurement_iterations
            ),
        },
        "uniform_mcts": {
            "reference": _measure_search(
                rules,
                replace(benchmark_mcts, engine="reference"),
                root_state,
                warmup_iterations,
                measurement_iterations,
            ),
            "native": _measure_search(
                rules,
                replace(benchmark_mcts, engine="native"),
                root_state,
                warmup_iterations,
                measurement_iterations,
            ),
        },
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "platform": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.platform(),
            "processor": platform.processor() or "unavailable",
        },
        "parameters": {
            "warmup_iterations": warmup_iterations,
            "measurement_iterations": measurement_iterations,
            "feature_batch_size": feature_batch_size,
            "mcts_simulations": mcts_simulations,
            "random_seed": random_seed,
        },
        "results": results,
    }


def _measure_search(
    rules: RulesConfig,
    mcts: MctsConfig,
    state: GameState,
    warmup_iterations: int,
    measurement_iterations: int,
) -> dict[str, int | float]:
    """Measure exact configured simulations using one deterministic evaluator."""

    search = MctsSearch(rules, mcts, _UniformEvaluator())
    timing = measure_operation(
        lambda: search.run(state, SearchMode.EVALUATION),
        warmup_iterations,
        measurement_iterations,
    )
    return {
        **timing,
        "simulations_per_second": (
            float(timing["throughput_ops_per_second"]) * mcts.evaluation_simulations
        ),
    }


def _build_position(rules: RulesConfig, target_plies: int, seed: int) -> Board:
    """Build a deterministic non-terminal reference position."""

    board = Board(rules)
    random_source = random.Random(seed)
    while board.ply < target_plies and not board.is_terminal:
        board.apply(random_source.choice(board.legal_actions()))
    if board.is_terminal:
        board.undo()
    return board


def _build_terminal_board(rules: RulesConfig, seed: int) -> Board:
    """Build a deterministic terminal board for official scoring work."""

    board = Board(rules)
    random_source = random.Random(seed)
    while not board.is_terminal:
        board.apply(random_source.choice(board.legal_actions()))
    return board


def _feature_states(
    rules: RulesConfig, batch_size: int, seed: int
) -> tuple[GameState, ...]:
    """Build a deterministic state batch without changing its input distribution."""

    environment = GameEnvironment(rules)
    random_source = random.Random(seed)
    states = [environment.state]
    while not environment.terminal() and len(states) < batch_size:
        environment.step(random_source.choice(environment.legal_moves()))
        states.append(environment.state)
    return tuple(states[index % len(states)] for index in range(batch_size))


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the standard benchmark CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run Stage 12.1 measurements and persist the JSON report."""

    parsed = _parse_arguments(arguments)
    config = load_config(profile_path=parsed.profile)
    specification = NetworkSpecification.from_config(config.rules, config.network)
    report = run_native_benchmark(
        config.rules,
        config.mcts,
        specification,
        warmup_iterations=config.benchmark.native_warmup_iterations,
        measurement_iterations=config.benchmark.native_measurement_iterations,
        feature_batch_size=config.benchmark.native_feature_batch_size,
        mcts_simulations=config.benchmark.native_mcts_simulations,
        random_seed=config.benchmark.random_seed,
    )
    destination = parsed.output or (
        config.paths.benchmark_directory / "stage12-native.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

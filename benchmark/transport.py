"""Stage 12.2 compact-versus-object self-play transport benchmark."""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import random
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from benchmark.timing import measure_operation
from config import load_config
from config.schema import BenchmarkConfig, RulesConfig
from game import GameEnvironment, GameState
from mcts import Evaluation
from training.inference_transport import (
    pack_evaluations,
    pack_state_batch,
    unpack_evaluations,
    unpack_state_batch,
)


def run_transport_benchmark(
    rules: RulesConfig,
    config: BenchmarkConfig,
    *,
    random_seed: int,
) -> dict[str, object]:
    """Measure equivalent object and compact transport work on valid states."""

    states = _states(rules, config.transport_batch_size, random_seed)
    evaluations = _evaluations(states)
    object_request = pickle.dumps(states, protocol=pickle.HIGHEST_PROTOCOL)
    compact_request = pack_state_batch(states)
    object_response = pickle.dumps(evaluations, protocol=pickle.HIGHEST_PROTOCOL)
    compact_response = pack_evaluations(evaluations, action_size=rules.action_size)
    warmup = config.transport_warmup_iterations
    iterations = config.transport_measurement_iterations

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "platform": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.platform(),
        },
        "parameters": {
            "batch_size": len(states),
            "warmup_iterations": warmup,
            "measurement_iterations": iterations,
            "random_seed": random_seed,
        },
        "request": _transport_report(
            object_payload=object_request,
            compact_payload=compact_request,
            object_encode=lambda: pickle.dumps(
                states, protocol=pickle.HIGHEST_PROTOCOL
            ),
            object_decode=lambda: pickle.loads(object_request),
            compact_encode=lambda: pack_state_batch(states),
            compact_decode=lambda: unpack_state_batch(
                compact_request,
                ruleset_id=rules.ruleset_id,
                board_size=rules.board_size,
            ),
            warmup=warmup,
            iterations=iterations,
        ),
        "response": _transport_report(
            object_payload=object_response,
            compact_payload=compact_response,
            object_encode=lambda: pickle.dumps(
                evaluations, protocol=pickle.HIGHEST_PROTOCOL
            ),
            object_decode=lambda: pickle.loads(object_response),
            compact_encode=lambda: pack_evaluations(
                evaluations, action_size=rules.action_size
            ),
            compact_decode=lambda: unpack_evaluations(
                compact_response, action_size=rules.action_size
            ),
            warmup=warmup,
            iterations=iterations,
        ),
    }


def _transport_report(
    *,
    object_payload: bytes,
    compact_payload: bytes,
    object_encode: Callable[[], object],
    object_decode: Callable[[], object],
    compact_encode: Callable[[], object],
    compact_decode: Callable[[], object],
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    """Build one size and time comparison with explicit callable checks."""

    object_size = len(object_payload)
    compact_size = len(compact_payload)
    return {
        "object_bytes": object_size,
        "compact_bytes": compact_size,
        "size_reduction_percent": 100.0 * (1.0 - compact_size / object_size),
        "object_encode": measure_operation(object_encode, warmup, iterations),
        "object_decode": measure_operation(object_decode, warmup, iterations),
        "compact_encode": measure_operation(compact_encode, warmup, iterations),
        "compact_decode": measure_operation(compact_decode, warmup, iterations),
    }


def _states(
    rules: RulesConfig, batch_size: int, random_seed: int
) -> tuple[GameState, ...]:
    """Build deterministic valid positions without terminal-state duplication."""

    environment = GameEnvironment(rules)
    random_source = random.Random(random_seed)
    states: list[GameState] = []
    while len(states) < batch_size:
        if environment.terminal():
            environment.reset()
        states.append(environment.state)
        environment.step(random_source.choice(environment.legal_moves()))
    return tuple(states)


def _evaluations(states: Sequence[GameState]) -> tuple[Evaluation, ...]:
    """Construct deterministic valid policy/value outputs for response timing."""

    evaluations: list[Evaluation] = []
    for state in states:
        legal_count = sum(state.legal_mask)
        evaluations.append(
            Evaluation(
                policy=tuple(
                    1.0 / legal_count if legal else 0.0 for legal in state.legal_mask
                ),
                value=0.125,
            )
        )
    return tuple(evaluations)


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the configured transport benchmark and persist a JSON report."""

    parsed = _parse_arguments(arguments)
    app_config = load_config(profile_path=parsed.profile)
    report = run_transport_benchmark(
        app_config.rules,
        app_config.benchmark,
        random_seed=app_config.benchmark.random_seed,
    )
    destination = parsed.output or (
        app_config.paths.benchmark_directory / "stage12-compact-transport.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

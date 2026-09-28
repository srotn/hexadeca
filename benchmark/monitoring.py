"""Stage 13 event-publication and WebSocket-serialization benchmark."""

from __future__ import annotations

import argparse
import json
import platform
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from benchmark.timing import measure_operation
from config import load_config
from config.schema import MctsConfig, RulesConfig
from monitoring.events import EventBuffer


def run_monitoring_benchmark(
    rules: RulesConfig,
    mcts: MctsConfig,
    *,
    warmup_events: int,
    measurement_events: int,
) -> dict[str, object]:
    """Measure monitor-event work for a complete board snapshot payload."""

    if warmup_events < 0 or measurement_events <= 0:
        raise ValueError(
            "Monitoring benchmark event counts must be nonnegative/positive"
        )
    payload = _self_play_payload(rules, mcts)

    publish_events = EventBuffer(warmup_events + measurement_events + 1)
    serialize_events = EventBuffer(warmup_events + measurement_events + 1)
    publish = measure_operation(
        lambda: publish_events.publish("self_play_move", payload),
        warmup_events,
        measurement_events,
    )
    serialize = measure_operation(
        lambda: json.dumps(
            serialize_events.publish("self_play_move", payload).to_dict(),
            separators=(",", ":"),
        ),
        warmup_events,
        measurement_events,
    )
    encoded_event = json.dumps(
        EventBuffer(1).publish("self_play_move", payload).to_dict(),
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "platform": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.platform(),
        },
        "parameters": {
            "board_size": rules.board_size,
            "warmup_events": warmup_events,
            "measurement_events": measurement_events,
        },
        "results": {
            "event_publish": publish,
            "websocket_json_serialization": serialize,
            "event_payload_bytes": len(encoded_event),
        },
    }


def _self_play_payload(rules: RulesConfig, mcts: MctsConfig) -> dict[str, object]:
    cells = [0] * rules.action_size
    legal_mask = [True] * rules.action_size
    return {
        "game_index": 0,
        "worker_index": 0,
        "action": 0,
        "row": 0,
        "column": 0,
        "player": 1,
        "simulations": mcts.training_simulations,
        "root_value": 0.0,
        "inference_batches": 1,
        "inference_positions": mcts.max_inference_batch_size,
        "maximum_inference_batch_size": mcts.max_inference_batch_size,
        "search_elapsed_seconds": 0.0,
        "state": {
            "board": [
                cells[offset : offset + rules.board_size]
                for offset in range(0, rules.action_size, rules.board_size)
            ],
            "legal_mask": legal_mask,
            "legal_move_count": rules.action_size,
            "current_player": 1,
            "ply": 0,
            "terminal": False,
            "zobrist_hash": 0,
        },
    }


def _parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the configured Stage 13 benchmark and persist its JSON report."""

    parsed = _parse_arguments(arguments)
    config = load_config(profile_path=parsed.profile)
    report = run_monitoring_benchmark(
        config.rules,
        config.mcts,
        warmup_events=config.benchmark.monitoring_warmup_events,
        measurement_events=config.benchmark.monitoring_measurement_events,
    )
    destination = parsed.output or (
        config.paths.benchmark_directory / "stage13-monitoring.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

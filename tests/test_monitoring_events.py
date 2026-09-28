"""Unit tests for bounded monitoring event recovery."""

from __future__ import annotations

import pytest

from monitoring.events import EVENT_SCHEMA_VERSION, EventBuffer


def test_event_buffer_assigns_monotonic_protocol_envelopes() -> None:
    """Published events retain order and expose the versioned JSON contract."""

    events = EventBuffer(3)
    first = events.publish("runtime_state", {"state": "idle"})
    second = events.publish("telemetry", {"cpu_percent": 10.0})

    assert first.schema_version == EVENT_SCHEMA_VERSION == 1
    assert first.sequence == 1
    assert second.sequence == 2
    assert first.to_dict()["type"] == "runtime_state"
    assert events.latest_sequence == 2

    recovered = events.read_after(0, 10)
    assert recovered.dropped is False
    assert [event.sequence for event in recovered.events] == [1, 2]


def test_event_buffer_reports_when_a_client_cursor_falls_outside_retention() -> None:
    """A reconnecting client cannot mistake an event-history gap for success."""

    events = EventBuffer(2)
    events.publish("one", {})
    events.publish("two", {})
    events.publish("three", {})

    stale = events.read_after(0, 10)
    retained = events.read_after(1, 10)

    assert stale.dropped is True
    assert stale.events == ()
    assert stale.latest_sequence == 3
    assert retained.dropped is False
    assert [event.sequence for event in retained.events] == [2, 3]


@pytest.mark.parametrize(
    ("cursor", "limit", "message"),
    [(-1, 1, "cursor"), (0, 0, "limit")],
)
def test_event_buffer_rejects_invalid_recovery_ranges(
    cursor: int, limit: int, message: str
) -> None:
    """Recovery input validation rejects ambiguous client cursors and limits."""

    with pytest.raises(ValueError, match=message):
        EventBuffer(1).read_after(cursor, limit)

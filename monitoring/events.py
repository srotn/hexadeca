"""Versioned, bounded event history for live training monitoring."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Any

EVENT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class MonitoringEvent:
    """One immutable JSON-compatible monitoring event."""

    schema_version: int
    sequence: int
    timestamp: str
    event_type: str
    payload: Mapping[str, Any]

    def to_dict(self) -> dict[str, object]:
        """Return the protocol envelope used by HTTP and WebSocket consumers."""

        return {
            "schema_version": self.schema_version,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "type": self.event_type,
            "payload": dict(self.payload),
        }


@dataclass(frozen=True, slots=True)
class EventRead:
    """A consistent recovery read from the bounded event history."""

    events: tuple[MonitoringEvent, ...]
    dropped: bool
    latest_sequence: int


class EventBuffer:
    """Assign monotonic event sequences and retain a bounded recovery window."""

    __slots__ = ("_events", "_lock", "_next_sequence")

    def __init__(self, capacity: int) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("Event buffer capacity must be a positive integer")
        self._events: deque[MonitoringEvent] = deque(maxlen=capacity)
        self._lock = Lock()
        self._next_sequence = 1

    @property
    def latest_sequence(self) -> int:
        """Return the most recently assigned sequence, or zero when empty."""

        with self._lock:
            return self._next_sequence - 1

    def publish(self, event_type: str, payload: Mapping[str, Any]) -> MonitoringEvent:
        """Append a new event and return its immutable envelope."""

        if not isinstance(event_type, str) or not event_type:
            raise ValueError("Monitoring event type must be a non-empty string")
        if not isinstance(payload, Mapping):
            raise TypeError("Monitoring event payload must be a mapping")
        with self._lock:
            event = MonitoringEvent(
                schema_version=EVENT_SCHEMA_VERSION,
                sequence=self._next_sequence,
                timestamp=datetime.now(UTC).isoformat(),
                event_type=event_type,
                payload=dict(payload),
            )
            self._events.append(event)
            self._next_sequence += 1
            return event

    def read_after(self, after_sequence: int, limit: int) -> EventRead:
        """Read retained events after a client cursor and detect retention gaps."""

        if type(after_sequence) is not int or after_sequence < 0:
            raise ValueError("Event cursor must be a nonnegative integer")
        if type(limit) is not int or limit <= 0:
            raise ValueError("Event read limit must be a positive integer")
        with self._lock:
            latest_sequence = self._next_sequence - 1
            if not self._events:
                return EventRead((), False, latest_sequence)
            oldest_sequence = self._events[0].sequence
            dropped = after_sequence < oldest_sequence - 1
            if dropped:
                return EventRead((), True, latest_sequence)
            events = tuple(
                event for event in self._events if event.sequence > after_sequence
            )[:limit]
            return EventRead(events, False, latest_sequence)

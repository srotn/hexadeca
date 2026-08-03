"""Structured logging utilities for long-running Hexadeca processes."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config.schema import LoggingConfig


class JsonFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def __init__(self, run_id: str) -> None:
        """Create a formatter that attaches the immutable run identifier."""

        super().__init__()
        self._run_id = run_id

    def format(self, record: logging.LogRecord) -> str:
        """Serialize a log record with stable fields for log aggregation."""

        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "run_id": self._run_id,
        }
        event = getattr(record, "event", None)
        if event is not None:
            payload["event"] = event
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def configure_logger(
    name: str,
    log_directory: Path,
    settings: LoggingConfig,
    run_id: str,
) -> logging.Logger:
    """Create a named logger with configured console and JSONL destinations."""

    if not name:
        raise ValueError("Logger name must not be empty")
    if not run_id:
        raise ValueError("Run identifier must not be empty")
    if not settings.console_enabled and not settings.file_enabled:
        raise ValueError("At least one logging destination must be enabled")

    logger = logging.getLogger(name)
    close_logger(logger)
    logger.setLevel(settings.level)
    logger.propagate = False

    if settings.console_enabled:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(settings.level)
        console_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
        logger.addHandler(console_handler)

    if settings.file_enabled:
        log_directory.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_directory / settings.file_name,
            encoding="utf-8",
        )
        file_handler.setLevel(settings.level)
        file_handler.setFormatter(JsonFormatter(run_id))
        logger.addHandler(file_handler)

    return logger


def close_logger(logger: logging.Logger) -> None:
    """Close and detach handlers controlled by ``configure_logger``."""

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

"""Tests for structured logging setup."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from config import load_config
from utils import close_logger, configure_logger


def test_file_logger_writes_structured_event(tmp_path: Path) -> None:
    """Configured JSONL logs contain stable metadata and the custom event."""

    config = load_config()
    settings = replace(config.logging, console_enabled=False)
    logger = configure_logger("tests.logging", tmp_path, settings, "run-test-001")

    try:
        logger.info("configuration loaded", extra={"event": "config_loaded"})
    finally:
        close_logger(logger)

    records = (tmp_path / settings.file_name).read_text(encoding="utf-8").splitlines()

    assert len(records) == 1
    record = json.loads(records[0])
    assert record["event"] == "config_loaded"
    assert record["level"] == "INFO"
    assert record["message"] == "configuration loaded"
    assert record["run_id"] == "run-test-001"
    assert record["timestamp"].endswith("+00:00")

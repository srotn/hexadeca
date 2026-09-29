"""Command-line entry point for the Hexadeca monitoring service."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import uvicorn

from config import load_config
from monitoring.server import create_app


def main() -> None:
    """Load an optional checkpoint and serve the monitoring UI."""

    parser = argparse.ArgumentParser(description="Run Hexadeca monitoring")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint ID to load for interactive evaluation.",
    )
    parser.add_argument(
        "--bind-port",
        type=int,
        default=None,
        help="HTTP port; defaults to the configured monitoring port.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Optional TOML profile layered over config/default.toml.",
    )
    arguments = parser.parse_args()
    if arguments.bind_port is not None and not 1 <= arguments.bind_port <= 65_535:
        parser.error("--bind-port must be between 1 and 65535")

    config = load_config(profile_path=arguments.profile)
    # This entry point is a local demo; the deployment server keeps its config.
    config = replace(
        config, monitoring=replace(config.monitoring, bind_host="127.0.0.1")
    )
    project_root = Path(__file__).resolve().parents[1]
    app = create_app(config, project_root=project_root)
    if arguments.checkpoint is not None:
        result = app.state.runtime.load_checkpoint(arguments.checkpoint)
        if not result.accepted:
            app.state.runtime.close()
            parser.error(result.message)

    try:
        uvicorn.run(
            app,
            host=config.monitoring.bind_host,
            port=(
                arguments.bind_port
                if arguments.bind_port is not None
                else config.monitoring.bind_port
            ),
            ws_ping_interval=config.monitoring.websocket_ping_interval_seconds,
            ws_ping_timeout=config.monitoring.websocket_ping_timeout_seconds,
            ws_max_size=config.monitoring.websocket_max_message_bytes,
        )
    finally:
        app.state.runtime.close()


if __name__ == "__main__":
    main()

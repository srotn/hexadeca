"""FastAPI service exposing public Hexadeca training monitoring."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import tarfile
from collections import deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, cast

import uvicorn
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from config import load_config
from config.schema import AppConfig
from game import BoardError, GameEnvironment, GameState, Player
from monitoring.runtime import (
    InteractiveAnalysisResult,
    InteractiveMoveResult,
    InteractiveSearchUnavailableError,
    TrainingRuntime,
)
from training import CheckpointManager
from utils.logging import configure_logger


class CheckpointLoadRequest(BaseModel):
    """Validated body for a manual checkpoint restore request."""

    identifier: str


class ManualPositionRequest(BaseModel):
    """Strict browser payload for reconstructing one manual game position."""

    model_config = ConfigDict(extra="forbid", strict=True, populate_by_name=True)

    board: list[list[int]]
    current_player: int = Field(alias="currentPlayer")
    history: list[int] | None = None


class PlayPositionRequest(ManualPositionRequest):
    """Manual position plus an independently selected neural evaluator."""

    model_identifier: str | None = Field(default=None, alias="model")


class PlayMoveRequest(ManualPositionRequest):
    """Manual position plus a selected participant and MCTS budget."""

    simulations: int | None = None
    agent_identifier: str | None = Field(default=None, alias="agent")


def create_app(
    config: AppConfig,
    *,
    project_root: Path,
    runtime: TrainingRuntime | None = None,
) -> FastAPI:
    """Create the HTTP/WebSocket application around one runtime instance."""

    resolved_root = project_root.resolve()
    service_runtime = runtime or TrainingRuntime(
        config,
        project_root=resolved_root,
        logger=configure_logger(
            "hexadeca.monitoring",
            _resolve_path(resolved_root, config.paths.log_directory),
            config.logging,
            run_id="monitoring-service",
        ),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            service_runtime.close()

    app = FastAPI(title="Hexadeca AlphaZero Monitor", lifespan=lifespan)
    app.state.runtime = service_runtime

    def _frontend_path(name: str) -> Path:
        """Resolve a small, known frontend asset from the project root."""

        return resolved_root / name

    @app.get("/", response_class=FileResponse)
    async def frontend() -> FileResponse:
        # The root frontend is the installable player app. Keep the cloud
        # training panel available as a fallback for older deployment bundles.
        frontend_path = _frontend_path(config.monitoring.frontend_file_name)
        if not frontend_path.is_file():
            frontend_path = (
                resolved_root / "cloud-training" / config.monitoring.frontend_file_name
            )
        if not frontend_path.is_file():
            raise HTTPException(
                status_code=503, detail="Monitoring frontend is unavailable"
            )
        return FileResponse(frontend_path, media_type="text/html")

    @app.get("/Hexadeca.html", response_class=FileResponse)
    async def frontend_file() -> FileResponse:
        frontend_path = _frontend_path(config.monitoring.frontend_file_name)
        if not frontend_path.is_file():
            raise HTTPException(
                status_code=503, detail="Monitoring frontend is unavailable"
            )
        return FileResponse(frontend_path, media_type="text/html")

    @app.get("/manifest.webmanifest", response_class=FileResponse)
    async def app_manifest() -> FileResponse:
        path = _frontend_path("manifest.webmanifest")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="App manifest is unavailable")
        return FileResponse(path, media_type="application/manifest+json")

    @app.get("/sw.js", response_class=FileResponse)
    async def app_service_worker() -> FileResponse:
        path = _frontend_path("sw.js")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="App shell is unavailable")
        return FileResponse(path, media_type="application/javascript")

    @app.get("/icons/{filename}", response_class=FileResponse)
    async def app_icon(filename: str) -> FileResponse:
        if filename not in {
            "hexadeca-icon.svg",
            "hexadeca-icon-192.png",
            "hexadeca-icon-512.png",
        }:
            raise HTTPException(status_code=404, detail="Icon is unavailable")
        path = _frontend_path("icons") / filename
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Icon is unavailable")
        return FileResponse(path)

    @app.get("/healthz")
    async def health() -> dict[str, object]:
        return {"ok": True, "state": service_runtime.status()["runtime"]}

    @app.get("/api/status")
    async def status() -> dict[str, object]:
        return service_runtime.status()

    @app.get("/api/frontend-config")
    async def frontend_config() -> dict[str, object]:
        return {
            "websocket_path": "/ws",
            "reconnect_delay_seconds": (
                config.monitoring.client_reconnect_delay_seconds
            ),
            "reconnect_max_delay_seconds": (
                config.monitoring.client_reconnect_max_delay_seconds
            ),
            "status_refresh_interval_seconds": (
                config.monitoring.status_refresh_interval_seconds
            ),
            "interactive_move_simulations": {
                "default": config.monitoring.interactive_move_simulations_default,
                "minimum": config.monitoring.interactive_move_simulations_minimum,
                "maximum": config.monitoring.interactive_move_simulations_maximum,
                "step": config.monitoring.interactive_move_simulations_step,
            },
            "interactive_analysis_simulations": (
                config.monitoring.interactive_analysis_simulations
            ),
        }

    @app.post("/api/runtime/start")
    async def start_runtime() -> dict[str, object]:
        return service_runtime.start().to_dict()

    @app.post("/api/runtime/pause")
    async def pause_runtime() -> dict[str, object]:
        return service_runtime.request_pause().to_dict()

    @app.post("/api/runtime/resume")
    async def resume_runtime() -> dict[str, object]:
        return service_runtime.resume().to_dict()

    @app.post("/api/runtime/stop")
    async def stop_runtime() -> dict[str, object]:
        return service_runtime.request_stop().to_dict()

    @app.post("/api/play/move")
    async def play_move(request: PlayMoveRequest) -> dict[str, object]:
        """Choose an AI placement using a validated interactive MCTS budget."""

        try:
            _validate_interactive_move_simulations(config, request.simulations)
            if request.agent_identifier is not None:
                service_runtime.validate_agent_identifier(request.agent_identifier)
            state = _manual_game_state(
                request.board, request.current_player, config, request.history
            )
            if request.simulations is None and request.agent_identifier is None:
                result = cast(
                    InteractiveMoveResult,
                    await run_in_threadpool(
                        service_runtime.select_interactive_move,
                        state,
                    ),
                )
            elif request.agent_identifier is None:
                result = cast(
                    InteractiveMoveResult,
                    await run_in_threadpool(
                        service_runtime.select_interactive_move,
                        state,
                        request.simulations,
                    ),
                )
            else:
                result = cast(
                    InteractiveMoveResult,
                    await run_in_threadpool(
                        service_runtime.select_interactive_move,
                        state,
                        request.simulations,
                        request.agent_identifier,
                    ),
                )
        except InteractiveSearchUnavailableError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except (BoardError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return result.to_dict()

    @app.post("/api/play/analyze")
    async def analyze_position(request: PlayPositionRequest) -> dict[str, object]:
        """Analyze a position using the fixed configured interactive budget."""

        try:
            if request.model_identifier is not None:
                service_runtime.validate_neural_identifier(request.model_identifier)
            state = _manual_game_state(
                request.board, request.current_player, config, request.history
            )
            if request.model_identifier is None:
                result = cast(
                    InteractiveAnalysisResult,
                    await run_in_threadpool(
                        service_runtime.analyze_interactive_position, state
                    ),
                )
            else:
                result = cast(
                    InteractiveAnalysisResult,
                    await run_in_threadpool(
                        service_runtime.analyze_interactive_position,
                        state,
                        request.model_identifier,
                    ),
                )
        except InteractiveSearchUnavailableError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except (BoardError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return result.to_dict()

    @app.get("/api/checkpoints")
    async def checkpoints() -> dict[str, object]:
        manager = service_runtime.checkpoint_manager
        identifiers = manager.list_checkpoint_ids()
        return {
            "checkpoints": list(identifiers),
            "latest": _checkpoint_alias(manager, "latest"),
            "best": _checkpoint_alias(manager, "best"),
        }

    @app.get("/api/checkpoints/{identifier}/download")
    async def download_checkpoint(identifier: str) -> StreamingResponse:
        """Download one complete checkpoint bundle as a small tarball."""

        try:
            checkpoint_id = service_runtime.checkpoint_manager.resolve_identifier(
                identifier
            )
            service_runtime.checkpoint_manager.read_metadata(checkpoint_id)
        except (ValueError, OSError, RuntimeError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        checkpoint_directory = (
            _resolve_path(resolved_root, config.paths.checkpoint_directory)
            / "bundles"
            / checkpoint_id
        )
        manifest_path = checkpoint_directory / "manifest.json"
        state_path = checkpoint_directory / "state.pt"
        if not manifest_path.is_file() or not state_path.is_file():
            raise HTTPException(status_code=404, detail="Checkpoint is unavailable")
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w:gz") as archive:
            archive.add(manifest_path, arcname=f"{checkpoint_id}/manifest.json")
            archive.add(state_path, arcname=f"{checkpoint_id}/state.pt")
        payload.seek(0)
        return StreamingResponse(
            payload,
            media_type="application/gzip",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{checkpoint_id}.tar.gz"'
                )
            },
        )

    @app.get("/api/play/participants")
    async def play_participants() -> dict[str, object]:
        """List selectable players and independently selectable evaluators."""

        identifiers = service_runtime.checkpoint_manager.list_checkpoint_ids()
        neural_models = [
            {
                "id": identifier,
                "kind": "neural",
                "label": _checkpoint_label(identifier),
            }
            for identifier in identifiers
        ]
        default_neural = identifiers[-1] if identifiers else None
        return {
            "players": [
                {"id": "human", "kind": "human", "label": "Human"},
                {
                    "id": "old_champion",
                    "kind": "old_champion",
                    "label": "Old champion",
                },
                *neural_models,
            ],
            "evaluation_models": neural_models,
            "defaults": {
                "blue": "human",
                "orange": default_neural or "old_champion",
                "black": "human",
                "white": default_neural or "old_champion",
                "evaluation": default_neural,
            },
        }

    @app.post("/api/checkpoints/load")
    async def load_checkpoint(request: CheckpointLoadRequest) -> dict[str, object]:
        if not request.identifier:
            raise HTTPException(
                status_code=422, detail="Checkpoint identifier is required"
            )
        return service_runtime.load_checkpoint(request.identifier).to_dict()

    @app.get("/api/logs")
    async def logs() -> dict[str, object]:
        path = _resolve_path(resolved_root, config.paths.log_directory) / (
            config.logging.file_name
        )
        return {"lines": _tail_lines(path, config.monitoring.log_tail_lines)}

    @app.get("/api/logs/download", response_class=FileResponse)
    async def download_logs() -> FileResponse:
        path = _resolve_path(resolved_root, config.paths.log_directory) / (
            config.logging.file_name
        )
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Log file is unavailable")
        return FileResponse(path, media_type="application/x-ndjson", filename=path.name)

    @app.get("/api/training/iterations")
    async def training_iterations() -> dict[str, object]:
        """Return the complete per-iteration metrics history for charting."""

        path = _resolve_path(resolved_root, config.paths.run_directory) / (
            "training_iterations.jsonl"
        )
        records: list[dict[str, object]] = []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    records.append(value)
        return {"file": str(path), "iterations": records}

    @app.get("/api/training/iterations/download", response_class=FileResponse)
    async def download_training_iterations() -> FileResponse:
        path = _resolve_path(resolved_root, config.paths.run_directory) / (
            "training_iterations.jsonl"
        )
        if not path.is_file():
            raise HTTPException(status_code=404, detail="Iteration log is unavailable")
        return FileResponse(
            path,
            media_type="application/x-ndjson",
            filename="training_iterations.jsonl",
        )

    @app.websocket("/ws")
    async def websocket_events(
        websocket: WebSocket,
        after_sequence: Annotated[int, Query(ge=0)] = 0,
    ) -> None:
        await websocket.accept()
        cursor = after_sequence
        try:
            while True:
                recovered = service_runtime.events.read_after(
                    cursor, config.monitoring.event_replay_limit
                )
                if recovered.dropped:
                    cursor = recovered.latest_sequence
                    await websocket.send_json(
                        {
                            "schema_version": 1,
                            "sequence": cursor,
                            "timestamp": None,
                            "type": "resync",
                            "payload": {"status": service_runtime.status()},
                        }
                    )
                else:
                    for event in recovered.events:
                        await websocket.send_json(event.to_dict())
                        cursor = event.sequence
                await asyncio.sleep(config.monitoring.websocket_poll_interval_seconds)
        except WebSocketDisconnect:
            return

    return app


def main() -> None:
    """Run the config-driven public monitoring service."""

    parser = argparse.ArgumentParser(description="Run Hexadeca monitoring")
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="Optional TOML profile layered over config/default.toml.",
    )
    arguments = parser.parse_args()
    config = load_config(profile_path=arguments.profile)
    project_root = Path(__file__).resolve().parents[1]
    app = create_app(config, project_root=project_root)
    uvicorn.run(
        app,
        host=config.monitoring.bind_host,
        port=config.monitoring.bind_port,
        ws_ping_interval=config.monitoring.websocket_ping_interval_seconds,
        ws_ping_timeout=config.monitoring.websocket_ping_timeout_seconds,
        ws_max_size=config.monitoring.websocket_max_message_bytes,
    )


def _resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def _checkpoint_alias(manager: CheckpointManager, alias: str) -> str | None:
    if not manager.has_alias(alias):
        return None
    return manager.read_metadata(alias).checkpoint_id


def _checkpoint_label(identifier: str) -> str:
    """Return a concise stable label while retaining the full checkpoint ID."""

    prefix = "iteration-"
    if identifier.startswith(prefix) and identifier[len(prefix) :].isdigit():
        return f"Neural {int(identifier[len(prefix) :])}"
    return identifier


def _tail_lines(path: Path, count: int) -> list[str]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as log_file:
        return [line.rstrip("\n") for line in deque(log_file, maxlen=count)]


def _validate_interactive_move_simulations(
    config: AppConfig, simulations: int | None
) -> None:
    """Validate an optional browser move budget against the configured slider."""

    if simulations is None:
        return
    monitoring = config.monitoring
    minimum = monitoring.interactive_move_simulations_minimum
    maximum = monitoring.interactive_move_simulations_maximum
    step = monitoring.interactive_move_simulations_step
    if type(simulations) is not int or not minimum <= simulations <= maximum:
        raise ValueError(
            f"Interactive move simulations must be between {minimum} and {maximum}"
        )
    if (simulations - minimum) % step:
        raise ValueError(f"Interactive move simulations must use increments of {step}")


def _manual_game_state(
    board: list[list[int]],
    current_player: int,
    config: AppConfig,
    history: list[int] | None = None,
) -> GameState:
    """Validate a browser board by replaying a legal alternating history."""

    board_size = config.rules.board_size
    if len(board) != board_size or any(len(row) != board_size for row in board):
        raise ValueError(f"Board must be exactly {board_size} by {board_size}")
    if current_player not in {Player.BLACK.value, Player.WHITE.value}:
        raise ValueError("currentPlayer must be 1 for Black or 2 for White")

    black_actions: list[int] = []
    white_actions: list[int] = []
    flattened: list[int] = []
    for row_index, row in enumerate(board):
        for column_index, cell in enumerate(row):
            if cell not in {0, Player.BLACK.value, Player.WHITE.value}:
                raise ValueError("Board cells must be encoded as 0, 1, or 2")
            action = row_index * board_size + column_index
            flattened.append(cell)
            if cell == Player.BLACK.value:
                black_actions.append(action)
            elif cell == Player.WHITE.value:
                white_actions.append(action)

    if len(black_actions) not in {len(white_actions), len(white_actions) + 1}:
        raise ValueError("Board colours do not form an alternating game history")
    expected_player = (
        Player.BLACK.value
        if len(black_actions) == len(white_actions)
        else Player.WHITE.value
    )
    if current_player != expected_player:
        raise ValueError("currentPlayer does not match the alternating game history")

    environment = GameEnvironment(config.rules)
    if history is not None:
        if len(history) != len(black_actions) + len(white_actions):
            raise ValueError("History length does not match the submitted board")
        for action in history:
            environment.step(action)
    else:
        for index, black_action in enumerate(black_actions):
            environment.step(black_action)
            if index < len(white_actions):
                environment.step(white_actions[index])

    state = environment.state
    if state.cells != tuple(flattened):
        raise ValueError("Board reconstruction did not preserve the submitted state")
    if state.to_play.value != current_player:
        raise ValueError("Board reconstruction has an unexpected current player")
    if state.terminal:
        raise ValueError("AI cannot move because this game is already terminal")
    return state


if __name__ == "__main__":
    main()

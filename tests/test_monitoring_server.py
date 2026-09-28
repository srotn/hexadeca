"""HTTP and WebSocket contract tests for the monitoring service."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from config import load_config
from config.schema import AppConfig
from monitoring.runtime import (
    AnalysisBranchStep,
    AnalysisCandidate,
    InteractiveAnalysisResult,
    InteractiveMoveResult,
    InteractiveSearchUnavailableError,
    TrainingRuntime,
)
from monitoring.server import create_app


def _config(tmp_path: Path) -> AppConfig:
    config = load_config()
    paths = replace(
        config.paths,
        run_directory=tmp_path / "runs",
        checkpoint_directory=tmp_path / "checkpoints",
        log_directory=tmp_path / "logs",
        evaluation_directory=tmp_path / "evaluation-results",
    )
    monitoring = replace(
        config.monitoring,
        training_device="cpu",
        auto_resume=False,
        event_buffer_capacity=2,
        event_replay_limit=2,
        telemetry_interval_seconds=60.0,
        websocket_poll_interval_seconds=0.001,
    )
    return replace(config, paths=paths, monitoring=monitoring)


def test_monitoring_http_endpoints_expose_idle_runtime(tmp_path: Path) -> None:
    """The public service serves the existing frontend and a complete status."""

    (tmp_path / "Hexadeca.html").write_text("<html>monitor</html>", encoding="utf-8")
    app = create_app(_config(tmp_path), project_root=tmp_path)

    with TestClient(app) as client:
        response = client.get("/")
        status = client.get("/api/status")
        frontend_config = client.get("/api/frontend-config")
        checkpoints = client.get("/api/checkpoints")
        participants = client.get("/api/play/participants")
        logs = client.get("/api/logs")
        downloaded_logs = client.get("/api/logs/download")

        assert response.status_code == 200
        assert "monitor" in response.text
        assert status.status_code == 200
        assert status.json()["runtime"]["state"] == "idle"
        assert frontend_config.json()["websocket_path"] == "/ws"
        assert frontend_config.json()["status_refresh_interval_seconds"] == 5.0
        assert frontend_config.json()["interactive_move_simulations"] == {
            "default": 1600,
            "minimum": 200,
            "maximum": 3200,
            "step": 200,
        }
        assert frontend_config.json()["interactive_analysis_simulations"] == 1600
        assert checkpoints.json()["checkpoints"] == []
        assert participants.json()["players"] == [
            {"id": "human", "kind": "human", "label": "Human"},
            {
                "id": "old_champion",
                "kind": "old_champion",
                "label": "Old champion",
            },
        ]
        assert participants.json()["evaluation_models"] == []
        assert len(logs.json()["lines"]) == 1
        assert "Runtime state published" in logs.json()["lines"][0]
        assert downloaded_logs.status_code == 200
        assert "Runtime state published" in downloaded_logs.text


def test_websocket_replays_events_and_resyncs_a_stale_cursor(tmp_path: Path) -> None:
    """WebSocket clients receive retained events or an explicit state snapshot."""

    (tmp_path / "Hexadeca.html").write_text("<html>monitor</html>", encoding="utf-8")
    app = create_app(_config(tmp_path), project_root=tmp_path)
    runtime = app.state.runtime
    runtime.events.publish("first", {"value": 1})
    runtime.events.publish("second", {"value": 2})
    runtime.events.publish("third", {"value": 3})

    with TestClient(app) as client:
        with client.websocket_connect("/ws?after_sequence=3") as websocket:
            event = websocket.receive_json()
        with client.websocket_connect("/ws?after_sequence=0") as websocket:
            resync = websocket.receive_json()

    assert event["type"] == "third"
    assert event["sequence"] == 4
    assert resync["type"] == "resync"
    assert resync["sequence"] == 4
    assert resync["payload"]["status"]["runtime"]["state"] == "idle"


def test_manual_ai_move_reconstructs_a_valid_board_without_client_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manual AI requests use the canonical board state and server MCTS budget."""

    (tmp_path / "Hexadeca.html").write_text("<html>monitor</html>", encoding="utf-8")
    captured: list[object] = []

    def select_move(_: TrainingRuntime, state: object) -> InteractiveMoveResult:
        captured.append(state)
        return InteractiveMoveResult(
            action=3,
            board_size=16,
            simulations=1600,
            root_value=0.25,
            inference_batches=5,
            inference_positions=8,
            maximum_inference_batch_size=2,
            elapsed_seconds=0.12,
            model_identifier="fresh",
        )

    monkeypatch.setattr(TrainingRuntime, "select_interactive_move", select_move)
    board = [[0 for _ in range(16)] for _ in range(16)]
    board[0][0] = 1
    board[0][2] = 2
    app = create_app(_config(tmp_path), project_root=tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/play/move",
            json={"board": board, "currentPlayer": 1},
        )

    assert response.status_code == 200
    assert response.json()["row"] == 0
    assert response.json()["column"] == 3
    assert response.json()["simulations"] == 1600
    assert len(captured) == 1


def test_manual_ai_move_accepts_only_configured_slider_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The move endpoint forwards an explicitly selected slider budget."""

    (tmp_path / "Hexadeca.html").write_text("<html>monitor</html>", encoding="utf-8")
    captured: list[int | None] = []

    def select_move(
        _: TrainingRuntime, __: object, simulations: int | None = None
    ) -> InteractiveMoveResult:
        captured.append(simulations)
        return InteractiveMoveResult(
            action=3,
            board_size=16,
            simulations=simulations or 1600,
            root_value=0.25,
            inference_batches=5,
            inference_positions=8,
            maximum_inference_batch_size=2,
            elapsed_seconds=0.12,
            model_identifier="best",
        )

    monkeypatch.setattr(TrainingRuntime, "select_interactive_move", select_move)
    board = [[0 for _ in range(16)] for _ in range(16)]
    board[0][0] = 1
    board[0][2] = 2
    app = create_app(_config(tmp_path), project_root=tmp_path)

    with TestClient(app) as client:
        accepted = client.post(
            "/api/play/move",
            json={"board": board, "currentPlayer": 1, "simulations": 3200},
        )
        rejected = client.post(
            "/api/play/move",
            json={"board": board, "currentPlayer": 1, "simulations": 300},
        )

    assert accepted.status_code == 200
    assert accepted.json()["simulations"] == 3200
    assert rejected.status_code == 422
    assert captured == [3200]


def test_manual_ai_move_rejects_invalid_or_client_controlled_input(
    tmp_path: Path,
) -> None:
    """The public endpoint rejects malformed game states and extra parameters."""

    (tmp_path / "Hexadeca.html").write_text("<html>monitor</html>", encoding="utf-8")
    app = create_app(_config(tmp_path), project_root=tmp_path)
    board = [[0 for _ in range(16)] for _ in range(16)]
    board[0][0] = 1
    board[0][1] = 2

    with TestClient(app) as client:
        illegal = client.post(
            "/api/play/move",
            json={"board": board, "currentPlayer": 1},
        )
        extra = client.post(
            "/api/play/move",
            json={
                "board": [[0 for _ in range(16)] for _ in range(16)],
                "currentPlayer": 1,
                "unknown": 1,
            },
        )

    assert illegal.status_code == 422
    assert extra.status_code == 422


def test_manual_move_forwards_an_explicit_participant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The selected player is request-scoped and never mutates a global model."""

    (tmp_path / "Hexadeca.html").write_text("<html>monitor</html>", encoding="utf-8")
    captured: list[tuple[int | None, str | None]] = []

    def select_move(
        _: TrainingRuntime,
        __: object,
        simulations: int | None = None,
        agent_identifier: str | None = None,
    ) -> InteractiveMoveResult:
        captured.append((simulations, agent_identifier))
        return InteractiveMoveResult(
            action=3,
            board_size=16,
            simulations=0,
            root_value=0.0,
            inference_batches=0,
            inference_positions=0,
            maximum_inference_batch_size=0,
            elapsed_seconds=0.01,
            model_identifier=agent_identifier or "fresh",
        )

    monkeypatch.setattr(TrainingRuntime, "select_interactive_move", select_move)
    app = create_app(_config(tmp_path), project_root=tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/play/move",
            json={
                "board": [[0 for _ in range(16)] for _ in range(16)],
                "currentPlayer": 1,
                "simulations": 800,
                "agent": "old_champion",
            },
        )

    assert response.status_code == 200
    assert response.json()["model_identifier"] == "old_champion"
    assert captured == [(800, "old_champion")]


def test_manual_ai_move_reports_a_training_ownership_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Manual search cannot race an active training model owner."""

    (tmp_path / "Hexadeca.html").write_text("<html>monitor</html>", encoding="utf-8")

    def unavailable(_: TrainingRuntime, __: object) -> InteractiveMoveResult:
        raise InteractiveSearchUnavailableError("Training owns the model")

    monkeypatch.setattr(TrainingRuntime, "select_interactive_move", unavailable)
    app = create_app(_config(tmp_path), project_root=tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/play/move",
            json={
                "board": [[0 for _ in range(16)] for _ in range(16)],
                "currentPlayer": 1,
            },
        )

    assert response.status_code == 409
    assert response.json()["detail"] == "Training owns the model"


def test_manual_move_rejects_an_unknown_participant(tmp_path: Path) -> None:
    """Arbitrary identifiers cannot reach checkpoint or legacy dispatch."""

    (tmp_path / "Hexadeca.html").write_text("<html>monitor</html>", encoding="utf-8")
    app = create_app(_config(tmp_path), project_root=tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/play/move",
            json={
                "board": [[0 for _ in range(16)] for _ in range(16)],
                "currentPlayer": 1,
                "agent": "not-a-model",
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "Unknown neural checkpoint: not-a-model"


def test_manual_analysis_exposes_ranked_moves_and_principal_variation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The analysis endpoint returns versioned search diagnostics."""

    (tmp_path / "Hexadeca.html").write_text("<html>monitor</html>", encoding="utf-8")
    captured: list[object] = []

    def analyze(_: TrainingRuntime, state: object) -> InteractiveAnalysisResult:
        captured.append(state)
        return InteractiveAnalysisResult(
            current_player=1,
            simulations=1600,
            root_value=0.2,
            win_probability=0.6,
            black_win_probability=0.6,
            predicted_black_score=136.5,
            predicted_white_score=119.5,
            predicted_score_margin=17.0,
            candidates=(AnalysisCandidate(3, 0, 3, 1, 740, 0.2, 1.0, 0.3, 0.65),),
            principal_variation=(AnalysisBranchStep(3, 0, 3, 1, 740, 0.3, 0.65),),
            elapsed_seconds=0.25,
            model_identifier="best",
        )

    monkeypatch.setattr(TrainingRuntime, "analyze_interactive_position", analyze)
    board = [[0 for _ in range(16)] for _ in range(16)]
    board[0][0] = 1
    board[0][2] = 2
    app = create_app(_config(tmp_path), project_root=tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/play/analyze",
            json={"board": board, "currentPlayer": 1, "history": [0, 2]},
        )

    assert response.status_code == 200
    assert response.json()["schema_version"] == 1
    assert response.json()["predicted_score_margin"] == 17.0
    assert response.json()["candidates"][0]["action"] == 3
    assert response.json()["principal_variation"][0]["visits"] == 740
    assert len(captured) == 1


def test_manual_analysis_rejects_history_that_does_not_match_board(
    tmp_path: Path,
) -> None:
    """History-aware analysis cannot silently use corrupted temporal features."""

    (tmp_path / "Hexadeca.html").write_text("<html>monitor</html>", encoding="utf-8")
    board = [[0 for _ in range(16)] for _ in range(16)]
    board[0][0] = 1
    board[0][2] = 2
    app = create_app(_config(tmp_path), project_root=tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/play/analyze",
            json={"board": board, "currentPlayer": 1, "history": [0]},
        )

    assert response.status_code == 422
    assert "History length" in response.json()["detail"]


def test_manual_analysis_forwards_an_independent_evaluation_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The evaluator selection is independent of either player selection."""

    (tmp_path / "Hexadeca.html").write_text("<html>monitor</html>", encoding="utf-8")
    captured: list[str | None] = []

    def analyze(
        _: TrainingRuntime, __: object, model_identifier: str | None = None
    ) -> InteractiveAnalysisResult:
        captured.append(model_identifier)
        return InteractiveAnalysisResult(
            current_player=1,
            simulations=2,
            root_value=0.0,
            win_probability=0.5,
            black_win_probability=0.5,
            predicted_black_score=128.0,
            predicted_white_score=128.0,
            predicted_score_margin=0.0,
            candidates=(),
            principal_variation=(),
            elapsed_seconds=0.01,
            model_identifier=model_identifier or "fresh",
        )

    monkeypatch.setattr(TrainingRuntime, "analyze_interactive_position", analyze)
    monkeypatch.setattr(TrainingRuntime, "validate_neural_identifier", lambda *_: None)
    app = create_app(_config(tmp_path), project_root=tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/play/analyze",
            json={
                "board": [[0 for _ in range(16)] for _ in range(16)],
                "currentPlayer": 1,
                "history": [],
                "model": "iteration-004520",
            },
        )

    assert response.status_code == 200
    assert response.json()["model_identifier"] == "iteration-004520"
    assert captured == ["iteration-004520"]


def test_manual_analysis_rejects_move_budget_override(tmp_path: Path) -> None:
    """The move slider cannot alter the fixed position-analysis budget."""

    (tmp_path / "Hexadeca.html").write_text("<html>monitor</html>", encoding="utf-8")
    app = create_app(_config(tmp_path), project_root=tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/play/analyze",
            json={
                "board": [[0 for _ in range(16)] for _ in range(16)],
                "currentPlayer": 1,
                "history": [],
                "simulations": 200,
            },
        )

    assert response.status_code == 422


def test_manual_analysis_rejects_a_non_neural_evaluator(tmp_path: Path) -> None:
    """Legacy and human participants cannot masquerade as analysis models."""

    (tmp_path / "Hexadeca.html").write_text("<html>monitor</html>", encoding="utf-8")
    app = create_app(_config(tmp_path), project_root=tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/api/play/analyze",
            json={
                "board": [[0 for _ in range(16)] for _ in range(16)],
                "currentPlayer": 1,
                "history": [],
                "model": "old_champion",
            },
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "Unknown neural checkpoint: old_champion"

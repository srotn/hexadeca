"""Regression checks for the browser play and review interface."""

from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[1] / "Hexadeca.html"


def test_training_workspace_is_removed_in_favour_of_review() -> None:
    """The local game UI exposes only Play and Review workspaces."""

    frontend = _FRONTEND.read_text(encoding="utf-8")

    assert 'data-tab="play"' in frontend
    assert 'data-tab="review"' in frontend
    assert 'data-tab="train"' not in frontend
    assert 'id="train-content"' not in frontend
    assert "function setWorkspaceTab(name)" in frontend
    assert 'setMetric("analysis-status","Finish a game to review it")' in frontend


def test_training_controls_are_not_exposed_in_game_interface() -> None:
    """Training runtime controls no longer appear in the local game surface."""

    frontend = _FRONTEND.read_text(encoding="utf-8")

    assert 'id="btn-train-toggle"' not in frontend
    assert 'id="btn-runtime-stop"' not in frontend
    assert 'id="checkpoint-list"' not in frontend
    assert "Start Training" not in frontend


def test_play_analysis_panel_covers_requested_engine_diagnostics() -> None:
    """The existing play surface contains all interactive analysis views."""

    frontend = _FRONTEND.read_text(encoding="utf-8")

    assert 'id="analysis-candidates"' in frontend
    assert 'id="analysis-win"' in frontend
    assert 'id="winrate-chart"' in frontend
    assert 'id="analysis-branch"' in frontend
    assert 'id="analysis-previous"' in frontend
    assert 'fetch(apiUrl("/api/play/analyze")' in frontend
    assert "function renderBoardRecommendations()" in frontend
    assert "function finalizePreviousMove(after)" in frontend
    assert "history:gameActions()" in frontend


def test_play_analysis_supports_compact_mode_and_persistent_score_margin() -> None:
    """Normal play keeps a signed P1-minus-P2 territory estimate visible."""

    frontend = _FRONTEND.read_text(encoding="utf-8")

    assert 'id="analysis-detail-toggle"' in frontend
    assert "function setAnalysisDetails(visible)" in frontend
    assert 'id="position-winrail"' in frontend
    assert 'id="position-bar-black"' in frontend
    assert 'id="position-bar-white"' in frontend
    assert 'id="position-score-margin"' in frontend
    assert (
        "function renderPositionScoreMargin(margin,blackScore,whiteScore)" in frontend
    )
    assert "predicted_score_margin" in frontend
    assert "if(reviewMode||blindMode||!analysisDetailsVisible)return" in frontend


def test_completed_game_keeps_results_inline_without_a_modal() -> None:
    """Terminal results stay in the analysis surface and review remains available."""

    frontend = _FRONTEND.read_text(encoding="utf-8")

    assert 'id="game-over-modal"' not in frontend
    assert "function closeGameOverModal()" not in frontend
    assert 'setMetric("analysis-status","Final")' in frontend
    assert 'setMetric("analysis-win",(currentWin*100).toFixed(1)+"%")' in frontend
    assert 'data-tab="review"' in frontend


def test_completed_game_enters_a_stepwise_review_workspace() -> None:
    """Back opens a review timeline with persistent chart position tracking."""

    frontend = _FRONTEND.read_text(encoding="utf-8")

    assert 'id="review-controls"' in frontend
    assert 'id="review-range"' in frontend
    assert 'id="review-chart-step"' in frontend
    assert 'id="review-batch-progress"' in frontend
    assert "enterReviewMode()" in frontend
    assert "startFullGameReview(true)" in frontend
    assert "function reviewTo(value)" in frontend
    assert "function reviewSnapshot(ply)" in frontend
    assert "ctx.setLineDash([4,3])" in frontend
    assert "markerX=pointX(reviewPly)" in frontend
    assert 'id="review-return-play"' in frontend
    assert 'onclick="returnToPlay()"' in frontend
    assert "function returnToPlay(){rg()}" in frontend


def test_review_move_grading_uses_seven_visual_quality_levels() -> None:
    """Optional review grading decorates the current stone with seven classes."""

    frontend = _FRONTEND.read_text(encoding="utf-8")

    assert 'id="review-evaluation-toggle"' in frontend
    assert "function requestReviewAnalysis(ply)" in frontend
    assert "function startFullGameReview(forceGrades)" in frontend
    assert "function gradeReviewPly(ply)" in frontend
    assert "function reviewPositionRequest(ply,model)" in frontend
    assert 'renderReviewBatchProgress("Full-game analysis complete")' in frontend
    assert (
        'setMetric("review-grade-name","Evaluating position before move…")' in frontend
    )
    assert (
        'setMetric("review-grade-name","Evaluating position after move…")' in frontend
    )
    assert 'Promise.all([fetch(apiUrl("/api/play/analyze")' not in frontend
    assert "function qualityFromLoss(loss)" in frontend
    assert "function renderReviewMarker()" in frontend
    assert 'className="move-quality-badge "+grade.className' in frontend
    for name in (
        "Best",
        "Brilliant",
        "Good",
        "Inaccuracy",
        "Mistake",
        "Blunder",
        "Catastrophe",
    ):
        assert f'name:"{name}"' in frontend


def test_desktop_layout_fits_board_and_controls_into_one_viewport() -> None:
    """Desktop review uses a viewport-sized board and aligned side column."""

    frontend = _FRONTEND.read_text(encoding="utf-8")

    assert "--board-size:min(600px,calc(100vh - 190px)" in frontend
    assert "--workspace-height:calc(var(--board-size) + 50px)" in frontend
    assert "body{height:100vh;overflow:hidden" in frontend
    assert (
        ".side-column{max-width:none!important;"
        "height:var(--workspace-height);min-height:0;overflow:visible" in frontend
    )
    assert "height:var(--workspace-height);padding:14px!important" in frontend
    assert "overflow-y:auto;overscroll-behavior:contain" in frontend
    assert "border-radius:16px" in frontend
    assert 'class="analysis-extras"' in frontend
    assert 'class="board-shell ' in frontend
    assert 'class="side-column ' in frontend


def test_move_simulation_slider_is_separate_from_fixed_analysis_budget() -> None:
    """The browser sends the move budget but leaves analysis at the server default."""

    frontend = _FRONTEND.read_text(encoding="utf-8")

    assert 'id="move-simulations"' in frontend
    assert 'min="200" max="3200" step="200"' in frontend
    assert "simulations:moveSimulations" in frontend
    assert "function setMoveSimulations(value)" in frontend
    assert "agent:p" in frontend
    assert "model:evaluator.value" in frontend


def test_ai_play_can_pause_before_applying_the_next_computed_move() -> None:
    """The play toolbar can pause an AI chain without mutating the backend."""

    frontend = _FRONTEND.read_text(encoding="utf-8")

    assert 'id="btn-pause-ai"' in frontend
    assert 'onclick="toggleAiPause()"' in frontend
    assert "function toggleAiPause()" in frontend
    assert "function updateAiPauseButton()" in frontend
    assert 'button.innerText=aiPaused?"Resume AI":"Pause AI"' in frontend
    assert "if(serial!==aiMoveSerial||!al||aiPaused)" in frontend
    assert 'if(bl||ia||analysisPending||aiPaused||md!=="play")return' in frontend


def test_play_participants_and_evaluator_are_independently_selectable() -> None:
    """Both colours and analysis use the server-provided model catalogue."""

    frontend = _FRONTEND.read_text(encoding="utf-8")

    assert 'id="p1-type"' in frontend
    assert 'id="p2-type"' in frontend
    assert 'id="evaluation-model"' in frontend
    assert 'fetch(apiUrl("/api/play/participants"))' in frontend
    assert "function loadPlayParticipants()" in frontend
    assert "function isHuman(agent)" in frontend
    assert 'value="ai"' not in frontend


def test_blind_mode_hides_live_evaluation_and_territory() -> None:
    """Blind play suppresses display analysis without changing move search."""

    frontend = _FRONTEND.read_text(encoding="utf-8")

    assert 'id="blind-mode-toggle"' in frontend
    assert "function setBlindMode(enabled,refresh)" in frontend
    assert 'id("score-display").classList.toggle("hidden",blindMode)' in frontend
    assert 'id("position-winrail").classList.toggle("invisible",blindMode)' in frontend
    assert 'id("analysis-panel").classList.toggle("hidden",blindMode)' in frontend
    assert 'if(reviewMode||blindMode||md!=="play"' in frontend
    assert "if(!blindMode&&b[r][c]===0)" in frontend
    assert "if(fb&&!blindMode)" in frontend


def test_terminal_game_download_contains_moves_and_encoded_final_matrix() -> None:
    """Completed games expose a download containing the full feature payload."""

    frontend = _FRONTEND.read_text(encoding="utf-8")

    assert 'id="btn-save-game"' in frontend
    assert "setTerminalDownloadState(true)" in frontend
    assert "function finalMatrixForBoard(board)" in frontend
    assert "moves:gameActions()" in frontend
    assert "var record={moves:gameActions(),final_matrix:finalMatrix}" in frontend
    assert "button.disabled=!complete" in frontend
    assert "if(!blindMode&&hr>=0" in frontend

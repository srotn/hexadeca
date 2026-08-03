"""Golden and reference-parity tests for Stage 4 terminal scoring."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from config import load_config
from game import (
    Board,
    CellScore,
    InvalidActionError,
    NonTerminalBoardError,
    Player,
    ScoreResult,
    UnsupportedScoringRuleError,
    encode_action,
    score_cell,
    score_terminal,
)


def test_single_cell_terminal_board_scores_its_occupied_cell() -> None:
    """An occupied cell participates and its distance-zero stone earns one."""

    rules = replace(load_config().rules, board_size=1)
    board = Board(rules)
    board.place(0, 0)

    result = score_terminal(board)

    assert result.ruleset_id == rules.ruleset_id
    assert result.zobrist_hash == board.zobrist_hash
    assert result.black_score == 1
    assert result.white_score == 0
    assert result.total_score == 1
    assert result.winner is Player.BLACK
    assert result.score_for(Player.BLACK) == 1
    assert result.score_for(Player.WHITE) == 0
    assert result.cell_scores == (
        CellScore(
            action=0,
            owner=Player.BLACK,
            points=1,
            decisive_distance_squared=0,
        ),
    )


def test_official_scoring_rejects_non_terminal_board_without_mutation() -> None:
    """Official winner calculation cannot be requested while moves remain."""

    board = Board(load_config().rules)
    state_before = (board.cells, board.history, board.zobrist_hash)

    with pytest.raises(NonTerminalBoardError):
        score_terminal(board)

    assert (board.cells, board.history, board.zobrist_hash) == state_before


def test_equal_final_scores_have_no_winner() -> None:
    """An official equal score is represented as a draw without a winner."""

    result = ScoreResult(
        ruleset_id="hexadeca-v1",
        zobrist_hash=1,
        black_score=12,
        white_score=12,
        cell_scores=(),
    )

    assert result.winner is None


def test_nearest_layer_majority_awards_its_full_colour_count() -> None:
    """A two-Black versus one-White nearest layer gives Black two points."""

    board = _board_with_moves([(6, 8), (8, 10), (10, 8)])

    result = score_cell(board, encode_action(8, 8, board.size))

    assert result.owner is Player.BLACK
    assert result.points == 2
    assert result.decisive_distance_squared == 4


def test_tied_nearest_layer_advances_to_the_next_distance() -> None:
    """Equal nearest counts are skipped until a farther layer decides ownership."""

    board = _board_with_moves([(6, 8), (8, 10), (12, 8)])

    result = score_cell(board, encode_action(8, 8, board.size))

    assert result.owner is Player.BLACK
    assert result.points == 1
    assert result.decisive_distance_squared == 16


def test_all_distance_layers_tied_awards_no_points() -> None:
    """A cell stays neutral when every distance layer is colour-balanced."""

    board = _board_with_moves([(6, 8), (8, 10)])

    result = score_cell(board, encode_action(8, 8, board.size))

    assert result.owner is None
    assert result.points == 0
    assert result.decisive_distance_squared is None


def test_score_cell_rejects_invalid_action() -> None:
    """Per-cell preview scoring keeps the board action-range contract."""

    with pytest.raises(InvalidActionError):
        score_cell(Board(load_config().rules), 256)


@pytest.mark.parametrize(
    ("field_name", "unsupported_value"),
    [
        ("distance_metric", "manhattan"),
        ("tie_break", "neutral-on-first-tie"),
        ("majority_award", "one-point"),
        ("score_occupied_cells", False),
    ],
)
def test_unsupported_scoring_semantics_fail_explicitly(
    field_name: str, unsupported_value: object
) -> None:
    """A scorer never silently substitutes semantics from another ruleset."""

    rules = replace(load_config().rules, **{field_name: unsupported_value})
    board = Board(rules)
    board.place(0, 0)

    with pytest.raises(UnsupportedScoringRuleError):
        score_cell(board, 0)


def test_terminal_score_matches_independent_reference_for_every_cell() -> None:
    """A seeded terminal game matches a separate direct implementation exactly."""

    board = _random_terminal_board(20260804)
    state_before = (board.cells, board.history, board.zobrist_hash)

    result = score_terminal(board)
    expected_black, expected_white, expected_cells = _reference_score(
        board.cells, board.size
    )

    assert result.black_score == expected_black
    assert result.white_score == expected_white
    assert result.cell_scores == expected_cells
    assert len(result.cell_scores) == board.action_size
    assert result.total_score == sum(cell.points for cell in result.cell_scores)
    assert (board.cells, board.history, board.zobrist_hash) == state_before

    for move in board.history:
        occupied_score = result.cell_scores[move.action]
        assert occupied_score.owner is move.player
        assert occupied_score.points == 1
        assert occupied_score.decisive_distance_squared == 0


def _board_with_moves(coordinates: list[tuple[int, int]]) -> Board:
    board = Board(load_config().rules)
    for row, column in coordinates:
        board.place(row, column)
    return board


def _random_terminal_board(seed: int) -> Board:
    board = Board(load_config().rules)
    random_source = random.Random(seed)
    while not board.is_terminal:
        board.apply(random_source.choice(board.legal_actions()))
    return board


def _reference_score(
    cells: tuple[int, ...], board_size: int
) -> tuple[int, int, tuple[CellScore, ...]]:
    stones = tuple(
        (action, Player(cell)) for action, cell in enumerate(cells) if cell != 0
    )
    black_score = 0
    white_score = 0
    cell_scores: list[CellScore] = []

    for action in range(board_size * board_size):
        row, column = divmod(action, board_size)
        layers: dict[int, list[Player]] = {}
        for stone_action, player in stones:
            stone_row, stone_column = divmod(stone_action, board_size)
            distance_squared = (row - stone_row) ** 2 + (column - stone_column) ** 2
            layers.setdefault(distance_squared, []).append(player)

        cell_score = CellScore(action, None, 0, None)
        for distance_squared in sorted(layers):
            players = layers[distance_squared]
            black_count = players.count(Player.BLACK)
            white_count = players.count(Player.WHITE)
            if black_count > white_count:
                cell_score = CellScore(
                    action, Player.BLACK, black_count, distance_squared
                )
                black_score += black_count
                break
            if white_count > black_count:
                cell_score = CellScore(
                    action, Player.WHITE, white_count, distance_squared
                )
                white_score += white_count
                break
        cell_scores.append(cell_score)

    return black_score, white_score, tuple(cell_scores)

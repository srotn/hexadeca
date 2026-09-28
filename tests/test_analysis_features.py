"""Contract tests for compact game-record feature extraction."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from analysis import GameRecordError, analyze_game, parse_game_record
from analysis.features import _intrusion, terminal_divergence_reference_bounds


def _record(moves: list[int]) -> dict[str, object]:
    matrix = [[0 for _ in range(16)] for _ in range(16)]
    for ply, action in enumerate(moves):
        row, column = divmod(action, 16)
        matrix[row][column] = 1 if ply % 2 == 0 else -1
    return {"moves": moves, "final_matrix": matrix}


def _signed_matrix(
    owner: Callable[[int, int], int],
) -> tuple[tuple[int, ...], ...]:
    """Build a territory matrix from ``owner(row, column)``."""

    return tuple(tuple(owner(row, column) for column in range(16)) for row in range(16))


def test_parser_reconstructs_compact_record_and_feature_schema() -> None:
    record = _record([0, 34])

    parsed = parse_game_record(record, require_terminal=False)
    result = analyze_game(parsed, require_terminal=False)
    payload = result.to_dict()

    assert parsed.blue_positions == (0,)
    assert parsed.orange_positions == (34,)
    assert parsed.piece_counts == (2, 1, 1)
    # The circular 7x7 Gaussian kernel includes both stones.  The scalar is
    # exactly the mean of the corresponding point-heatmap contributions.
    assert result.D == pytest.approx(
        sum(
            result.point_heatmaps.divergence[action // 16][action % 16]
            for action in (0, 34)
        )
        / 2
    )
    assert result.I is not None
    assert 0.0 <= result.I <= 1.0
    assert set(payload["features"]) == {
        "D",
        "G0",
        "G1",
        "G2",
        "I",
        "S0",
        "S1",
        "S2",
        "B0",
        "B1",
        "B2",
    }
    assert len(payload["features"]["G0"]) == 4  # type: ignore[index]
    assert len(payload["heatmaps"]["G0"]["sparse"]) == 11  # type: ignore[index]
    assert len(payload["heatmaps"]["G0"]["dense"]) == 13  # type: ignore[index]
    assert payload["heatmaps"]["G0"]["sparse"][0][0] >= 0  # type: ignore[index]
    assert payload["schema_version"] == 3
    assert len(payload["point_heatmaps"]["divergence"]) == 16  # type: ignore[index]
    assert len(payload["point_heatmaps"]["intrusion"]) == 16  # type: ignore[index]


def test_point_divergence_matches_stone_feature_contributions() -> None:
    moves = [0, 34, 3]
    result = analyze_game(_record(moves), require_terminal=False)

    point_divergence = result.point_heatmaps.divergence
    divergence_at_stones = [
        point_divergence[action // 16][action % 16] for action in moves
    ]

    assert sum(divergence_at_stones) / len(moves) == pytest.approx(result.D)
    # The Gaussian-weighted circular window is smooth locally; the empty
    # corner uses the clipped-kernel baseline and therefore remains below 2.
    assert 0.0 < point_divergence[15][15] < 2.0
    assert point_divergence[0][1] > 0.0
    assert all(value > 0.0 for row in point_divergence for value in row)
    assert all(
        0.0 <= value <= 1.0 for row in result.point_heatmaps.intrusion for value in row
    )


def test_intrusion_scalar_is_weighted_mean_of_its_heatmap() -> None:
    matrix = _signed_matrix(lambda _row, column: 2 if column < 8 else -2)
    result = analyze_game({"moves": [], "final_matrix": matrix}, require_terminal=False)

    # Every cell is assigned, hence q=1 and the q-weighted scalar reduces to
    # the ordinary mean of the shared 16x16 heatmap.
    assert result.I == pytest.approx(
        sum(value for row in result.point_heatmaps.intrusion for value in row) / 256
    )


def test_divergence_uses_circular_truncated_gaussian_support() -> None:
    result = analyze_game(_record([119]), require_terminal=False)
    heatmap = result.point_heatmaps.divergence

    # The period filter is applied before taking the reciprocal, so the
    # isolated centre contribution is larger than the numerator scale 2.
    assert result.D == pytest.approx(heatmap[7][7])
    assert result.D > 2.0
    # (10, 7) lies on the radius-3 circle and receives a Gaussian contribution;
    # (10, 10) is outside the effective support and remains at the baseline.
    assert heatmap[10][7] > heatmap[10][10]
    assert heatmap[10][10] == pytest.approx(2.0)


def test_terminal_divergence_reference_bounds_use_dense_and_sparse_packings() -> None:
    lower, upper = terminal_divergence_reference_bounds()
    assert lower == pytest.approx(0.460745079754418)
    assert upper == pytest.approx(0.8988766846714716)
    assert lower < upper


def test_intrusion_separates_interleaved_and_partitioned_territory() -> None:
    interleaved = _signed_matrix(
        lambda row, column: 2 if (row + column) % 2 == 0 else -2
    )
    partitioned = _signed_matrix(lambda _row, column: 2 if column < 8 else -2)

    interleaved_score = _intrusion((), (), interleaved, 16)
    partitioned_score = _intrusion((), (), partitioned, 16)
    assert interleaved_score is not None
    assert partitioned_score is not None
    assert interleaved_score > partitioned_score


def test_intrusion_boundary_is_smooth_and_one_sided_territory_is_zero() -> None:
    one_sided = _signed_matrix(lambda _row, _column: 2)
    assert _intrusion((), (), one_sided, 16) == 0.0

    partitioned = _signed_matrix(lambda _row, column: 2 if column < 8 else -2)
    result = analyze_game(
        {"moves": [], "final_matrix": partitioned}, require_terminal=False
    )
    heatmap = result.point_heatmaps.intrusion

    assert heatmap[8][0] == pytest.approx(0.0)
    assert heatmap[8][15] == pytest.approx(0.0)
    assert heatmap[8][7] > heatmap[8][6] > heatmap[8][5]
    assert heatmap[8][8] > heatmap[8][9] > heatmap[8][10]


def test_real_stone_surrounded_by_enemy_territory_adds_penetration_response() -> None:
    action = 8 * 16 + 8
    territory_only = [list(row) for row in _signed_matrix(lambda _r, _c: -2)]
    territory_only[8][8] = 2
    without_stone = analyze_game(
        {"moves": [], "final_matrix": territory_only}, require_terminal=False
    )

    with_stone_matrix = [row[:] for row in territory_only]
    with_stone_matrix[8][8] = 1
    with_stone = analyze_game(
        {"moves": [action], "final_matrix": with_stone_matrix},
        require_terminal=False,
    )

    assert with_stone.I is not None
    assert without_stone.I is not None
    assert with_stone.I > without_stone.I
    assert (
        with_stone.point_heatmaps.intrusion[8][8]
        > without_stone.point_heatmaps.intrusion[8][8]
    )


def test_intrusion_is_null_only_when_no_territory_is_assigned() -> None:
    neutral = _signed_matrix(lambda _row, _column: 0)
    result = analyze_game(
        {"moves": [], "final_matrix": neutral}, require_terminal=False
    )

    assert result.I is None
    assert all(value == 0.0 for row in result.point_heatmaps.intrusion for value in row)


def test_edge_degree_uses_base_1_2() -> None:
    result = analyze_game(_record([0, 34]), require_terminal=False)

    assert result.B0 == pytest.approx(1 - 1 / 1.2)
    assert result.B1 == pytest.approx(1 - 1 / 1.2)
    assert result.B2 == 0.0


def test_parser_rejects_a_matrix_stone_that_is_not_in_moves() -> None:
    record = _record([0])
    record["final_matrix"][1][1] = -1  # type: ignore[index]

    with pytest.raises(GameRecordError, match="unexpected stone"):
        parse_game_record(record, require_terminal=False)


def test_terminal_requirement_is_explicit() -> None:
    record = _record([0, 255])

    with pytest.raises(GameRecordError, match="not terminal"):
        parse_game_record(record)

    parsed = parse_game_record(record, require_terminal=False)
    assert parsed.terminal is False
    assert parsed.legal_moves_remaining > 0
    assert parsed.piece_counts == (2, 1, 1)

    with pytest.raises(GameRecordError, match="not terminal"):
        analyze_game(parsed)

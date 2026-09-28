"""Exact squared-Euclidean distance-layer scoring for Hexadeca."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from config.schema import RulesConfig
from game.board import Board, Player, decode_action
from game.errors import NonTerminalBoardError, UnsupportedScoringRuleError
from native import is_available
from native import score_cells as native_score_cells


@dataclass(frozen=True, slots=True)
class CellScore:
    """The contribution of one row-major board cell to the final score."""

    action: int
    owner: Player | None
    points: int
    decisive_distance_squared: int | None


@dataclass(frozen=True, slots=True)
class ScoreResult:
    """An immutable official terminal score bound to one exact board state."""

    ruleset_id: str
    zobrist_hash: int
    black_score: int
    white_score: int
    cell_scores: tuple[CellScore, ...]

    @property
    def winner(self) -> Player | None:
        """Return the winning colour, or ``None`` when the score is tied."""

        if self.black_score > self.white_score:
            return Player.BLACK
        if self.white_score > self.black_score:
            return Player.WHITE
        return None

    @property
    def total_score(self) -> int:
        """Return the sum of points awarded to both players."""

        return self.black_score + self.white_score

    def score_for(self, player: Player) -> int:
        """Return the final score awarded to one player."""

        if player is Player.BLACK:
            return self.black_score
        if player is Player.WHITE:
            return self.white_score
        raise TypeError("player must be Player.BLACK or Player.WHITE")


def score_terminal(board: Board) -> ScoreResult:
    """Score every cell of a terminal board under its configured rules."""

    if not board.is_terminal:
        raise NonTerminalBoardError("Official scoring requires a terminal board")
    _validate_scoring_rules(board.rules)

    if is_available():
        native_result = _native_score(board.cells, board.size)
        cell_scores = native_result[2]
        black_score, white_score = native_result[:2]
    else:
        cells = board.cells
        stones = _collect_stones(cells)
        distances = _distance_table(board.size)
        cell_scores = tuple(
            _score_cell(action, board.size, stones, distances)
            for action in range(board.action_size)
        )
        black_score = sum(
            cell.points for cell in cell_scores if cell.owner is Player.BLACK
        )
        white_score = sum(
            cell.points for cell in cell_scores if cell.owner is Player.WHITE
        )
    return ScoreResult(
        ruleset_id=board.rules.ruleset_id,
        zobrist_hash=board.zobrist_hash,
        black_score=black_score,
        white_score=white_score,
        cell_scores=cell_scores,
    )


def score_cell(board: Board, action: int) -> CellScore:
    """Score one cell geometrically, including on a non-terminal preview board."""

    _validate_scoring_rules(board.rules)
    decode_action(action, board.size)
    if is_available():
        return _native_score(board.cells, board.size)[2][action]
    return _score_cell(
        action,
        board.size,
        _collect_stones(board.cells),
        _distance_table(board.size),
    )


def _native_score(
    cells: tuple[int, ...], board_size: int
) -> tuple[int, int, tuple[CellScore, ...]]:
    """Convert the exact native scoring result into the stable Python DTOs."""

    if native_score_cells is None:
        raise RuntimeError("Native scorer was selected but is unavailable")
    result = native_score_cells(cells, board_size)
    cell_scores = tuple(
        CellScore(
            action=action,
            owner=None if owner == 0 else Player(owner),
            points=0 if owner == 0 else 1,
            decisive_distance_squared=(None if distance < 0 else int(distance)),
        )
        for action, (owner, distance) in enumerate(
            zip(result.owners, result.distances, strict=True)
        )
    )
    return int(result.black_score), int(result.white_score), cell_scores


def _score_cell(
    action: int,
    board_size: int,
    stones: tuple[tuple[int, Player], ...],
    distances: tuple[tuple[int, ...], ...],
) -> CellScore:
    layers: dict[int, list[int]] = {}
    for stone_action, player in stones:
        distance_squared = distances[action][stone_action]
        counts = layers.setdefault(distance_squared, [0, 0])
        counts[player.value - 1] += 1

    for distance_squared in sorted(layers):
        black_count, white_count = layers[distance_squared]
        if black_count > white_count:
            return CellScore(
                action=action,
                owner=Player.BLACK,
                points=1,
                decisive_distance_squared=distance_squared,
            )
        if white_count > black_count:
            return CellScore(
                action=action,
                owner=Player.WHITE,
                points=1,
                decisive_distance_squared=distance_squared,
            )

    return CellScore(
        action=action,
        owner=None,
        points=0,
        decisive_distance_squared=None,
    )


def _collect_stones(cells: tuple[int, ...]) -> tuple[tuple[int, Player], ...]:
    return tuple(
        (action, Player(cell)) for action, cell in enumerate(cells) if cell != 0
    )


@lru_cache(maxsize=16)
def _distance_table(board_size: int) -> tuple[tuple[int, ...], ...]:
    """Return exact all-pairs squared Euclidean distances for one board size."""

    coordinates = tuple(divmod(action, board_size) for action in range(board_size**2))
    return tuple(
        tuple(
            (row - stone_row) ** 2 + (column - stone_column) ** 2
            for stone_row, stone_column in coordinates
        )
        for row, column in coordinates
    )


def _validate_scoring_rules(rules: RulesConfig) -> None:
    expected_values = {
        "distance_metric": (rules.distance_metric, "squared-euclidean"),
        "tie_break": (rules.tie_break, "next-distance-layer"),
        "majority_award": (rules.majority_award, "one-point"),
    }
    for field_name, (actual, expected) in expected_values.items():
        if actual != expected:
            raise UnsupportedScoringRuleError(
                f"Unsupported {field_name}: expected {expected!r}, got {actual!r}"
            )
    if not rules.score_occupied_cells:
        raise UnsupportedScoringRuleError(
            "The current scorer requires occupied cells to participate"
        )

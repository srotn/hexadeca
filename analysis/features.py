"""Fast, dependency-free Hexadeca game-record feature extraction.

The public entry points are :func:`parse_game_record`, :func:`analyze_game`,
and :class:`HexadecaFeatureParser`.  The parser accepts the compact browser
record format containing only ``moves`` and ``final_matrix``.  All feature
calculation is performed from the stone locations reconstructed from ``moves``;
``2`` and ``-2`` in the final matrix are territory labels, not stones.

The default lattice phases are explicit and replaceable.  Adjacent kernel
tiles share their boundary, so a kernel of side ``s`` repeats with stride
``s - 1``.  This gives the first ``s - 1`` phase offsets on each axis (9 phases
for the 4x4 kernel and 4 phases for the 3x3 kernel), matching the requested
9/4 lattice counts.  A caller that has a different canonical origin list can
inject :class:`LatticeConfiguration` without changing any feature code.

Intrusion ``I`` is computed solely from the signed final-territory field.
A smoothed blue/orange boundary response is combined with a local response
from real stones surrounded by opposing territory.  The scalar is the
territory-confidence-weighted mean of that shared 16x16 heatmap, so both
outputs are naturally bounded to [0, 1].
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeAlias

BOARD_SIZE = 16
ACTION_SIZE = BOARD_SIZE * BOARD_SIZE
EDGE_BASE = 1.2
# Divergence uses the radius-3 circular support inside a 7x7 bounding box.
# The Gaussian is deliberately truncated at that circle so every contribution
# remains local and the scalar feature and point heatmap share one kernel.
DIVERGENCE_RADIUS = 3
DIVERGENCE_SIGMA = 2.0
DIVERGENCE_NUMERATOR_SCALE = 2.0
# After the period filter, a distant stone can leave a very small positive
# tail.  Treating that tail as density would create huge reciprocal spikes;
# this floor preserves a finite empty-region baseline instead.
DIVERGENCE_DENSITY_EPSILON = 0.1
_DIVERGENCE_KERNEL: tuple[tuple[int, int, float], ...] = tuple(
    (
        row_offset,
        column_offset,
        math.exp(
            -(row_offset * row_offset + column_offset * column_offset)
            / (2.0 * DIVERGENCE_SIGMA * DIVERGENCE_SIGMA)
        ),
    )
    for row_offset in range(-DIVERGENCE_RADIUS, DIVERGENCE_RADIUS + 1)
    for column_offset in range(-DIVERGENCE_RADIUS, DIVERGENCE_RADIUS + 1)
    if row_offset * row_offset + column_offset * column_offset
    <= DIVERGENCE_RADIUS * DIVERGENCE_RADIUS
)
DIVERGENCE_KERNEL_MASS = sum(weight for _, _, weight in _DIVERGENCE_KERNEL)
# Separable finite-difference smoothing/notch filter.  Its frequency
# response is zero at period 2 (the alternating mode) and period 3, while
# preserving the DC component.  Applying it to both density and available
# mass keeps edge correction consistent.
_DIVERGENCE_PERIOD_FILTER: tuple[float, ...] = (
    1.0 / 12.0,
    3.0 / 12.0,
    4.0 / 12.0,
    3.0 / 12.0,
    1.0 / 12.0,
)
# Intrusion deliberately uses tighter kernels than divergence.  A radius-2,
# sigma-1.15 territory kernel keeps borders continuous without turning one
# isolated signed cell into a 7x7 mass.  The still narrower sigma-0.75 piece
# kernel concentrates penetration evidence at the stone and its immediate
# neighbours.
INTRUSION_TERRITORY_RADIUS = 2
INTRUSION_TERRITORY_SIGMA = 1.15
INTRUSION_PIECE_RADIUS = 2
INTRUSION_PIECE_SIGMA = 0.75
_INTRUSION_TERRITORY_KERNEL: tuple[tuple[int, int, float], ...] = tuple(
    (
        row_offset,
        column_offset,
        math.exp(
            -(row_offset * row_offset + column_offset * column_offset)
            / (2.0 * INTRUSION_TERRITORY_SIGMA * INTRUSION_TERRITORY_SIGMA)
        ),
    )
    for row_offset in range(-INTRUSION_TERRITORY_RADIUS, INTRUSION_TERRITORY_RADIUS + 1)
    for column_offset in range(
        -INTRUSION_TERRITORY_RADIUS, INTRUSION_TERRITORY_RADIUS + 1
    )
    if row_offset * row_offset + column_offset * column_offset
    <= INTRUSION_TERRITORY_RADIUS * INTRUSION_TERRITORY_RADIUS
)
_INTRUSION_PIECE_KERNEL: tuple[tuple[int, int, float], ...] = tuple(
    (
        row_offset,
        column_offset,
        math.exp(
            -(row_offset * row_offset + column_offset * column_offset)
            / (2.0 * INTRUSION_PIECE_SIGMA * INTRUSION_PIECE_SIGMA)
        ),
    )
    for row_offset in range(-INTRUSION_PIECE_RADIUS, INTRUSION_PIECE_RADIUS + 1)
    for column_offset in range(-INTRUSION_PIECE_RADIUS, INTRUSION_PIECE_RADIUS + 1)
    if row_offset * row_offset + column_offset * column_offset
    <= INTRUSION_PIECE_RADIUS * INTRUSION_PIECE_RADIUS
)
_ALLOWED_MATRIX_VALUES = frozenset({-2, -1, 0, 1, 2})
_KERNEL_4: tuple[tuple[int, ...], ...] = (
    (1, 0, 0, 1),
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (1, 0, 0, 1),
)
_KERNEL_3: tuple[tuple[int, ...], ...] = (
    (1, 0, 1),
    (0, 0, 0),
    (1, 0, 1),
)
_DEFAULT_SPARSE_ORIGINS = tuple(
    (row, column) for row in range(3) for column in range(3)
)
_DEFAULT_DENSE_ORIGINS = tuple((row, column) for row in range(2) for column in range(2))

Matrix: TypeAlias = tuple[tuple[int, ...], ...]
FloatMatrix: TypeAlias = tuple[tuple[float, ...], ...]
RecordSource: TypeAlias = Mapping[str, Any] | str | Path


class GameRecordError(ValueError):
    """Raised when a compact game record violates the Hexadeca contract."""


def _validate_origins(
    origins: tuple[tuple[int, int], ...], *, kernel_size: int, name: str
) -> None:
    if len(set(origins)) != len(origins):
        raise ValueError(f"{name} must not contain duplicate origins")
    period = kernel_size - 1
    for origin in origins:
        if (
            not isinstance(origin, tuple)
            or len(origin) != 2
            or type(origin[0]) is not int
            or type(origin[1]) is not int
            or not 0 <= origin[0] < period
            or not 0 <= origin[1] < period
        ):
            raise ValueError(
                f"{name} origins must be integer phase pairs in [0, {period - 1}]"
            )


@dataclass(frozen=True, slots=True)
class LatticeConfiguration:
    """The phase origins used by the global lattice scores.

    Origins are phase offsets, not individual window positions.  The periodic
    masks generated from them are always board-sized.  The required 9 and 4
    origin counts are enforced so that the global formulas remain comparable.
    """

    sparse_origins: tuple[tuple[int, int], ...] = _DEFAULT_SPARSE_ORIGINS
    dense_origins: tuple[tuple[int, int], ...] = _DEFAULT_DENSE_ORIGINS

    def __post_init__(self) -> None:
        if len(self.sparse_origins) != 9:
            raise ValueError("sparse_origins must contain exactly 9 origins")
        if len(self.dense_origins) != 4:
            raise ValueError("dense_origins must contain exactly 4 origins")
        _validate_origins(self.sparse_origins, kernel_size=4, name="sparse_origins")
        _validate_origins(self.dense_origins, kernel_size=3, name="dense_origins")


DEFAULT_LATTICE_CONFIGURATION = LatticeConfiguration()


@dataclass(frozen=True, slots=True)
class ParsedGameRecord:
    """Validated compact game record with reconstructed stone locations."""

    moves: tuple[int, ...]
    final_matrix: Matrix
    blue_positions: tuple[int, ...]
    orange_positions: tuple[int, ...]
    terminal: bool
    legal_moves_remaining: int
    board_size: int = BOARD_SIZE

    @property
    def all_positions(self) -> tuple[int, ...]:
        """Return all stone actions in the original move order."""

        return tuple(self.moves)

    @property
    def piece_counts(self) -> tuple[int, int, int]:
        """Return ``(all, blue, orange)`` stone counts."""

        return (
            len(self.moves),
            len(self.blue_positions),
            len(self.orange_positions),
        )


@dataclass(frozen=True, slots=True)
class _Heatmaps:
    sparse: FloatMatrix
    dense: FloatMatrix


@dataclass(frozen=True, slots=True)
class PointHeatmaps:
    """Board-sized point features for visualization and model-side reuse."""

    divergence: FloatMatrix
    intrusion: FloatMatrix


@dataclass(frozen=True, slots=True)
class FeatureResult:
    """All scalar features and retained lattice heatmaps.

    ``gridness`` and ``heatmaps`` are ordered as ``G0, G1, G2``.  Each gridness
    vector is ordered as ``(g1, g2, g3, g4)``.
    """

    divergence: float
    gridness: tuple[tuple[float, float, float, float], ...]
    intrusion: float | None
    symmetry: tuple[float, float, float]
    edge: tuple[float, float, float]
    heatmaps: tuple[_Heatmaps, ...]
    point_heatmaps: PointHeatmaps
    piece_counts: tuple[int, int, int]
    move_count: int
    lattice: LatticeConfiguration = DEFAULT_LATTICE_CONFIGURATION
    board_size: int = BOARD_SIZE

    @property
    def D(self) -> float:
        """Chinese-spec shorthand for divergence."""

        return self.divergence

    @property
    def G0(self) -> tuple[float, float, float, float]:
        return self.gridness[0]

    @property
    def G1(self) -> tuple[float, float, float, float]:
        return self.gridness[1]

    @property
    def G2(self) -> tuple[float, float, float, float]:
        return self.gridness[2]

    @property
    def I(self) -> float | None:
        return self.intrusion

    @property
    def S0(self) -> float:
        return self.symmetry[0]

    @property
    def S1(self) -> float:
        return self.symmetry[1]

    @property
    def S2(self) -> float:
        return self.symmetry[2]

    @property
    def B0(self) -> float:
        return self.edge[0]

    @property
    def B1(self) -> float:
        return self.edge[1]

    @property
    def B2(self) -> float:
        return self.edge[2]

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-compatible analysis payload."""

        def vector(values: Iterable[float]) -> list[float]:
            return [float(value) for value in values]

        return {
            "schema_version": 3,
            "board_size": self.board_size,
            "move_count": self.move_count,
            "piece_counts": {
                "all": self.piece_counts[0],
                "blue": self.piece_counts[1],
                "orange": self.piece_counts[2],
            },
            "lattice": {
                "sparse_origins": [
                    list(origin) for origin in self.lattice.sparse_origins
                ],
                "dense_origins": [
                    list(origin) for origin in self.lattice.dense_origins
                ],
            },
            "features": {
                "D": self.D,
                "G0": vector(self.G0),
                "G1": vector(self.G1),
                "G2": vector(self.G2),
                "I": self.I,
                "S0": self.S0,
                "S1": self.S1,
                "S2": self.S2,
                "B0": self.B0,
                "B1": self.B1,
                "B2": self.B2,
            },
            "heatmaps": {
                f"G{index}": {
                    "sparse": [list(row) for row in heatmap.sparse],
                    "dense": [list(row) for row in heatmap.dense],
                }
                for index, heatmap in enumerate(self.heatmaps)
            },
            "point_heatmaps": {
                "divergence": [list(row) for row in self.point_heatmaps.divergence],
                "intrusion": [list(row) for row in self.point_heatmaps.intrusion],
            },
        }


class HexadecaFeatureParser:
    """Reusable parser/analyzer facade for web and training integrations."""

    __slots__ = ("_board_size", "_lattice")

    def __init__(
        self,
        *,
        lattice: LatticeConfiguration = DEFAULT_LATTICE_CONFIGURATION,
        board_size: int = BOARD_SIZE,
    ) -> None:
        if type(board_size) is not int or board_size < 3:
            raise ValueError("board_size must be an integer >= 3")
        self._board_size = board_size
        self._lattice = lattice

    @property
    def lattice(self) -> LatticeConfiguration:
        """Return the immutable lattice-origin configuration."""

        return self._lattice

    def parse(
        self,
        source: RecordSource,
        *,
        require_terminal: bool = True,
    ) -> ParsedGameRecord:
        return parse_game_record(
            source,
            require_terminal=require_terminal,
            board_size=self._board_size,
        )

    def analyze(
        self,
        source: RecordSource | ParsedGameRecord,
        *,
        require_terminal: bool = True,
    ) -> FeatureResult:
        if isinstance(source, ParsedGameRecord):
            if require_terminal and not source.terminal:
                raise GameRecordError(
                    "The move sequence is not terminal; final_matrix cannot be "
                    "treated as a final matrix"
                )
            record = source
        else:
            record = self.parse(source, require_terminal=require_terminal)
        return _extract_features(record, self._lattice)


def parse_game_record(
    source: RecordSource,
    *,
    require_terminal: bool = True,
    board_size: int = BOARD_SIZE,
) -> ParsedGameRecord:
    """Parse and validate a browser JSON record.

    ``source`` may be a mapping, a JSON string, or a filesystem path.  By
    default the move sequence must reach a terminal position because the
    supplied matrix is explicitly a final matrix.  Set ``require_terminal``
    to ``False`` for live front-end snapshots.
    """

    if type(board_size) is not int or board_size < 3:
        raise ValueError("board_size must be an integer >= 3")
    raw = _load_record_mapping(source)
    moves = _parse_moves(raw.get("moves"), board_size)
    final_matrix = _parse_final_matrix(raw.get("final_matrix"), board_size)
    (
        blue_positions,
        orange_positions,
        reconstructed,
        terminal,
        legal_moves_remaining,
    ) = _reconstruct_moves(moves, board_size)
    _validate_stones_match_matrix(reconstructed, final_matrix, board_size)
    if require_terminal and not terminal:
        raise GameRecordError(
            "The move sequence is not terminal; final_matrix cannot be treated "
            "as a final matrix"
        )
    return ParsedGameRecord(
        moves=moves,
        final_matrix=final_matrix,
        blue_positions=blue_positions,
        orange_positions=orange_positions,
        terminal=terminal,
        legal_moves_remaining=legal_moves_remaining,
        board_size=board_size,
    )


def analyze_game(
    source: RecordSource | ParsedGameRecord,
    *,
    lattice: LatticeConfiguration = DEFAULT_LATTICE_CONFIGURATION,
    require_terminal: bool = True,
    board_size: int = BOARD_SIZE,
) -> FeatureResult:
    """Parse a record and compute all D/G/I/S/B features."""

    parser = HexadecaFeatureParser(lattice=lattice, board_size=board_size)
    return parser.analyze(source, require_terminal=require_terminal)


def _extract_features(
    record: ParsedGameRecord,
    lattice: LatticeConfiguration,
) -> FeatureResult:
    scopes = (
        record.all_positions,
        record.blue_positions,
        record.orange_positions,
    )
    divergence = _divergence(scopes[0], record.board_size)
    intrusion, intrusion_heatmap = _intrusion_analysis(
        scopes[1],
        scopes[2],
        record.final_matrix,
        record.board_size,
    )
    point_heatmaps = PointHeatmaps(
        divergence=_point_divergence_heatmap(scopes[0], record.board_size),
        intrusion=intrusion_heatmap,
    )
    gridness: list[tuple[float, float, float, float]] = []
    heatmaps: list[_Heatmaps] = []
    symmetry: list[float] = []
    edge: list[float] = []
    for positions in scopes:
        g1, g3 = _global_gridness(positions, record.board_size, lattice)
        sparse_heatmap = _local_gridness_heatmap(
            positions,
            record.board_size,
            window_size=6,
            kernel=_KERNEL_4,
            origins=lattice.sparse_origins,
        )
        dense_heatmap = _local_gridness_heatmap(
            positions,
            record.board_size,
            window_size=4,
            kernel=_KERNEL_3,
            origins=lattice.dense_origins,
        )
        gridness.append(
            (
                g1,
                _heatmap_mean(sparse_heatmap, divisor=1),
                g3,
                _heatmap_mean(dense_heatmap, divisor=1),
            )
        )
        heatmaps.append(_Heatmaps(sparse_heatmap, dense_heatmap))
        symmetry.append(_symmetry(positions, record.board_size))
        edge.append(_edge(positions, record.board_size))
    return FeatureResult(
        divergence=divergence,
        gridness=tuple(gridness),
        intrusion=intrusion,
        symmetry=tuple(symmetry),
        edge=tuple(edge),
        heatmaps=tuple(heatmaps),
        point_heatmaps=point_heatmaps,
        piece_counts=record.piece_counts,
        move_count=len(record.moves),
        lattice=lattice,
        board_size=record.board_size,
    )


def _load_record_mapping(source: RecordSource) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    if isinstance(source, Path):
        try:
            return _decode_json(source.read_text(encoding="utf-8"))
        except OSError as error:
            raise GameRecordError(f"Cannot read game record: {source}") from error
    if isinstance(source, str):
        stripped = source.lstrip()
        if stripped.startswith("{"):
            return _decode_json(source)
        path = Path(source)
        try:
            if path.is_file():
                return _decode_json(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise GameRecordError(f"Cannot read game record: {source}") from error
        return _decode_json(source)
    raise TypeError("source must be a mapping, JSON string, or filesystem path")


def _decode_json(payload: str) -> Mapping[str, Any]:
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as error:
        raise GameRecordError(f"Invalid game-record JSON: {error.msg}") from error
    if not isinstance(raw, Mapping):
        raise GameRecordError("Game record root must be a JSON object")
    return raw


def _parse_moves(raw: object, board_size: int = BOARD_SIZE) -> tuple[int, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise GameRecordError("moves must be an array")
    action_size = board_size * board_size
    if len(raw) > action_size:
        raise GameRecordError(f"moves cannot contain more than {action_size} actions")
    moves: list[int] = []
    for index, action in enumerate(raw):
        if type(action) is not int or not 0 <= action < action_size:
            raise GameRecordError(
                f"moves[{index}] must be an integer in [0, {action_size - 1}]"
            )
        moves.append(action)
    return tuple(moves)


def _parse_final_matrix(raw: object, board_size: int) -> Matrix:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise GameRecordError(f"final_matrix must be a {board_size}x{board_size} array")
    if len(raw) != board_size:
        raise GameRecordError(f"final_matrix must have {board_size} rows")
    rows: list[tuple[int, ...]] = []
    for row_index, row in enumerate(raw):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise GameRecordError(f"final_matrix[{row_index}] must be an array")
        if len(row) != board_size:
            raise GameRecordError(
                f"final_matrix[{row_index}] must have {board_size} columns"
            )
        parsed_row: list[int] = []
        for column_index, value in enumerate(row):
            if type(value) is not int or value not in _ALLOWED_MATRIX_VALUES:
                raise GameRecordError(
                    "final_matrix[{}][{}] must be one of -2, -1, 0, 1, 2".format(
                        row_index, column_index
                    )
                )
            parsed_row.append(value)
        rows.append(tuple(parsed_row))
    return tuple(rows)


def _reconstruct_moves(
    moves: tuple[int, ...], board_size: int
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], bool, int]:
    action_size = board_size * board_size
    cells = [0] * action_size
    blocked = [0] * action_size
    legal_count = action_size
    blue: list[int] = []
    orange: list[int] = []
    for ply, action in enumerate(moves):
        if cells[action] != 0 or blocked[action] != 0:
            row, column = divmod(action, board_size)
            raise GameRecordError(
                f"moves[{ply}] is illegal at coordinate ({row}, {column})"
            )
        player = 1 if ply % 2 == 0 else -1
        row, column = divmod(action, board_size)
        for next_row in range(max(0, row - 1), min(board_size, row + 2)):
            base = next_row * board_size
            for next_column in range(max(0, column - 1), min(board_size, column + 2)):
                blocked_action = base + next_column
                if cells[blocked_action] == 0 and blocked[blocked_action] == 0:
                    legal_count -= 1
                blocked[blocked_action] += 1
        cells[action] = player
        (blue if player == 1 else orange).append(action)
    return tuple(blue), tuple(orange), tuple(cells), legal_count == 0, legal_count


def _validate_stones_match_matrix(
    reconstructed: tuple[int, ...], final_matrix: Matrix, board_size: int
) -> None:
    flattened = tuple(value for row in final_matrix for value in row)
    for action, player in enumerate(reconstructed):
        if player == 0:
            continue
        matrix_value = flattened[action]
        expected = player
        if matrix_value != expected:
            row, column = divmod(action, board_size)
            raise GameRecordError(
                "final_matrix does not contain the stone from moves at "
                f"({row}, {column})"
            )
    for action, matrix_value in enumerate(flattened):
        if matrix_value not in {1, -1}:
            continue
        if reconstructed[action] != matrix_value:
            row, column = divmod(action, board_size)
            raise GameRecordError(
                f"final_matrix contains an unexpected stone at ({row}, {column})"
            )


def _divergence(positions: tuple[int, ...], board_size: int) -> float:
    """Mean reciprocal density after Gaussian and period-2/3 filtering.

    A radius-3 circular Gaussian density is first computed over the board.
    The separable ``[1, 3, 4, 3, 1] / 12`` filter is then applied in both
    directions to remove the period-2 and period-3 lattice modes.  The same
    filter is applied to the available Gaussian mass for edge correction.
    """

    count = len(positions)
    if count == 0:
        return 0.0
    density, mass = _divergence_density_fields(positions, board_size)
    filtered_density = _separable_filter(density, _DIVERGENCE_PERIOD_FILTER)
    filtered_mass = _separable_filter(mass, _DIVERGENCE_PERIOD_FILTER)
    total = 0.0
    for action in positions:
        row, column = divmod(action, board_size)
        nearby = filtered_density[row][column]
        window_coefficient = filtered_mass[row][column] / DIVERGENCE_KERNEL_MASS
        total += _divergence_value(nearby, window_coefficient)
    return total / count


def terminal_divergence_reference_bounds(
    board_size: int = BOARD_SIZE,
) -> tuple[float, float]:
    """Return plotting references from the two canonical terminal packings.

    A terminal board is a maximal set under the game's 3x3 blocking rule.
    The regular step-2 packing (64 stones) is the densest legal reference;
    the regular step-3 packing (36 stones) is the sparsest dominating
    reference.  Their measured divergence values provide stable colour-scale
    endpoints without changing the full-precision feature value.
    """

    if type(board_size) is not int or board_size < 3:
        raise ValueError("board_size must be an integer >= 3")
    dense_positions = tuple(
        row * board_size + column
        for row in range(0, board_size, 2)
        for column in range(0, board_size, 2)
    )
    sparse_positions = tuple(
        row * board_size + column
        for row in range(0, board_size, 3)
        for column in range(0, board_size, 3)
    )
    return (
        _divergence(dense_positions, board_size),
        _divergence(sparse_positions, board_size),
    )


def _point_divergence_heatmap(
    positions: tuple[int, ...],
    board_size: int,
) -> FloatMatrix:
    """Evaluate the filtered Gaussian divergence contribution everywhere.

    The same ``2 * (M_c / M0) / R_c`` rule as :func:`_divergence` is used for
    occupied windows; an empty window receives the corresponding scaled edge
    coefficient so the heatmap remains defined over all 256 coordinates.
    """

    density, mass = _divergence_density_fields(positions, board_size)
    filtered_density = _separable_filter(density, _DIVERGENCE_PERIOD_FILTER)
    filtered_mass = _separable_filter(mass, _DIVERGENCE_PERIOD_FILTER)
    result: list[tuple[float, ...]] = []
    for row in range(board_size):
        values: list[float] = []
        for column in range(board_size):
            nearby = filtered_density[row][column]
            window_coefficient = filtered_mass[row][column] / DIVERGENCE_KERNEL_MASS
            values.append(_divergence_value(nearby, window_coefficient))
        result.append(tuple(values))
    return tuple(result)


def _divergence_density_fields(
    positions: tuple[int, ...],
    board_size: int,
) -> tuple[FloatMatrix, FloatMatrix]:
    """Build raw Gaussian density and available-kernel-mass fields."""

    occupied = bytearray(board_size * board_size)
    for action in positions:
        occupied[action] = 1
    density: list[tuple[float, ...]] = []
    mass: list[tuple[float, ...]] = []
    for row in range(board_size):
        density_row: list[float] = []
        mass_row: list[float] = []
        for column in range(board_size):
            weighted_nearby, available_mass = _divergence_window_stats(
                row, column, occupied, board_size
            )
            density_row.append(weighted_nearby)
            mass_row.append(available_mass)
        density.append(tuple(density_row))
        mass.append(tuple(mass_row))
    return tuple(density), tuple(mass)


def _divergence_value(filtered_density: float, window_coefficient: float) -> float:
    """Convert filtered density to a finite, edge-corrected contribution."""

    if filtered_density <= DIVERGENCE_DENSITY_EPSILON:
        return DIVERGENCE_NUMERATOR_SCALE * window_coefficient
    return DIVERGENCE_NUMERATOR_SCALE * window_coefficient / filtered_density


def _divergence_window_stats(
    row: int,
    column: int,
    occupied: Sequence[int],
    board_size: int,
) -> tuple[float, float]:
    """Return weighted occupancy and raw available kernel mass for one point."""

    weighted_nearby = 0.0
    available_mass = 0.0
    for row_offset, column_offset, weight in _DIVERGENCE_KERNEL:
        next_row = row + row_offset
        next_column = column + column_offset
        if not (0 <= next_row < board_size and 0 <= next_column < board_size):
            continue
        available_mass += weight
        weighted_nearby += weight * occupied[next_row * board_size + next_column]
    return weighted_nearby, available_mass


def _separable_filter(
    values: FloatMatrix,
    kernel: tuple[float, ...],
) -> FloatMatrix:
    """Apply a zero-padded separable 2-D filter to a board-sized matrix."""

    if not values:
        return values
    radius = len(kernel) // 2
    height = len(values)
    width = len(values[0])
    horizontal: list[list[float]] = [[0.0] * width for _ in range(height)]
    for row in range(height):
        for column in range(width):
            total = 0.0
            for offset, coefficient in enumerate(kernel):
                source_column = column + offset - radius
                if 0 <= source_column < width:
                    total += coefficient * values[row][source_column]
            horizontal[row][column] = total
    filtered: list[tuple[float, ...]] = []
    for row in range(height):
        output_row: list[float] = []
        for column in range(width):
            total = 0.0
            for offset, coefficient in enumerate(kernel):
                source_row = row + offset - radius
                if 0 <= source_row < height:
                    total += coefficient * horizontal[source_row][column]
            output_row.append(total)
        filtered.append(tuple(output_row))
    return tuple(filtered)


def _intrusion(
    blue_positions: tuple[int, ...],
    orange_positions: tuple[int, ...],
    final_matrix: Matrix,
    board_size: int,
) -> float | None:
    """Return the final-territory intrusion score in ``[0, 1]``."""

    intrusion, _ = _intrusion_analysis(
        blue_positions, orange_positions, final_matrix, board_size
    )
    return intrusion


def _point_intrusion_heatmap(
    blue_positions: tuple[int, ...],
    orange_positions: tuple[int, ...],
    final_matrix: Matrix,
    board_size: int,
) -> FloatMatrix:
    """Return the final-territory intrusion response over all board points."""

    _, heatmap = _intrusion_analysis(
        blue_positions, orange_positions, final_matrix, board_size
    )
    return heatmap


def _intrusion_analysis(
    blue_positions: tuple[int, ...],
    orange_positions: tuple[int, ...],
    final_matrix: Matrix,
    board_size: int,
) -> tuple[float | None, FloatMatrix]:
    """Compute the shared scalar and heatmap intrusion representation.

    ``1``/``2`` are blue territory, ``-1``/``-2`` are orange territory, and
    ``0`` is neutral.  The boundary component is the confidence-weighted
    harmonic two-colour mix ``q * 4bo/(b+o)^2``.  Each real stone additionally
    contributes its local opposing-territory share through a narrow Gaussian;
    independent stone responses are combined as a probabilistic union.
    """

    blue_field, orange_field, confidence = _intrusion_territory_fields(
        final_matrix, board_size
    )
    boundary: list[list[float]] = [[0.0] * board_size for _ in range(board_size)]
    confidence_sum = 0.0
    for row in range(board_size):
        for column in range(board_size):
            blue = blue_field[row][column]
            orange = orange_field[row][column]
            assigned = blue + orange
            local_confidence = confidence[row][column]
            confidence_sum += local_confidence
            if assigned > 0.0:
                boundary[row][column] = _unit_interval(
                    local_confidence * 4.0 * blue * orange / (assigned * assigned)
                )

    # ``survival`` is the complement of the accumulated penetration response.
    # Updating it by splatting each piece kernel costs O(N * 29), instead of
    # checking every stone independently at all 256 output points.
    survival: list[list[float]] = [[1.0] * board_size for _ in range(board_size)]
    stones = tuple((action, 1) for action in blue_positions) + tuple(
        (action, -1) for action in orange_positions
    )
    for action, colour in stones:
        stone_row, stone_column = divmod(action, board_size)
        blue = blue_field[stone_row][stone_column]
        orange = orange_field[stone_row][stone_column]
        assigned = blue + orange
        if assigned <= 0.0:
            continue
        enemy_share = orange / assigned if colour > 0 else blue / assigned
        if enemy_share <= 0.0:
            continue
        for row_offset, column_offset, weight in _INTRUSION_PIECE_KERNEL:
            row = stone_row + row_offset
            column = stone_column + column_offset
            if 0 <= row < board_size and 0 <= column < board_size:
                survival[row][column] *= 1.0 - enemy_share * weight

    heatmap_rows: list[tuple[float, ...]] = []
    weighted_total = 0.0
    for row in range(board_size):
        heatmap_row: list[float] = []
        for column in range(board_size):
            penetration = 1.0 - survival[row][column]
            value = _unit_interval(
                1.0 - (1.0 - boundary[row][column]) * (1.0 - penetration)
            )
            heatmap_row.append(value)
            weighted_total += confidence[row][column] * value
        heatmap_rows.append(tuple(heatmap_row))

    heatmap = tuple(heatmap_rows)
    if confidence_sum <= 0.0:
        return None, heatmap
    return _unit_interval(weighted_total / confidence_sum), heatmap


def _intrusion_territory_fields(
    final_matrix: Matrix,
    board_size: int,
) -> tuple[FloatMatrix, FloatMatrix, FloatMatrix]:
    """Smooth signed final territory and compute local coverage confidence."""

    blue_rows: list[tuple[float, ...]] = []
    orange_rows: list[tuple[float, ...]] = []
    confidence_rows: list[tuple[float, ...]] = []
    for row in range(board_size):
        blue_values: list[float] = []
        orange_values: list[float] = []
        confidence_values: list[float] = []
        for column in range(board_size):
            blue = 0.0
            orange = 0.0
            available_mass = 0.0
            for row_offset, column_offset, weight in _INTRUSION_TERRITORY_KERNEL:
                source_row = row + row_offset
                source_column = column + column_offset
                if not (
                    0 <= source_row < board_size and 0 <= source_column < board_size
                ):
                    continue
                available_mass += weight
                owner = final_matrix[source_row][source_column]
                if owner > 0:
                    blue += weight
                elif owner < 0:
                    orange += weight
            blue_values.append(blue)
            orange_values.append(orange)
            confidence_values.append((blue + orange) / available_mass)
        blue_rows.append(tuple(blue_values))
        orange_rows.append(tuple(orange_values))
        confidence_rows.append(tuple(confidence_values))
    return tuple(blue_rows), tuple(orange_rows), tuple(confidence_rows)


def _unit_interval(value: float) -> float:
    """Clamp floating-point roundoff to the feature's exact range."""

    return min(1.0, max(0.0, value))


def _global_gridness(
    positions: tuple[int, ...],
    board_size: int,
    lattice: LatticeConfiguration,
) -> tuple[float, float]:
    count = len(positions)
    if count == 0:
        return 0.0, 0.0
    sparse_masks = _lattice_masks(board_size, _KERNEL_4, lattice.sparse_origins)
    dense_masks = _lattice_masks(board_size, _KERNEL_3, lattice.dense_origins)
    sparse = _global_grid_score(positions, sparse_masks, count)
    dense = _global_grid_score(positions, dense_masks, count)
    return sparse, dense


def _global_grid_score(
    positions: tuple[int, ...],
    masks: tuple[tuple[int, ...], ...],
    count: int,
) -> float:
    score = 0.0
    for mask in masks:
        lattice_count = sum(mask)
        if lattice_count == 0:
            continue
        matched = sum(mask[action] for action in positions)
        score += (matched * matched) / (lattice_count * count)
    return score


def _local_gridness_heatmap(
    positions: tuple[int, ...],
    board_size: int,
    *,
    window_size: int,
    kernel: tuple[tuple[int, ...], ...],
    origins: tuple[tuple[int, int], ...],
) -> FloatMatrix:
    """Compute global lattice score independently inside each sliding window."""

    occupied = bytearray(board_size * board_size)
    for action in positions:
        occupied[action] = 1
    period = len(kernel) - 1
    active_residues = frozenset(
        (kernel_row % period, kernel_column % period)
        for kernel_row, row_values in enumerate(kernel)
        for kernel_column, value in enumerate(row_values)
        if value
    )
    result: list[tuple[float, ...]] = []
    for top in range(board_size - window_size + 1):
        values: list[float] = []
        for left in range(board_size - window_size + 1):
            window_positions = [
                (row - top) * window_size + (column - left)
                for row in range(top, top + window_size)
                for column in range(left, left + window_size)
                if occupied[row * board_size + column]
            ]
            count = len(window_positions)
            if count == 0:
                values.append(0.0)
                continue
            score = 0.0
            for origin_row, origin_column in origins:
                lattice_count = 0
                matched = 0
                for row in range(window_size):
                    for column in range(window_size):
                        active = (
                            (row - origin_row) % period,
                            (column - origin_column) % period,
                        ) in active_residues
                        if active:
                            lattice_count += 1
                            if occupied[(top + row) * board_size + left + column]:
                                matched += 1
                if lattice_count:
                    score += (matched * matched) / (lattice_count * count)
            values.append(score)
        result.append(tuple(values))
    return tuple(result)


def _heatmap_mean(
    heatmap: FloatMatrix,
    *,
    divisor: int,
) -> float:
    area = len(heatmap) * len(heatmap[0]) if heatmap else 0
    if area == 0:
        return 0.0
    return sum(sum(row) for row in heatmap) / (area * divisor)


def _symmetry(positions: tuple[int, ...], board_size: int) -> float:
    count = len(positions)
    if count == 0:
        return 0.0
    occupied = [False] * (board_size * board_size)
    for action in positions:
        occupied[action] = True
    overlap_total = 0
    for transform in _symmetry_transforms(board_size):
        overlap_total += sum(
            1
            for action, present in enumerate(occupied)
            if present and occupied[transform[action]]
        )
    return (overlap_total * overlap_total) / (25.0 * count * count)


def _edge(positions: tuple[int, ...], board_size: int) -> float:
    """Return edge degree using the current exponential base 1.2."""

    boundary_count = 0
    for action in positions:
        row, column = divmod(action, board_size)
        if row in {0, board_size - 1} or column in {0, board_size - 1}:
            boundary_count += 1
    return 1.0 - EDGE_BASE ** (-boundary_count) if boundary_count else 0.0


@lru_cache(maxsize=1)
def _symmetry_transforms(board_size: int) -> tuple[tuple[int, ...], ...]:
    return (
        tuple(
            (board_size - 1 - row) * board_size + (board_size - 1 - column)
            for row in range(board_size)
            for column in range(board_size)
        ),
        tuple(
            row * board_size + (board_size - 1 - column)
            for row in range(board_size)
            for column in range(board_size)
        ),
        tuple(
            (board_size - 1 - row) * board_size + column
            for row in range(board_size)
            for column in range(board_size)
        ),
        tuple(
            column * board_size + row
            for row in range(board_size)
            for column in range(board_size)
        ),
        tuple(
            (board_size - 1 - column) * board_size + (board_size - 1 - row)
            for row in range(board_size)
            for column in range(board_size)
        ),
    )


@lru_cache(maxsize=8)
def _lattice_masks(
    board_size: int,
    kernel: tuple[tuple[int, ...], ...],
    origins: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, ...], ...]:
    kernel_size = len(kernel)
    period = kernel_size - 1
    occupied_residues = frozenset(
        (kernel_row % period, kernel_column % period)
        for kernel_row, row_values in enumerate(kernel)
        for kernel_column, value in enumerate(row_values)
        if value
    )
    masks: list[tuple[int, ...]] = []
    for origin_row, origin_column in origins:
        mask = tuple(
            int(
                ((row - origin_row) % period, (column - origin_column) % period)
                in occupied_residues
            )
            for row in range(board_size)
            for column in range(board_size)
        )
        masks.append(mask)
    return tuple(masks)

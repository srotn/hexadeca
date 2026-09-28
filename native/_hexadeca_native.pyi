"""Static interface for the compiled Hexadeca C++20 extension."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

class NativeScore:
    """Exact native terminal scoring data."""

    black_score: int
    white_score: int
    owners: list[int]
    distances: list[int]

class NativeBoard:
    """Mutable native board state."""

    def __init__(
        self, board_size: int, neighborhood_radius: int, zobrist_seed: int
    ) -> None: ...

    size: int
    action_size: int
    ply: int
    to_play: int
    legal_count: int
    is_terminal: bool
    zobrist_hash: int
    cells: list[int]
    history: list[int]

    def is_legal(self, action: int) -> bool: ...
    def piece_at(self, action: int) -> int: ...
    def legal_actions(self) -> list[int]: ...
    def legal_mask(self) -> list[bool]: ...
    def apply(self, action: int) -> int: ...
    def undo(self) -> int: ...
    def score(self) -> NativeScore: ...

class NativeSearchTree:
    """Native PUCT tree with Python-owned inference boundaries.

    Tree traversal, board snapshots, and backup release the Python GIL while
    they execute.  Python sequence conversion and result materialization stay
    on the interpreter side of the boundary.
    """

    def __init__(
        self,
        board_size: int,
        neighborhood_radius: int,
        zobrist_seed: int,
        root_actions: Sequence[int],
        c_puct: float,
        fpu_reduction: float,
        virtual_loss: int,
    ) -> None: ...
    def initialize_root(self, policy: Sequence[float]) -> None: ...
    def select_batch(
        self, maximum_leaves: int, remaining_simulations: int
    ) -> dict[str, Any]: ...
    def leaf_state(self, leaf_id: int) -> dict[str, Any]: ...
    def leaf_states(self, leaf_ids: Sequence[int]) -> list[dict[str, Any]]: ...
    def leaf_packed_states(self, leaf_ids: Sequence[int]) -> bytes: ...
    def commit(self, leaf_id: int, policy: Sequence[float], value: float) -> None: ...
    def commit_batch(
        self,
        leaf_ids: Sequence[int],
        policies: Sequence[Sequence[float]],
        values: Sequence[float],
    ) -> None: ...
    def cancel(self, leaf_ids: Sequence[int]) -> None: ...
    def root_statistics(self) -> dict[str, Any]: ...
    def export_nodes(self) -> list[dict[str, Any]]: ...

def score_cells(values: Sequence[int], board_size: int) -> NativeScore: ...
def compute_handcrafted_features(
    current_cells: Sequence[Sequence[int]],
    previous_cells: Sequence[Sequence[int]],
    board_size: int,
) -> dict[str, Any]: ...
def old_champion_choose(
    cells: Sequence[int],
    legal_actions: Sequence[int],
    board_size: int,
    color: int,
    interior_diagonal: float,
    edge_diagonal: float,
    corner_diagonal: float,
    opponent_reply_coefficient: float,
    reply_top_k: int,
    reply_max_share: float,
) -> dict[str, Any]: ...
def old_champion_generate_games(
    games: int,
    board_size: int,
    neighborhood_radius: int,
    zobrist_seed: int,
    random_seed: int,
    interior_diagonal: float,
    edge_diagonal: float,
    corner_diagonal: float,
    opponent_reply_coefficient: float,
    reply_top_k: int,
    reply_max_share: float,
) -> list[dict[str, Any]]: ...
def encode_features(
    states: Sequence[Any],
    board_size: int,
    input_planes: int,
    ruleset_id: str,
) -> Any: ...
def encode_packed_features(
    packed_states: bytes,
    board_size: int,
    input_planes: int,
) -> Any: ...
def pack_compact_states(
    states: Sequence[Any],
    board_size: int,
    ruleset_id: str,
) -> bytes: ...
def pack_compact_evaluations(evaluations: Sequence[Any], action_size: int) -> bytes: ...
def unpack_compact_evaluations(
    packed_evaluations: bytes, action_size: int
) -> list[tuple[tuple[float, ...], float]]: ...

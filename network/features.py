"""Versioned conversion from immutable game snapshots to NCHW tensors."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from game import GameState, Player
from network.errors import FeatureEncodingError
from network.specification import NetworkSpecification

HISTORY_POSITIONS = 4


def encode_state(
    state: GameState,
    specification: NetworkSpecification,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Encode one validated state as a ``[16, H, W]`` float32 tensor.

    Planes 0-3 contain current Black, White, legal, and side-to-play data.
    Planes 4-11 contain Black/White occupancy before each of the latest four
    moves. Planes 12-13 mark each colour's latest move, and 14-15 contain row
    and column coordinates normalized to ``[-1, 1]``.
    """

    specification.validate()
    _validate_state(state, specification)
    board_size = specification.board_size
    features = torch.zeros(
        (specification.input_planes, board_size, board_size),
        dtype=torch.float32,
        device=device,
    )
    cells = torch.tensor(state.cells, dtype=torch.int64, device=device).reshape(
        board_size, board_size
    )
    features[0] = cells == Player.BLACK.value
    features[1] = cells == Player.WHITE.value
    features[2] = torch.tensor(
        state.legal_mask, dtype=torch.bool, device=device
    ).reshape(board_size, board_size)
    if state.to_play is Player.BLACK:
        features[3].fill_(1.0)

    historical_cells = cells.clone()
    for history_offset in range(HISTORY_POSITIONS):
        if history_offset < len(state.history):
            move = state.history[-history_offset - 1]
            historical_cells[move.row, move.column] = 0
        plane_offset = 4 + history_offset * 2
        features[plane_offset] = historical_cells == Player.BLACK.value
        features[plane_offset + 1] = historical_cells == Player.WHITE.value

    seen_colours: set[Player] = set()
    for move in reversed(state.history):
        if move.player in seen_colours:
            continue
        plane = 12 if move.player is Player.BLACK else 13
        features[plane, move.row, move.column] = 1.0
        seen_colours.add(move.player)
        if len(seen_colours) == 2:
            break

    coordinates = torch.linspace(
        -1.0, 1.0, board_size, dtype=torch.float32, device=device
    )
    features[14] = coordinates[:, None].expand(board_size, board_size)
    features[15] = coordinates[None, :].expand(board_size, board_size)
    return features


def encode_batch(
    states: Sequence[GameState],
    specification: NetworkSpecification,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Encode a non-empty state sequence as one NCHW float32 tensor."""

    if not states:
        raise FeatureEncodingError("Cannot encode an empty state batch")
    return torch.stack(
        [encode_state(state, specification, device=device) for state in states]
    )


def _validate_state(state: GameState, specification: NetworkSpecification) -> None:
    if state.ruleset_id != specification.ruleset_id:
        raise FeatureEncodingError(
            "State ruleset does not match the network specification"
        )
    if state.board_size != specification.board_size:
        raise FeatureEncodingError(
            "State board size does not match the network specification"
        )
    if len(state.cells) != specification.policy_size:
        raise FeatureEncodingError("State cell count does not match policy_size")
    if len(state.legal_mask) != specification.policy_size:
        raise FeatureEncodingError("State legal mask does not match policy_size")
    if state.ply != len(state.history):
        raise FeatureEncodingError("State ply does not match its history length")
    if any(type(value) is not int or value not in {0, 1, 2} for value in state.cells):
        raise FeatureEncodingError("State cells must use encodings 0, 1, and 2")
    if any(type(value) is not bool for value in state.legal_mask):
        raise FeatureEncodingError("State legal mask values must be booleans")

    reconstructed = [0] * specification.policy_size
    expected_player = Player.BLACK
    for move in state.history:
        if move.player is not expected_player:
            raise FeatureEncodingError("State history does not alternate players")
        if not 0 <= move.action < specification.policy_size:
            raise FeatureEncodingError("State history contains an invalid action")
        if divmod(move.action, specification.board_size) != (
            move.row,
            move.column,
        ):
            raise FeatureEncodingError("State history coordinates do not match action")
        if reconstructed[move.action] != 0:
            raise FeatureEncodingError("State history places on an occupied action")
        reconstructed[move.action] = move.player.value
        expected_player = expected_player.opponent

    if tuple(reconstructed) != state.cells:
        raise FeatureEncodingError("State cells do not match its move history")
    if state.to_play is not expected_player:
        raise FeatureEncodingError("State side to play does not match its history")

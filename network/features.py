"""Versioned conversion from immutable game snapshots to NCHW tensors."""

from __future__ import annotations

from collections.abc import Sequence
from struct import pack, unpack

import torch

from game import GameState, Player
from native import encode_features as native_encode_features
from native import encode_packed_features as native_encode_packed_features
from native import is_available
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

    if is_available():
        return _encode_native_batch((state,), specification, device=device)[0]
    return _encode_state_reference(state, specification, device=device)


def _encode_state_reference(
    state: GameState,
    specification: NetworkSpecification,
    *,
    device: torch.device | str | None,
) -> torch.Tensor:
    """Encode one state through the retained Python reference implementation."""

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

    coordinates = torch.tensor(
        _canonical_coordinates(board_size),
        dtype=torch.float32,
        device=device,
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
    if is_available():
        return _encode_native_batch(states, specification, device=device)
    return _encode_reference_batch(states, specification, device=device)


def encode_batch_gpu(
    states: Sequence[GameState],
    specification: NetworkSpecification,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Encode a batch with dense feature construction on CUDA.

    Native MCTS still owns the authoritative board snapshots.  Once a leaf
    batch reaches the evaluator, however, all 16 feature planes, history
    removal, latest-move markers, and legal masks are built with batched Torch
    kernels on the target device instead of running a per-state C++ encoder and
    then copying its NumPy result.  This is the GPU portion of the hybrid
    native MCTS path.
    """

    if not states:
        raise FeatureEncodingError("Cannot encode an empty state batch")
    specification.validate()
    target = torch.device(device)
    if target.type != "cuda":
        raise FeatureEncodingError("GPU feature encoding requires a CUDA device")
    board_size = specification.board_size
    action_size = specification.policy_size
    if any(
        state.ruleset_id != specification.ruleset_id
        or state.board_size != board_size
        or len(state.cells) != action_size
        or len(state.legal_mask) != action_size
        for state in states
    ):
        raise FeatureEncodingError("State dimensions do not match the specification")

    # Construct the compact integer inputs once and let Torch perform the
    # remaining work on CUDA.  ``torch.tensor(..., device=target)`` performs a
    # single contiguous host-to-device transfer per field.
    cells = torch.tensor(
        [state.cells for state in states], dtype=torch.int64, device=target
    ).reshape(-1, board_size, board_size)
    legal = torch.tensor(
        [state.legal_mask for state in states], dtype=torch.bool, device=target
    ).reshape(-1, board_size, board_size)
    batch_size = cells.shape[0]
    features = torch.zeros(
        (batch_size, specification.input_planes, board_size, board_size),
        dtype=torch.float32,
        device=target,
    )
    features[:, 0] = cells == Player.BLACK.value
    features[:, 1] = cells == Player.WHITE.value
    features[:, 2] = legal
    black_to_play = torch.tensor(
        [state.to_play is Player.BLACK for state in states],
        dtype=torch.bool,
        device=target,
    )
    features[:, 3] = black_to_play[:, None, None]

    historical = cells.clone()
    for history_offset in range(HISTORY_POSITIONS):
        actions = [
            state.history[-history_offset - 1].action
            if history_offset < len(state.history)
            else -1
            for state in states
        ]
        action_tensor = torch.tensor(actions, dtype=torch.long, device=target)
        valid = action_tensor >= 0
        if bool(valid.any().item()):
            batch_indices = torch.arange(batch_size, device=target)[valid]
            rows = torch.div(action_tensor[valid], board_size, rounding_mode="floor")
            columns = torch.remainder(action_tensor[valid], board_size)
            historical[batch_indices, rows, columns] = 0
        plane_offset = 4 + history_offset * 2
        features[:, plane_offset] = historical == Player.BLACK.value
        features[:, plane_offset + 1] = historical == Player.WHITE.value

    # Mark the latest move by each colour.  The history is at most 400 moves,
    # but only two scatter operations are needed per batch to place the marker.
    latest_black_actions: list[int] = []
    latest_white_actions: list[int] = []
    for state in states:
        black_action = -1
        white_action = -1
        for move in reversed(state.history):
            if move.player is Player.BLACK and black_action < 0:
                black_action = move.action
            elif move.player is Player.WHITE and white_action < 0:
                white_action = move.action
            if black_action >= 0 and white_action >= 0:
                break
        latest_black_actions.append(black_action)
        latest_white_actions.append(white_action)
    latest_black = torch.tensor(latest_black_actions, dtype=torch.long, device=target)
    latest_white = torch.tensor(latest_white_actions, dtype=torch.long, device=target)
    rows_black = torch.div(latest_black.clamp_min(0), board_size, rounding_mode="floor")
    cols_black = torch.remainder(latest_black.clamp_min(0), board_size)
    rows_white = torch.div(latest_white.clamp_min(0), board_size, rounding_mode="floor")
    cols_white = torch.remainder(latest_white.clamp_min(0), board_size)
    batch_indices = torch.arange(batch_size, device=target)
    valid_black = latest_black >= 0
    valid_white = latest_white >= 0
    features[
        batch_indices[valid_black], 12, rows_black[valid_black], cols_black[valid_black]
    ] = 1.0
    features[
        batch_indices[valid_white], 13, rows_white[valid_white], cols_white[valid_white]
    ] = 1.0

    # Upload the canonical float32 coordinates rather than calling
    # ``linspace`` on CUDA: the latter can round a few interior values one ULP
    # differently from the reference/native encoder.
    coordinates = torch.tensor(
        _canonical_coordinates(board_size), dtype=torch.float32, device=target
    )
    features[:, 14] = coordinates[None, :, None]
    features[:, 15] = coordinates[None, None, :]
    return features


def encode_packed_batch(
    payload: bytes,
    specification: NetworkSpecification,
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Encode compact self-play transport data through the native kernel."""

    specification.validate()
    if native_encode_packed_features is None:
        raise FeatureEncodingError("Native packed feature encoder is unavailable")
    try:
        encoded = native_encode_packed_features(
            payload,
            specification.board_size,
            specification.input_planes,
        )
    except (IndexError, TypeError, ValueError) as error:
        raise FeatureEncodingError(str(error)) from error
    features = torch.from_numpy(encoded)
    return features.to(device=device) if device is not None else features


def _encode_reference_batch(
    states: Sequence[GameState],
    specification: NetworkSpecification,
    *,
    device: torch.device | str | None,
) -> torch.Tensor:
    """Encode a batch through the retained Python reference implementation."""

    return torch.stack(
        [
            _encode_state_reference(state, specification, device=device)
            for state in states
        ]
    )


def _encode_native_batch(
    states: Sequence[GameState],
    specification: NetworkSpecification,
    *,
    device: torch.device | str | None,
) -> torch.Tensor:
    """Encode a validated batch through the C++ kernel and transfer it once."""

    specification.validate()
    if native_encode_features is None:
        raise FeatureEncodingError("Native feature encoder is unavailable")
    try:
        encoded = native_encode_features(
            states,
            specification.board_size,
            specification.input_planes,
            specification.ruleset_id,
        )
    except (IndexError, TypeError, ValueError) as error:
        raise FeatureEncodingError(str(error)) from error
    features = torch.from_numpy(encoded)
    return features.to(device=device) if device is not None else features


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


def _canonical_coordinates(board_size: int) -> tuple[float, ...]:
    """Return C++-identical IEEE-754 coordinate values for every board axis."""

    if board_size == 1:
        return (-1.0,)
    coordinate_step = _float32(2.0 / float(board_size - 1))
    coordinates: list[float] = []
    for index in range(board_size):
        if index < board_size // 2:
            coordinate = _float32(-1.0 + _float32(coordinate_step * index))
        else:
            coordinate = _float32(
                1.0 - _float32(coordinate_step * (board_size - index - 1))
            )
        coordinates.append(coordinate)
    return tuple(coordinates)


def _float32(value: float) -> float:
    """Round one finite Python float through the portable IEEE-754 float32 form."""

    return unpack("<f", pack("<f", value))[0]

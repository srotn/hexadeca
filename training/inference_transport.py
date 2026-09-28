"""Versioned compact payloads for self-play inference process boundaries.

The self-play worker owns MCTS state while the coordinator owns the neural
model.  Sending nested :class:`game.GameState` objects through a process queue
serializes hundreds of Python objects for every leaf.  This module defines a
small binary protocol that preserves every field needed by feature encoding
and by the retained object-transport fallback.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from struct import Struct

from game import GameState, Move, Player
from mcts import Evaluation
from native import pack_compact_evaluations as native_pack_compact_evaluations
from native import pack_compact_states as native_pack_compact_states
from native import unpack_compact_evaluations as native_unpack_compact_evaluations

# Version 1 stores one byte per history action. Version 2 uses little-endian
# uint16 actions for forward-compatible board schemas without changing the
# outer packet framing.
_PACKET_VERSION = 2
_LEGACY_PACKET_VERSION = 1
_BATCH_HEADER = Struct("<H")
_RECORD_HEADER = Struct("<BBHQ")
_RECORD_LENGTH = Struct("<H")
_FLOAT32 = Struct("<f")


class InferenceTransportError(ValueError):
    """Raised when compact inference data is malformed or incompatible."""


def pack_state_batch(states: Sequence[GameState]) -> bytes:
    """Encode non-empty immutable game states into one versioned byte packet."""

    if not states:
        raise InferenceTransportError("Cannot pack an empty inference batch")
    if len(states) > 65_535:
        raise InferenceTransportError("Inference batch exceeds compact protocol limit")
    # The native extension selects the compact packet width from the validated
    # board schema while preserving the same outer packet framing.
    if native_pack_compact_states is not None:
        try:
            return bytes(
                native_pack_compact_states(
                    states, states[0].board_size, states[0].ruleset_id
                )
            )
        except (TypeError, ValueError) as error:
            raise InferenceTransportError(str(error)) from error

    packet = bytearray(_BATCH_HEADER.pack(len(states)))
    for state in states:
        record = pack_state(state)
        if len(record) > 65_535:
            raise InferenceTransportError("Compact state record exceeds protocol limit")
        packet.extend(_RECORD_LENGTH.pack(len(record)))
        packet.extend(record)
    return bytes(packet)


def combine_state_batches(payloads: Sequence[bytes]) -> bytes:
    """Merge validated compact batches without decoding individual states."""

    if not payloads:
        raise InferenceTransportError("Cannot combine empty inference batches")
    records = bytearray()
    total_count = 0
    for payload in payloads:
        if len(payload) < _BATCH_HEADER.size:
            raise InferenceTransportError("Compact inference packet is truncated")
        count = _BATCH_HEADER.unpack_from(payload)[0]
        if count == 0:
            raise InferenceTransportError("Compact inference packet is empty")
        offset = _BATCH_HEADER.size
        for _ in range(count):
            if len(payload) < offset + _RECORD_LENGTH.size:
                raise InferenceTransportError("Compact state length is truncated")
            record_length = _RECORD_LENGTH.unpack_from(payload, offset)[0]
            end = offset + _RECORD_LENGTH.size + record_length
            if len(payload) < end:
                raise InferenceTransportError("Compact state record is truncated")
            records.extend(payload[offset:end])
            offset = end
        if offset != len(payload):
            raise InferenceTransportError("Compact inference packet has trailing data")
        total_count += count
    if total_count > 65_535:
        raise InferenceTransportError("Combined inference batch exceeds protocol limit")
    return _BATCH_HEADER.pack(total_count) + bytes(records)


def pack_state(state: GameState) -> bytes:
    """Encode one state while validating the compact protocol invariants."""

    action_size = state.action_size
    if action_size <= 0 or action_size > 65_535:
        raise InferenceTransportError("Compact state action size is unsupported")
    if state.ply != len(state.history):
        raise InferenceTransportError("State ply does not match state history")
    if len(state.legal_mask) != action_size:
        raise InferenceTransportError("State legal mask length is invalid")
    if state.to_play not in {Player.BLACK, Player.WHITE}:
        raise InferenceTransportError("State player is invalid")
    if any(type(cell) is not int or cell not in {0, 1, 2} for cell in state.cells):
        raise InferenceTransportError("State cells must use encodings 0, 1, and 2")
    if any(type(legal) is not bool for legal in state.legal_mask):
        raise InferenceTransportError("State legal mask must contain bool values")
    if state.ply > action_size:
        raise InferenceTransportError("State history exceeds board capacity")

    history_actions: list[int] = []
    expected_player = Player.BLACK
    reconstructed = [0] * action_size
    for move in state.history:
        if move.player is not expected_player:
            raise InferenceTransportError("State history does not alternate players")
        if not 0 <= move.action < action_size:
            raise InferenceTransportError("State history contains an invalid action")
        if move.row != move.action // state.board_size or move.column != (
            move.action % state.board_size
        ):
            raise InferenceTransportError("State history coordinates are invalid")
        if reconstructed[move.action] != 0:
            raise InferenceTransportError("State history contains duplicate actions")
        reconstructed[move.action] = move.player.value
        history_actions.append(move.action)
        expected_player = expected_player.opponent
    if tuple(reconstructed) != state.cells:
        raise InferenceTransportError("State cells do not match state history")
    if state.to_play is not expected_player:
        raise InferenceTransportError("State player does not match state history")

    legal_bits = bytearray((action_size + 7) // 8)
    for action, legal in enumerate(state.legal_mask):
        if legal:
            legal_bits[action // 8] |= 1 << (action % 8)

    if action_size == 256:
        # Keep the byte-for-byte legacy representation for 16x16 so existing
        # native consumers and golden packets remain valid.
        version = _LEGACY_PACKET_VERSION
        history = bytes(history_actions)
    else:
        version = _PACKET_VERSION
        history = b"".join(
            int(action).to_bytes(2, byteorder="little", signed=False)
            for action in history_actions
        )
    return b"".join(
        (
            _RECORD_HEADER.pack(
                version, state.to_play.value, state.ply, state.zobrist_hash
            ),
            bytes(state.cells),
            bytes(legal_bits),
            history,
        )
    )


def unpack_state_batch(
    payload: bytes,
    *,
    ruleset_id: str,
    board_size: int,
) -> tuple[GameState, ...]:
    """Decode and validate a compact state packet for a legacy evaluator."""

    if type(board_size) is not int or board_size <= 0 or board_size > 255:
        raise InferenceTransportError("Compact board size is unsupported")
    if len(payload) < _BATCH_HEADER.size:
        raise InferenceTransportError("Compact inference packet is truncated")
    count = _BATCH_HEADER.unpack_from(payload)[0]
    if count == 0:
        raise InferenceTransportError("Compact inference packet is empty")

    offset = _BATCH_HEADER.size
    states: list[GameState] = []
    for _ in range(count):
        if len(payload) < offset + _RECORD_LENGTH.size:
            raise InferenceTransportError("Compact state length is truncated")
        record_length = _RECORD_LENGTH.unpack_from(payload, offset)[0]
        offset += _RECORD_LENGTH.size
        end = offset + record_length
        if len(payload) < end:
            raise InferenceTransportError("Compact state record is truncated")
        states.append(
            _unpack_state(
                payload[offset:end], ruleset_id=ruleset_id, board_size=board_size
            )
        )
        offset = end
    if offset != len(payload):
        raise InferenceTransportError("Compact inference packet has trailing data")
    return tuple(states)


def pack_evaluations(evaluations: Sequence[Evaluation], *, action_size: int) -> bytes:
    """Encode policy/value results without serializing ``Evaluation`` objects."""

    if not evaluations:
        raise InferenceTransportError("Cannot pack empty inference evaluations")
    if len(evaluations) > 65_535:
        raise InferenceTransportError("Evaluation batch exceeds compact protocol limit")
    if native_pack_compact_evaluations is not None:
        try:
            return bytes(native_pack_compact_evaluations(evaluations, action_size))
        except (TypeError, ValueError) as error:
            raise InferenceTransportError(str(error)) from error
    packet = bytearray(_BATCH_HEADER.pack(len(evaluations)))
    policy_format = f"<{action_size + 1}f"
    for evaluation in evaluations:
        if len(evaluation.policy) != action_size:
            raise InferenceTransportError("Evaluation policy has an invalid length")
        values = (*evaluation.policy, evaluation.value)
        if not all(isfinite(value) for value in values):
            raise InferenceTransportError("Evaluation values must be finite")
        packet.extend(Struct(policy_format).pack(*values))
    return bytes(packet)


def unpack_evaluations(payload: bytes, *, action_size: int) -> tuple[Evaluation, ...]:
    """Decode exactly ordered policy/value results from a compact response."""

    if len(payload) < _BATCH_HEADER.size:
        raise InferenceTransportError("Compact evaluation packet is truncated")
    if native_unpack_compact_evaluations is not None:
        try:
            decoded = native_unpack_compact_evaluations(payload, action_size)
        except (TypeError, ValueError) as error:
            raise InferenceTransportError(str(error)) from error
        return tuple(
            Evaluation(policy=tuple(policy), value=value) for policy, value in decoded
        )
    count = _BATCH_HEADER.unpack_from(payload)[0]
    if count == 0:
        raise InferenceTransportError("Compact evaluation packet is empty")
    value_format = Struct(f"<{action_size + 1}f")
    expected_size = _BATCH_HEADER.size + count * value_format.size
    if len(payload) != expected_size:
        raise InferenceTransportError("Compact evaluation packet length is invalid")
    evaluations: list[Evaluation] = []
    offset = _BATCH_HEADER.size
    for _ in range(count):
        values = value_format.unpack_from(payload, offset)
        offset += value_format.size
        policy = values[:-1]
        value = values[-1]
        if not all(isfinite(item) for item in values):
            raise InferenceTransportError("Compact evaluation packet is non-finite")
        evaluations.append(Evaluation(policy=policy, value=value))
    return tuple(evaluations)


def _unpack_state(record: bytes, *, ruleset_id: str, board_size: int) -> GameState:
    action_size = board_size * board_size
    legal_bytes = (action_size + 7) // 8
    minimum_size = _RECORD_HEADER.size + action_size + legal_bytes
    if len(record) < minimum_size:
        raise InferenceTransportError("Compact state record is truncated")
    version, raw_player, ply, zobrist_hash = _RECORD_HEADER.unpack_from(record)
    if version not in {_LEGACY_PACKET_VERSION, _PACKET_VERSION}:
        raise InferenceTransportError("Compact state protocol version is unsupported")
    try:
        to_play = Player(raw_player)
    except ValueError as error:
        raise InferenceTransportError("Compact state player is invalid") from error
    history_width = 1 if version == _LEGACY_PACKET_VERSION else 2
    expected_history_bytes = ply * history_width
    if ply > action_size or len(record) != minimum_size + expected_history_bytes:
        raise InferenceTransportError("Compact state history length is invalid")

    cells_offset = _RECORD_HEADER.size
    cells = tuple(record[cells_offset : cells_offset + action_size])
    if any(cell not in {0, 1, 2} for cell in cells):
        raise InferenceTransportError("Compact state cells are invalid")
    legal_offset = cells_offset + action_size
    legal_payload = record[legal_offset : legal_offset + legal_bytes]
    legal_mask = tuple(
        bool(legal_payload[action // 8] & (1 << (action % 8)))
        for action in range(action_size)
    )
    if action_size % 8 and legal_payload[-1] >> (action_size % 8):
        raise InferenceTransportError("Compact state legal mask has invalid padding")

    history_offset = legal_offset + legal_bytes
    raw_actions = record[history_offset:]
    if history_width == 1:
        actions = tuple(raw_actions)
    else:
        actions = tuple(
            int.from_bytes(raw_actions[index : index + 2], "little", signed=False)
            for index in range(0, len(raw_actions), 2)
        )
    if any(action >= action_size for action in actions):
        raise InferenceTransportError(
            "Compact state history contains an invalid action"
        )
    reconstructed = [0] * action_size
    player = Player.BLACK
    history: list[Move] = []
    for action in actions:
        if reconstructed[action] != 0:
            raise InferenceTransportError("Compact state history contains duplicates")
        reconstructed[action] = player.value
        history.append(
            Move(
                action=action,
                row=action // board_size,
                column=action % board_size,
                player=player,
            )
        )
        player = player.opponent
    if tuple(reconstructed) != cells:
        raise InferenceTransportError("Compact state cells do not match history")
    if to_play is not player:
        raise InferenceTransportError("Compact state player does not match history")
    return GameState(
        ruleset_id=ruleset_id,
        board_size=board_size,
        cells=cells,
        to_play=to_play,
        ply=ply,
        history=tuple(history),
        legal_mask=legal_mask,
        zobrist_hash=zobrist_hash,
        terminal=not any(legal_mask),
    )


__all__ = [
    "InferenceTransportError",
    "combine_state_batches",
    "pack_evaluations",
    "pack_state",
    "pack_state_batch",
    "unpack_evaluations",
    "unpack_state_batch",
]

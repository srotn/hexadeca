"""Compiled C++20 acceleration surface with explicit availability reporting."""

from __future__ import annotations

from typing import Any, Final

_IMPORT_ERROR: ImportError | None
NativeBoard: Any
NativeSearchTree: Any
encode_features: Any
encode_packed_features: Any
pack_compact_evaluations: Any
pack_compact_states: Any
score_cells: Any
compute_handcrafted_features: Any
decode_packed_hybrid_inputs: Any
unpack_compact_evaluations: Any
old_champion_choose: Any
old_champion_generate_games: Any

try:
    from native import _hexadeca_native as _implementation
except ImportError as error:
    _IMPORT_ERROR = error
    NativeBoard = None
    NativeSearchTree = None
    encode_features = None
    encode_packed_features = None
    pack_compact_evaluations = None
    pack_compact_states = None
    score_cells = None
    compute_handcrafted_features = None
    decode_packed_hybrid_inputs = None
    unpack_compact_evaluations = None
    old_champion_choose = None
    old_champion_generate_games = None
else:
    NativeBoard = _implementation.NativeBoard
    NativeSearchTree = _implementation.NativeSearchTree
    encode_features = _implementation.encode_features
    encode_packed_features = getattr(_implementation, "encode_packed_features", None)
    pack_compact_evaluations = getattr(
        _implementation, "pack_compact_evaluations", None
    )
    pack_compact_states = getattr(_implementation, "pack_compact_states", None)
    score_cells = _implementation.score_cells
    compute_handcrafted_features = getattr(
        _implementation, "compute_handcrafted_features", None
    )
    decode_packed_hybrid_inputs = getattr(
        _implementation, "decode_packed_hybrid_inputs", None
    )
    unpack_compact_evaluations = getattr(
        _implementation, "unpack_compact_evaluations", None
    )
    old_champion_choose = getattr(_implementation, "old_champion_choose", None)
    old_champion_generate_games = getattr(
        _implementation, "old_champion_generate_games", None
    )
    _IMPORT_ERROR = None

NATIVE_MODULE_NAME: Final = "native._hexadeca_native"


def is_available() -> bool:
    """Return whether the compiled native module loaded successfully."""

    return _IMPORT_ERROR is None


def require_native() -> None:
    """Raise a concise diagnostic when native acceleration is unavailable."""

    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "The required native extension is unavailable. Reinstall the project "
            "with a C++20 compiler and pybind11 available."
        ) from _IMPORT_ERROR


__all__ = [
    "NATIVE_MODULE_NAME",
    "NativeBoard",
    "NativeSearchTree",
    "encode_features",
    "encode_packed_features",
    "is_available",
    "pack_compact_evaluations",
    "pack_compact_states",
    "require_native",
    "score_cells",
    "compute_handcrafted_features",
    "decode_packed_hybrid_inputs",
    "unpack_compact_evaluations",
    "old_champion_choose",
    "old_champion_generate_games",
]

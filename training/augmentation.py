"""Deterministic D4 transforms for AlphaZero training batches.

The replay buffer stores one canonical position and one canonical policy target.
Augmentation is applied only when a training batch is materialized: spatial
feature planes, legal masks, and policy targets receive the same board
symmetry, while scalar targets remain unchanged.  The coordinate planes are
kept canonical because ``encode_state`` always emits coordinates for the
transformed board's own row/column axes.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from game import GameState

D4_TRANSFORM_NAMES: tuple[str, ...] = (
    "identity",
    "rotate90",
    "rotate180",
    "rotate270",
    "mirror_horizontal",
    "mirror_horizontal_rotate90",
    "mirror_horizontal_rotate180",
    "mirror_horizontal_rotate270",
)
D4_TRANSFORM_COUNT = len(D4_TRANSFORM_NAMES)
_D4_MASK = (1 << 64) - 1
_SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15
_SPLITMIX_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
_SPLITMIX_MULTIPLIER_2 = 0x94D049BB133111EB
_SPATIAL_PLANE_COUNT = 14


def select_d4_transform(
    state: GameState,
    *,
    seed: int,
    sample_offset: int = 0,
) -> int:
    """Select one D4 transform deterministically for a replay state.

    The state hash and batch-local offset make repeated positions receive
    independent-looking transforms, while the explicit seed makes the choice
    reproducible across resumed training and DataLoader worker layouts.
    """

    if type(seed) is not int or seed < 0:
        raise ValueError("Augmentation seed must be a nonnegative integer")
    if type(sample_offset) is not int or sample_offset < 0:
        raise ValueError("Augmentation sample offset must be nonnegative")
    value = (
        seed
        ^ state.zobrist_hash
        ^ ((state.ply + 1) * _SPLITMIX_INCREMENT)
        ^ (sample_offset * _SPLITMIX_MULTIPLIER_1)
    ) & _D4_MASK
    return _splitmix64(value) % D4_TRANSFORM_COUNT


def augment_training_batch(
    features: torch.Tensor,
    policy: torch.Tensor,
    legal_mask: torch.Tensor,
    transform_indices: Sequence[int],
    *,
    board_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply per-sample D4 transforms to features and policy-side targets.

    ``features`` must be ``[N, 16, H, W]``.  Planes 0--13 are spatial and are
    transformed; planes 14--15 are canonical coordinate planes and are copied
    unchanged.  Policy and legal tensors are transformed as ``[N, H, W]``
    grids, preserving row-major action encoding and target normalization.
    """

    _validate_batch_shapes(features, policy, legal_mask, transform_indices, board_size)
    if not transform_indices:
        return features, policy, legal_mask

    augmented_features = features.clone()
    augmented_policy = policy.clone()
    augmented_legal_mask = legal_mask.clone()
    index_tensor = torch.arange(
        len(transform_indices), dtype=torch.int64, device=features.device
    )

    for transform_index in range(D4_TRANSFORM_COUNT):
        selected = index_tensor[
            torch.tensor(
                [value == transform_index for value in transform_indices],
                dtype=torch.bool,
                device=features.device,
            )
        ]
        if selected.numel() == 0:
            continue

        selected_features = features.index_select(0, selected)
        spatial = _transform_grid(
            selected_features[:, :_SPATIAL_PLANE_COUNT], transform_index
        )
        selected_features = selected_features.clone()
        selected_features[:, :_SPATIAL_PLANE_COUNT] = spatial
        augmented_features.index_copy_(0, selected, selected_features)

        selected_policy = policy.index_select(0, selected).reshape(
            -1, board_size, board_size
        )
        augmented_policy.index_copy_(
            0,
            selected,
            _transform_grid(selected_policy, transform_index).reshape(
                -1, board_size * board_size
            ),
        )

        selected_legal = legal_mask.index_select(0, selected).reshape(
            -1, board_size, board_size
        )
        augmented_legal_mask.index_copy_(
            0,
            selected,
            _transform_grid(selected_legal, transform_index).reshape(
                -1, board_size * board_size
            ),
        )

    return augmented_features, augmented_policy, augmented_legal_mask


def _transform_grid(grid: torch.Tensor, transform_index: int) -> torch.Tensor:
    """Apply one of the eight board symmetries to a trailing 2D grid."""

    _validate_transform_index(transform_index)
    rotations = transform_index % 4
    transformed = torch.rot90(grid, rotations, dims=(-2, -1))
    if transform_index >= 4:
        transformed = torch.flip(transformed, dims=(-1,))
    return transformed


def _validate_batch_shapes(
    features: torch.Tensor,
    policy: torch.Tensor,
    legal_mask: torch.Tensor,
    transform_indices: Sequence[int],
    board_size: int,
) -> None:
    if type(board_size) is not int or board_size <= 0:
        raise ValueError("Board size must be a positive integer")
    if features.ndim != 4 or features.shape[1] != 16:
        raise ValueError("Features must have shape [N, 16, board_size, board_size]")
    if features.shape[-2:] != (board_size, board_size):
        raise ValueError("Feature spatial dimensions do not match board size")
    expected = (features.shape[0], board_size * board_size)
    if policy.shape != expected or legal_mask.shape != expected:
        raise ValueError("Policy and legal-mask shapes must match the feature batch")
    if policy.device != features.device or legal_mask.device != features.device:
        raise ValueError("Features and targets must share a device")
    if legal_mask.dtype is not torch.bool:
        raise ValueError("Legal mask must use torch.bool")
    if len(transform_indices) != features.shape[0]:
        raise ValueError("One D4 transform is required for every batch sample")
    for transform_index in transform_indices:
        _validate_transform_index(transform_index)


def _validate_transform_index(transform_index: int) -> None:
    if (
        type(transform_index) is not int
        or not 0 <= transform_index < D4_TRANSFORM_COUNT
    ):
        raise ValueError("D4 transform index is out of range")


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_INCREMENT) & _D4_MASK
    value = ((value ^ (value >> 30)) * _SPLITMIX_MULTIPLIER_1) & _D4_MASK
    value = ((value ^ (value >> 27)) * _SPLITMIX_MULTIPLIER_2) & _D4_MASK
    return (value ^ (value >> 31)) & _D4_MASK

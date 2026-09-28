"""Tests for deterministic D4 training-batch augmentation."""

from __future__ import annotations

import torch

from config import load_config
from game import GameEnvironment
from network import NetworkSpecification, encode_state
from training import (
    D4_TRANSFORM_COUNT,
    augment_training_batch,
    select_d4_transform,
)


def test_d4_transforms_map_one_corner_to_all_four_corners() -> None:
    """The eight transforms are distinct and cover every board orientation."""

    board_size = 4
    features = torch.zeros((1, 16, board_size, board_size))
    features[0, 0, 0, 0] = 1.0
    policy = torch.zeros((1, board_size * board_size))
    policy[0, 0] = 1.0
    legal_mask = torch.zeros((1, board_size * board_size), dtype=torch.bool)
    legal_mask[0, 0] = True

    transformed_actions: list[int] = []
    for transform_index in range(D4_TRANSFORM_COUNT):
        transformed_features, transformed_policy, transformed_legal = (
            augment_training_batch(
                features,
                policy,
                legal_mask,
                (transform_index,),
                board_size=board_size,
            )
        )
        action = int(torch.argmax(transformed_policy[0]).item())
        transformed_actions.append(action)
        assert transformed_features[0, 0].flatten()[action] == 1.0
        assert bool(transformed_legal[0, action])

    assert set(transformed_actions) == {0, 3, 12, 15}
    assert len(transformed_actions) == D4_TRANSFORM_COUNT


def test_d4_keeps_coordinates_and_policy_legal_alignment() -> None:
    """Every transformed batch remains a valid model input and target."""

    config = load_config()
    specification = NetworkSpecification.from_config(config.rules, config.network)
    environment = GameEnvironment(config.rules)
    for action in (0, 2, 34, 36, 68):
        environment.step(action)
    state = environment.state
    original_features = encode_state(state, specification).unsqueeze(0)
    legal_mask = torch.tensor((state.legal_mask,), dtype=torch.bool)
    legal_count = int(legal_mask.sum().item())
    policy = legal_mask.to(torch.float32) / legal_count

    for transform_index in range(D4_TRANSFORM_COUNT):
        features, transformed_policy, transformed_legal = augment_training_batch(
            original_features,
            policy,
            legal_mask,
            (transform_index,),
            board_size=specification.board_size,
        )
        assert torch.equal(features[0, 14:], original_features[0, 14:])
        assert torch.equal(features[0, 2].flatten().bool(), transformed_legal[0])
        assert torch.all(
            transformed_policy[0].masked_select(~transformed_legal[0]) == 0
        )
        assert torch.isclose(
            transformed_policy[0].sum(), torch.tensor(1.0), atol=1e-6, rtol=0.0
        )


def test_d4_selection_is_reproducible_and_seeded() -> None:
    """Resume/restart runs select the same transform for the same seed."""

    state = GameEnvironment(load_config().rules).state
    first = tuple(
        select_d4_transform(state, seed=20260816, sample_offset=offset)
        for offset in range(32)
    )
    second = tuple(
        select_d4_transform(state, seed=20260816, sample_offset=offset)
        for offset in range(32)
    )

    assert first == second
    assert set(first) == set(range(D4_TRANSFORM_COUNT))

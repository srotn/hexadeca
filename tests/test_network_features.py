"""Tests for the versioned 16-plane game-state encoder."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from config import load_config
from game import GameEnvironment, Player
from network import (
    FeatureEncodingError,
    NetworkSpecification,
    encode_batch,
    encode_state,
)


@pytest.fixture
def specification() -> NetworkSpecification:
    """Return the default versioned network specification."""

    config = load_config()
    return NetworkSpecification.from_config(config.rules, config.network)


def test_empty_state_planes_and_coordinates(
    specification: NetworkSpecification,
) -> None:
    """The initial position has exact occupancy, legal, side, and coordinates."""

    state = GameEnvironment(load_config().rules).state
    features = encode_state(state, specification)

    assert features.shape == (16, 16, 16)
    assert features.dtype is torch.float32
    assert torch.count_nonzero(features[0:2]).item() == 0
    assert torch.all(features[2] == 1)
    assert torch.all(features[3] == 1)
    assert torch.count_nonzero(features[4:14]).item() == 0
    assert torch.allclose(features[14, :, 0], torch.linspace(-1.0, 1.0, 16))
    assert torch.allclose(features[15, 0, :], torch.linspace(-1.0, 1.0, 16))


def test_history_and_latest_move_planes_are_exact(
    specification: NetworkSpecification,
) -> None:
    """Four prior positions are reconstructed by removing moves in reverse."""

    environment = GameEnvironment(load_config().rules)
    for action in (0, 2, 4, 6, 8):
        environment.step(action)

    features = encode_state(environment.state, specification)

    assert environment.state.to_play is Player.WHITE
    assert torch.count_nonzero(features[3]).item() == 0
    assert _occupied_actions(features[0]) == {0, 4, 8}
    assert _occupied_actions(features[1]) == {2, 6}
    assert _occupied_actions(features[4]) == {0, 4}
    assert _occupied_actions(features[5]) == {2, 6}
    assert _occupied_actions(features[6]) == {0, 4}
    assert _occupied_actions(features[7]) == {2}
    assert _occupied_actions(features[8]) == {0}
    assert _occupied_actions(features[9]) == {2}
    assert _occupied_actions(features[10]) == {0}
    assert _occupied_actions(features[11]) == set()
    assert _occupied_actions(features[12]) == {8}
    assert _occupied_actions(features[13]) == {6}


def test_batch_encoder_stacks_states_without_changing_order(
    specification: NetworkSpecification,
) -> None:
    """Batch encoding preserves input ordering and produces NCHW tensors."""

    environment = GameEnvironment(load_config().rules)
    initial = environment.state
    environment.step(34)
    after_move = environment.state

    batch = encode_batch((initial, after_move), specification)

    assert batch.shape == (2, 16, 16, 16)
    assert torch.equal(batch[0], encode_state(initial, specification))
    assert torch.equal(batch[1], encode_state(after_move, specification))


def test_encoder_rejects_corrupted_snapshot(
    specification: NetworkSpecification,
) -> None:
    """External snapshots cannot disagree with their immutable move history."""

    environment = GameEnvironment(load_config().rules)
    environment.step(0)
    state = environment.state
    corrupted = replace(state, cells=(0,) * state.action_size)

    with pytest.raises(FeatureEncodingError, match="move history"):
        encode_state(corrupted, specification)


def test_encoder_rejects_an_empty_batch(
    specification: NetworkSpecification,
) -> None:
    """An empty batch has no well-defined NCHW tensor shape."""

    with pytest.raises(FeatureEncodingError, match="empty"):
        encode_batch((), specification)


def _occupied_actions(plane: torch.Tensor) -> set[int]:
    return set(torch.nonzero(plane.flatten(), as_tuple=False).flatten().tolist())

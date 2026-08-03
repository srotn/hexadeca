"""Replay snapshot to PyTorch training-batch integration tests."""

from __future__ import annotations

from dataclasses import replace

import torch

from config import load_config
from game import GameEnvironment, Player
from network import AlphaZeroLoss, LossWeights, NetworkOutput, NetworkSpecification
from training import ReplayBuffer, ReplaySample, build_replay_data_loader


def _populated_buffer(count: int) -> ReplayBuffer:
    config = load_config()
    specification = NetworkSpecification.from_config(config.rules, config.network)
    buffer = ReplayBuffer(config.rules, specification, count + 1)
    environment = GameEnvironment(config.rules)
    samples: list[ReplaySample] = []
    for marker in range(count):
        state = environment.state
        legal_count = sum(state.legal_mask)
        samples.append(
            ReplaySample.create(
                state,
                tuple(
                    1.0 / legal_count if legal else 0.0 for legal in state.legal_mask
                ),
                win=1.0 if state.to_play is Player.BLACK else 0.0,
                black_score=100 + marker,
                white_score=10,
            )
        )
        environment.step(environment.legal_moves()[0])
    buffer.extend(samples)
    return buffer


def test_data_loader_builds_loss_compatible_batches_from_snapshot() -> None:
    """Compact histories become exact features, masks, and normalized targets."""

    config = load_config()
    training_config = replace(
        config.training,
        batch_size=2,
        data_loader_workers=0,
        data_loader_pin_memory=False,
        data_loader_drop_last=False,
    )
    buffer = _populated_buffer(4)
    specification = buffer.specification
    generator = torch.Generator().manual_seed(71)
    loader = build_replay_data_loader(
        buffer, specification, training_config, generator=generator
    )

    batch = next(iter(loader))
    assert batch.features.shape == (2, 16, 16, 16)
    assert batch.targets.policy.shape == (2, 256)
    assert batch.targets.legal_mask.shape == (2, 256)
    assert torch.allclose(batch.targets.policy.sum(dim=1), torch.ones(2))
    assert torch.all(batch.targets.policy.masked_select(~batch.targets.legal_mask) == 0)
    assert torch.all(
        (batch.targets.black_score >= 0) & (batch.targets.black_score <= 1)
    )

    output = NetworkOutput(
        policy_logits=torch.zeros(2, 256, requires_grad=True),
        black_score=torch.full((2,), 0.5, requires_grad=True),
        white_score=torch.full((2,), 0.5, requires_grad=True),
        win_logit=torch.zeros(2, requires_grad=True),
    )
    loss = AlphaZeroLoss(LossWeights.from_config(config.training))(
        output, batch.targets
    )
    loss.total.backward()
    assert torch.isfinite(loss.total)
    assert batch.to("cpu").features.device.type == "cpu"


def test_data_loader_owns_a_point_in_time_snapshot() -> None:
    """Later self-play writes cannot change an already constructed epoch."""

    config = load_config()
    training_config = replace(
        config.training,
        batch_size=2,
        data_loader_shuffle=False,
        data_loader_workers=0,
        data_loader_pin_memory=False,
        data_loader_drop_last=False,
    )
    buffer = _populated_buffer(3)
    loader = build_replay_data_loader(buffer, buffer.specification, training_config)
    original_batches = len(loader)
    first = buffer.snapshot()[0]
    buffer.append(first)

    assert len(loader) == original_batches == 2
    assert sum(batch.features.shape[0] for batch in loader) == 3

"""Immutable checkpoint persistence, compatibility, and recovery tests."""

from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from config import load_config
from network import NetworkSpecification, PolicyValueNetwork
from training import (
    CheckpointCompatibilityError,
    CheckpointError,
    CheckpointManager,
)


def _small_training_stack() -> tuple[
    object,
    NetworkSpecification,
    PolicyValueNetwork,
    torch.optim.AdamW,
    torch.optim.lr_scheduler.StepLR,
]:
    config = load_config()
    network_config = replace(
        config.network,
        residual_blocks=1,
        channels=8,
        value_hidden_features=16,
    )
    small_config = replace(config, network=network_config)
    specification = NetworkSpecification.from_config(
        small_config.rules, small_config.network
    )
    model = PolicyValueNetwork(specification)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    inputs = torch.randn(2, 16, 16, 16)
    model(inputs).policy_logits.mean().backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return small_config, specification, model, optimizer, scheduler


def test_checkpoint_round_trip_restores_training_and_rng_state(
    tmp_path: Path,
) -> None:
    """Model, optimizer, scheduler, metadata, aliases, and RNG resume exactly."""

    config, specification, model, optimizer, scheduler = _small_training_stack()
    manager = CheckpointManager(tmp_path, specification)
    expected_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    random.seed(31)
    torch.manual_seed(47)

    metadata = manager.save(
        "iteration-000007",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        iteration=7,
        config=config,
        metrics={"loss": 1.25, "positions": 512},
    )
    expected_python_random = random.random()
    expected_torch_random = torch.rand(3)
    for parameter in model.parameters():
        parameter.data.zero_()
    optimizer.param_groups[0]["lr"] = 9.0
    random.seed(99)
    torch.manual_seed(99)

    restored = manager.load(
        "latest",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        restore_rng=True,
    )

    assert restored == metadata
    assert restored.iteration == 7
    assert restored.metrics == {"loss": 1.25, "positions": 512}
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.005)
    assert random.random() == expected_python_random
    assert torch.equal(torch.rand(3), expected_torch_random)
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, expected_parameters[name])

    manager.publish_alias("best", restored.checkpoint_id)
    assert manager.read_metadata("best") == restored
    assert manager.list_checkpoint_ids() == ("iteration-000007",)
    manifest = json.loads(
        (tmp_path / "bundles" / restored.checkpoint_id / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["configuration"]["replay"]["persistence_enabled"] is True


def test_checkpoint_is_immutable_and_rejects_checksum_corruption(
    tmp_path: Path,
) -> None:
    """An existing ID cannot be overwritten and modified tensor bytes never load."""

    config, specification, model, optimizer, scheduler = _small_training_stack()
    manager = CheckpointManager(tmp_path, specification)
    arguments = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "iteration": 1,
        "config": config,
        "metrics": {},
    }
    manager.save("checkpoint-1", **arguments)
    with pytest.raises(CheckpointError, match="already exists"):
        manager.save("checkpoint-1", **arguments)

    state_path = tmp_path / "bundles" / "checkpoint-1" / "state.pt"
    with state_path.open("ab") as state_file:
        state_file.write(b"corrupt")
    with pytest.raises(CheckpointError, match="checksum"):
        manager.load("latest", model=model)


def test_checkpoint_rejects_incompatible_network_before_state_load(
    tmp_path: Path,
) -> None:
    """Architecture drift is rejected from manifest metadata before mutation."""

    config, specification, model, optimizer, scheduler = _small_training_stack()
    manager = CheckpointManager(tmp_path, specification)
    manager.save(
        "checkpoint-1",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        iteration=1,
        config=config,
        metrics={},
    )
    incompatible = replace(specification, channels=16)
    incompatible_model = PolicyValueNetwork(incompatible)
    incompatible_manager = CheckpointManager(tmp_path, incompatible)

    with pytest.raises(CheckpointCompatibilityError, match="incompatible"):
        incompatible_manager.load("latest", model=incompatible_model)

"""PyTorch data loading from immutable replay snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, cast

import torch
from torch.utils.data import DataLoader, Dataset

from config.schema import TrainingConfig
from network import NetworkSpecification, TrainingTargets, encode_batch
from training.errors import ReplayDataLoaderError
from training.replay import ReplayBuffer, ReplaySample, ReplaySnapshot


class ReplayBatch(NamedTuple):
    """Feature tensors and all targets consumed by one training step."""

    features: torch.Tensor
    targets: TrainingTargets

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> ReplayBatch:
        """Move one complete batch to a training device."""

        return ReplayBatch(
            features=self.features.to(device, non_blocking=non_blocking),
            targets=TrainingTargets(
                policy=self.targets.policy.to(device, non_blocking=non_blocking),
                legal_mask=self.targets.legal_mask.to(
                    device, non_blocking=non_blocking
                ),
                win=self.targets.win.to(device, non_blocking=non_blocking),
                black_score=self.targets.black_score.to(
                    device, non_blocking=non_blocking
                ),
                white_score=self.targets.white_score.to(
                    device, non_blocking=non_blocking
                ),
            ),
        )


@dataclass(frozen=True, slots=True)
class ReplayCollator:
    """Encode compact states and normalize raw score targets per batch."""

    specification: NetworkSpecification

    def __call__(self, samples: list[ReplaySample]) -> ReplayBatch:
        if not samples:
            raise ReplayDataLoaderError("Cannot collate an empty replay batch")
        states = tuple(sample.state for sample in samples)
        features = encode_batch(states, self.specification)
        policy = torch.tensor(
            [sample.policy for sample in samples], dtype=torch.float32
        )
        legal_mask = torch.tensor(
            [sample.state.legal_mask for sample in samples], dtype=torch.bool
        )
        normalizer = self.specification.score_normalizer
        return ReplayBatch(
            features=features,
            targets=TrainingTargets(
                policy=policy,
                legal_mask=legal_mask,
                win=torch.tensor(
                    [sample.win for sample in samples], dtype=torch.float32
                ),
                black_score=torch.tensor(
                    [sample.black_score / normalizer for sample in samples],
                    dtype=torch.float32,
                ),
                white_score=torch.tensor(
                    [sample.white_score / normalizer for sample in samples],
                    dtype=torch.float32,
                ),
            ),
        )


class _ReplayDataset(Dataset[ReplaySample]):
    """Typed PyTorch adapter around an immutable replay snapshot."""

    def __init__(self, snapshot: ReplaySnapshot) -> None:
        self._snapshot = snapshot

    def __len__(self) -> int:
        return len(self._snapshot)

    def __getitem__(self, index: int) -> ReplaySample:
        return self._snapshot[index]


def build_replay_data_loader(
    source: ReplayBuffer | ReplaySnapshot,
    specification: NetworkSpecification,
    config: TrainingConfig,
    *,
    generator: torch.Generator | None = None,
) -> DataLoader[ReplayBatch]:
    """Build a deterministic loader over a point-in-time replay snapshot."""

    snapshot = source.snapshot() if isinstance(source, ReplayBuffer) else source
    if snapshot.specification != specification:
        raise ReplayDataLoaderError(
            "Replay snapshot specification does not match the data loader"
        )
    if len(snapshot) == 0:
        raise ReplayDataLoaderError("Cannot load training data from empty replay")
    if config.data_loader_drop_last and len(snapshot) < config.batch_size:
        raise ReplayDataLoaderError(
            "Replay snapshot is smaller than one configured full batch"
        )
    if config.data_loader_shuffle and generator is None:
        raise ReplayDataLoaderError(
            "Shuffled replay loading requires an explicit torch.Generator"
        )

    dataset = _ReplayDataset(snapshot)
    collator = ReplayCollator(specification)
    if config.data_loader_workers > 0:
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.data_loader_shuffle,
            num_workers=config.data_loader_workers,
            collate_fn=collator,
            pin_memory=config.data_loader_pin_memory,
            drop_last=config.data_loader_drop_last,
            generator=generator,
            persistent_workers=True,
            prefetch_factor=config.data_loader_prefetch_factor,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.data_loader_shuffle,
            num_workers=0,
            collate_fn=collator,
            pin_memory=config.data_loader_pin_memory,
            drop_last=config.data_loader_drop_last,
            generator=generator,
            persistent_workers=False,
        )
    return cast(DataLoader[ReplayBatch], loader)

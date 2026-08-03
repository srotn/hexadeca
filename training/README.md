# Training Module

Stage 9 adds deterministic multiprocess self-play to the Stage 8 replay, data
loading, and checkpoint contracts.

Stage 10 adds the optimizer loop. Self-play and training remain sequential
owners of the active model: completed self-play samples enter Replay atomically,
then `Trainer` takes an immutable snapshot for one training iteration. This
prevents mutable Replay writes or inference-mode ownership from racing model
updates.

## Trainer

One training iteration performs exactly the configured number of optimizer
updates. If a snapshot contains fewer batches, the deterministic DataLoader is
reshuffled and cycled; the minimum Replay size still guarantees at least one
full batch. Each iteration derives its data-order seed independently, so a
checkpoint resumed at the next iteration reproduces uninterrupted CPU updates.

The optimizer is AdamW with explicit learning rate, weight decay, betas,
epsilon, and AMSGrad settings. L2 remains exclusively decoupled weight decay and
is not added to `AlphaZeroLoss`. The scheduler counts successful optimizer
steps: it linearly warms up for 1,000 steps, cosine-decays for 100,000 further
steps, then holds the configured minimum learning rate.

CUDA training uses FP16 autocast and `GradScaler`. A detected overflow does not
advance the optimizer-step counter or scheduler; the scaler backs off and the
batch is retried up to the configured limit. Gradients are unscaled before the
configured global-norm clip. CPU training uses float32 and a no-op scaler.

Batch and iteration metrics include all loss components, policy entropy,
learning rate, gradient norm, AMP retries, positions, and elapsed time.
TensorBoard is the default metric sink, while the small `MetricSink` protocol
allows Stage 13 monitoring to subscribe without changing Trainer logic.

Periodic immutable checkpoints contain the model, AdamW, scheduler, GradScaler,
Python/PyTorch/CUDA RNG states, exact training configuration, update count, and
metrics. Checkpoint schema v2 adds scaler state while retaining read support for
Stage 8 schema v1. Resume rejects training-configuration drift and continues
only from a complete iteration boundary.

## Self-play

`SelfPlayCoordinator` starts workers with the cross-platform, CUDA-safe
`spawn` method. Each worker exclusively owns its game environment, MCTS tree,
and per-game random stream. The main process exclusively owns the active model
and evaluator, aggregates worker requests up to the configured inference
capacity, and returns evaluations in request order. CUDA models are therefore
never copied into workers, and parallel games can share larger neural batches.

Per-game seeds are derived from the master seed and stable game index with
SplitMix64, so process scheduling and worker count do not reassign randomness.
Completed games are validated by replaying their full action history through
the authoritative environment, then ordered by game index. `SelfPlayBatch`
does not mutate Replay automatically; its `commit` method validates and inserts
the full flattened batch atomically. A worker or inference failure returns no
batch and cannot partially update Replay.

An optional move callback publishes immutable `SelfPlayProgress` records for
later monitoring without giving the frontend ownership of game logic. Worker
count, thread count, inference aggregation, timeouts, game count, and random
seed all come from the unified configuration.

## Replay data

`ReplaySample` stores a non-terminal canonical game state, normalized MCTS root
visit distribution, outcome from the position player's perspective, and final
raw Black/White scores. `ReplayBuffer` validates the complete sample before it
enters a bounded position-based ring. Multi-sample insertion validates every
entry before committing any of them.

Large fixed-width targets use preallocated CPU tensors. Variable-length action
histories use compact little-endian 16-bit records and are replayed through the
authoritative game engine when read. A `ReplaySnapshot` is chronological and
isolated from later writes, so a trainer epoch cannot change underneath a
self-play producer.

## Persistence

`SqliteReplayStore` writes a complete, versioned SQLite database containing a
strict metadata table and ordered sample rows. Policy targets are portable
little-endian float32 blobs. The new database is written beside its destination,
checked with SQLite `integrity_check`, flushed, and atomically published only
after it closes successfully.

Restore opens the database read-only and validates its schema, metadata, model
specification, row count, every compact blob, and every reconstructed sample.
When the configured capacity is smaller than the stored snapshot, only newest
positions are retained and `ReplayLoadResult.dropped_positions` reports the
exact reduction. Persistence is enabled by default but remains independently
configurable because a 200,000-position snapshot is a material disk artifact.

## Data loading

`build_replay_data_loader` owns a point-in-time snapshot. Its collator rebuilds
the versioned 16 feature planes only when a batch is requested and produces the
existing `TrainingTargets` contract with normalized scores. Shuffle, worker
count, prefetch, pinned memory, and incomplete-batch behavior all come from the
unified configuration. Shuffling requires an explicit `torch.Generator`.

## Checkpoints

`CheckpointManager` publishes immutable bundles:

```text
checkpoints/
  bundles/<checkpoint-id>/manifest.json
  bundles/<checkpoint-id>/state.pt
  aliases/latest.json
  aliases/best.json
```

The JSON manifest contains the complete configuration snapshot, network
specification, iteration, metrics, parent provenance, Python RNG state, and a
SHA-256 checksum. The tensor file contains model, optional optimizer/scheduler,
and PyTorch CPU/CUDA RNG state. Loading verifies metadata compatibility and the
checksum before using PyTorch's restricted `weights_only` loader. Alias files
are atomically replaced and never overwrite immutable bundle history.

Candidate-versus-best promotion and checkpoint retention policy begin in Stage
11; Stage 10 publishes only the immutable `latest` training lineage.

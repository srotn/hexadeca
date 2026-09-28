# Training Module

Stage 9 adds deterministic multiprocess self-play to the Stage 8 replay, data
loading, and checkpoint contracts.

Stage 10 adds the optimizer loop. Self-play and training remain sequential
owners of the active model: completed self-play samples enter Replay atomically,
then `Trainer` takes an immutable snapshot for one training iteration. This
prevents mutable Replay writes or inference-mode ownership from racing model
updates.

Stage 11 adds candidate-versus-best Arena evaluation. It does not make game
rules or model quality decisions in the frontend: the authoritative engine,
MCTS, terminal scorer, statistics, checkpoint gate, and report writer all
remain backend components.

## Arena evaluation

The configured 100 games are derived from 50 unique legal opening prefixes.
Each prefix is played twice with candidate and incumbent colors exchanged.
Openings are generated from the configured seed, have distinct first actions,
and are stored verbatim in the report. After the forced two-ply prefix, both
models use evaluation MCTS with 1,600 simulations, no root noise, and zero
temperature.

`ArenaCoordinator` uses spawn workers for isolated game and tree ownership.
The main process exclusively owns both models, groups inference requests by
candidate or best role, and batches each model independently. A worker,
inference, replay-validation, or shutdown failure returns no partial Arena.

Wins, draws, and losses give the candidate 1, 0.5, and 0 match points. The 95%
interval resamples complete two-game opening pairs with 10,000 deterministic
bootstrap samples. Performance Elo uses the configured logistic scale and a
symmetric half-point prior so all-win and all-loss results remain finite. A
candidate is promoted only when its score is at least 55% and the paired
confidence lower bound is greater than 50%.

## Evaluation reports

`CheckpointEvaluationGate` fully loads the first candidate before initializing
the `best` alias. Later candidates are compared to the resolved immutable best
checkpoint. The complete report is published through an atomic directory
rename before an accepted candidate updates `best`; matching report publication
is idempotent so a pointer-update interruption can be retried. Existing corrupt
or identity-mismatched reports are never silently reused.

Each report records checkpoint provenance, exact MCTS and Arena configuration,
all openings, color assignments, action histories, official scores, per-game
simulation counts, aggregate inference counters, paired confidence bounds,
performance Elo, and the promotion decision.

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

An optional scheduler-only recovery cycle is bound to one exact checkpoint ID.
It restores model, AdamW moments, GradScaler, and RNG state normally, then
replaces only scheduler progress. The cycle warms from its configured initial
LR to a bounded recovery peak and cosine-decays to its minimum. Subsequent
checkpoints store the independent scheduler-step counter, so restarting the
service does not apply the recovery twice. A checkpoint other than the
configured source follows strict resume validation and is rejected while the
one-shot profile remains enabled.

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

The default `training.symmetry_augmentation = "d4"` setting applies one of the
eight board rotations/reflections independently to each materialized sample.
The same transform is applied to spatial feature planes, legal masks, and
policy targets; scalar score and outcome targets are unchanged. Coordinate
planes remain canonical for the transformed board. Transform selection is
derived from the iteration seed and state hash, so resumed training remains
reproducible even with multiple DataLoader workers. Set the option to `"none"`
only for a controlled non-augmented baseline.

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

Stage 10 publishes the immutable `latest` training lineage. Stage 11 validates
the first best model and changes `best` only through the audited evaluation
gate. Checkpoint retention beyond immutable bundle history remains a later
operational policy.

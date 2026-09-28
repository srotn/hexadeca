# Benchmarks

Run the Stage 3 reference-board benchmark from the repository root:

```powershell
.\.venv\Scripts\python -m benchmark.board_engine
```

Run the Stage 4 exact-scoring benchmark:

```powershell
.\.venv\Scripts\python -m benchmark.scoring
```

Run the Stage 5 full-environment replay benchmark:

```powershell
.\.venv\Scripts\python -m benchmark.environment
```

Run the Stage 6 CPU/CUDA policy-value inference benchmark:

```powershell
.\.venv\Scripts\python -m benchmark.network
```

Use `--cpu-only` on hosts without a configured CUDA runtime. The network
report distinguishes batch calls per second from positions per second and
records PyTorch, CUDA, device, architecture, precision, and batch size.

Run the Stage 7 Python-core and CUDA AMP neural-MCTS benchmark:

```powershell
.\.venv\Scripts\python -m benchmark.mcts
```

The report distinguishes complete searches per second, simulations per second,
and the actual average/maximum neural inference batch formed by the tree.

Run the Stage 8 replay, collation, and SQLite persistence benchmark:

```powershell
.\.venv\Scripts\python -m benchmark.replay
```

The report measures ring insertion, uniform sampling without replacement,
feature/target collation, complete SQLite snapshot writes, and validated
restores in positions per second. It also records database bytes per retained
position.

Run the Stage 9 end-to-end multiprocess self-play benchmark:

```powershell
.\.venv\Scripts\python -m benchmark.self_play
```

This uses the real configured ResNet and centralized `TorchBatchEvaluator`.
CUDA AMP is selected when available; use `--cpu-only` to force CPU inference.
The report includes worker startup and shutdown in wall-clock throughput and
records games, generated positions, simulations, centralized inference calls,
and actual average/maximum inference batch size.

For production-like bottleneck profiling, pause training and use the same
checkpoint, games, simulations, worker count, and inference batch capacity as
the active training run:

```bash
python3 -m benchmark.self_play \
  --profile config/compact-inference.toml \
  --checkpoint latest \
  --profile-timing \
  --games 32 \
  --simulations 800 \
  --workers 16 \
  --inference-batch-size 256 \
  --output benchmark-results/self-play-production-profile.json
```

`--profile-timing` is benchmark-only. It reports separate coordinator wall
time, aggregate worker MCTS time, remote inference wait time, feature encoding,
GPU model and post-processing time, host transfer, and result materialization.
It does not read or modify Replay data, aliases, or training configuration.

Run the Stage 10 real optimizer benchmark:

```powershell
.\.venv\Scripts\python -m benchmark.trainer
```

The configured full ResNet performs warm-up and measured batches through
forward, all four losses, backward, AMP unscale, gradient clipping, AdamW, and
the scheduler. TensorBoard, checkpoint I/O, and data collation are excluded
from the timed optimizer scope. CUDA FP16 AMP is selected when available; use
`--cpu-only` for float32 CPU training.

Run the Stage 11 paired candidate-versus-best Arena benchmark:

```powershell
.\.venv\Scripts\python -m benchmark.evaluation
```

The benchmark loads two identical full ResNets to establish a neutral baseline,
generates color-swapped opening pairs, runs multiprocess evaluation MCTS, and
times the complete Arena lifecycle. It reports games and simulations per
second plus centralized inference positions and actual batch sizes. The
benchmark uses a reduced configured game/search budget; production promotion
retains 100 games and 1,600 simulations per searched move.

Applicable warm-up, measurement, position, game, simulation, and random-seed
settings come from the merged project configuration. The commands write
machine-readable reports to the configured benchmark-results directory.

Run the Stage 13 monitoring event benchmark:

```powershell
.\.venv\Scripts\python -m benchmark.monitoring
```

This measures publication and JSON serialization for one complete 16x16
self-play board snapshot, which is the payload sent to every monitoring client.

## Offline Decision Quality

This inference-only benchmark does not write replay data, checkpoints,
configuration, or alter production MCTS behavior. Artifacts are checksum-bound
to one model and long searches resume by completed position.

```bash
python -m benchmark.decision_quality --profile config/compact-inference.toml \
  --checkpoint iteration-001360 generate \
  --output benchmark-results/decision-quality/iteration-001360-dataset.json

python -m benchmark.decision_quality --profile config/compact-inference.toml \
  --checkpoint iteration-001360 sweep \
  --dataset benchmark-results/decision-quality/iteration-001360-dataset.json \
  --output-directory benchmark-results/decision-quality/ablation

python -m benchmark.decision_quality --profile config/compact-inference.toml \
  --checkpoint iteration-001360 score \
  --dataset benchmark-results/decision-quality/iteration-001360-dataset.json \
  --reference benchmark-results/decision-quality/ablation/search-s6400-c2p3.json \
  --output benchmark-results/decision-quality/score-heads.json

python -m benchmark.decision_quality.arena \
  --profile config/compact-inference.toml \
  --checkpoint iteration-001360 --device cuda \
  --games 100 --opening-plies 2 \
  --baseline-simulations 800 --changed-simulations 1600 \
  --workers 1 --inference-batch-size 32 --seed 20260814 \
  --output benchmark-results/decision-quality/arena/sims-800-vs-1600-final.json
```

The default fixed set contains 50 no-noise games and 300 positions balanced
across opening, middle, and endgame. The sweep compares simulations
`200/800/1600/3200/6400` and `c_puct` `1.5/2.0/2.3/2.8` against a fixed
6400-simulation reference. Pressing `Ctrl+C` is safe; repeat the same command
to continue. The Arena uses the same checkpoint on both sides, swaps colors
over 50 opening pairs, disables root noise, and reports a paired-bootstrap
confidence interval and finite performance Elo. Use one worker on memory-limited
Windows hosts; this changes throughput but not the match specification.

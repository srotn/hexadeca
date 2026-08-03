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

Applicable warm-up, measurement, position, game, simulation, and random-seed
settings come from the merged project configuration. The commands write
machine-readable reports to the configured benchmark-results directory.

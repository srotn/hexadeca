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

Warm-up count, measurement count, position depth, and random seed come from the
merged project configuration. The commands write machine-readable reports to
the configured benchmark-results directory.

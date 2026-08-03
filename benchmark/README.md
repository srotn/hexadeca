# Benchmarks

Run the Stage 3 reference-board benchmark from the repository root:

```powershell
.\.venv\Scripts\python -m benchmark.board_engine
```

Warm-up count, measurement count, position depth, and random seed come from the
merged project configuration. The command writes a machine-readable report to
`benchmark-results/stage3-board.json` by default.

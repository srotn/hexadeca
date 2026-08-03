# Hexadeca AlphaZero

This repository develops a neural-network-guided AlphaZero agent for the
Hexadeca board game through gated stages. The current implementation is Stage
9: deterministic multiprocess self-play with centralized neural inference,
validated replay storage, PyTorch data loading, and resumable immutable
checkpoints.

The authoritative requirements and architecture are documented in
[`STAGE1_DESIGN.md`](STAGE1_DESIGN.md). Trainer lifecycle and optimization
remain deliberately absent until Stage 10.

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

Run the project quality checks:

```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\ruff format --check .
.\.venv\Scripts\ruff check .
.\.venv\Scripts\mypy
```

Run the configured Stage 3 board benchmark:

```powershell
.\.venv\Scripts\python -m benchmark.board_engine
```

Run the configured Stage 4 scoring benchmark:

```powershell
.\.venv\Scripts\python -m benchmark.scoring
```

Run the configured Stage 5 environment benchmark:

```powershell
.\.venv\Scripts\python -m benchmark.environment
```

Run the configured Stage 6 CPU/CUDA inference benchmark:

```powershell
.\.venv\Scripts\python -m benchmark.network
```

Run the configured Stage 7 MCTS core/CUDA benchmark:

```powershell
.\.venv\Scripts\python -m benchmark.mcts
```

Run the configured Stage 8 replay and persistence benchmark:

```powershell
.\.venv\Scripts\python -m benchmark.replay
```

Run the configured Stage 9 multiprocess self-play benchmark:

```powershell
.\.venv\Scripts\python -m benchmark.self_play
```

The command uses CUDA AMP when CUDA is available and falls back to CPU. Pass
`--cpu-only` to force CPU inference.

## Repository layout

```text
benchmark/  Reproducible benchmark scripts and reports.
config/     Versioned TOML configuration and validation.
cpp/        Reserved for profile-validated C++20 acceleration.
game/       Board, scoring, and immutable self-play environment contracts.
mcts/       PUCT nodes, virtual loss, batched evaluator, and neural search.
network/    Versioned features, residual policy-value model, mask, and loss.
tests/      Unit and integration tests.
training/   Self-play, replay, data loading, checkpoints, and training stages.
utils/      Cross-cutting utilities, currently structured logging.
```

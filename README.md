# Hexadeca AlphaZero

This repository develops a neural-network-guided AlphaZero agent for the
Hexadeca board game through gated stages. The current implementation is Stage
4: a reversible reference board engine with exact terminal distance-layer
scoring.

The authoritative requirements and architecture are documented in
[`STAGE1_DESIGN.md`](STAGE1_DESIGN.md). The game environment, MCTS, and
neural-network implementation remain deliberately absent until their stages.

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

## Repository layout

```text
benchmark/  Reproducible benchmark scripts and reports.
config/     Versioned TOML configuration and validation.
cpp/        Reserved for profile-validated C++20 acceleration.
game/       Canonical board, undo, legal-move, hashing, and scoring engine.
mcts/       Reserved for neural MCTS (Stage 7).
network/    Reserved for the policy-value network (Stage 6).
tests/      Unit and integration tests.
training/   Reserved for replay, self-play, and training stages.
utils/      Cross-cutting utilities, currently structured logging.
```

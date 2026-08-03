# Hexadeca AlphaZero

This repository develops a neural-network-guided AlphaZero agent for the
Hexadeca board game through gated stages. The current implementation is Stage
6: a self-play-ready game environment plus a versioned 16-plane PyTorch
residual policy-value network with legal masking and composite training loss.

The authoritative requirements and architecture are documented in
[`STAGE1_DESIGN.md`](STAGE1_DESIGN.md). MCTS remains deliberately absent until
Stage 7.

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

## Repository layout

```text
benchmark/  Reproducible benchmark scripts and reports.
config/     Versioned TOML configuration and validation.
cpp/        Reserved for profile-validated C++20 acceleration.
game/       Board, scoring, and immutable self-play environment contracts.
mcts/       Reserved for neural MCTS (Stage 7).
network/    Versioned features, residual policy-value model, mask, and loss.
tests/      Unit and integration tests.
training/   Reserved for replay, self-play, and training stages.
utils/      Cross-cutting utilities, currently structured logging.
```

# Hexadeca AlphaZero

This repository develops a neural-network-guided AlphaZero agent for the
Hexadeca board game through gated stages. The current implementation is Stage
2: project foundations, configuration, structured logging, tests, and CI.

The authoritative requirements and architecture are documented in
[`STAGE1_DESIGN.md`](STAGE1_DESIGN.md). No board, scoring, MCTS, or neural
network implementation exists yet.

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

Run the Stage 2 quality checks:

```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\ruff format --check .
.\.venv\Scripts\ruff check .
.\.venv\Scripts\mypy
```

## Repository layout

```text
benchmark/  Reproducible benchmark scripts and reports.
config/     Versioned TOML configuration and validation.
cpp/        Reserved for profile-validated C++20 acceleration.
game/       Reserved for the canonical game engine (Stage 3).
mcts/       Reserved for neural MCTS (Stage 7).
network/    Reserved for the policy-value network (Stage 6).
tests/      Unit and integration tests.
training/   Reserved for replay, self-play, and training stages.
utils/      Cross-cutting utilities, currently structured logging.
```

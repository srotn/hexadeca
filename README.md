# Hexadeca AlphaZero

This repository develops a neural-network-guided AlphaZero agent for the
Hexadeca board game through gated stages. The current implementation is Stage
11: deterministic multiprocess self-play, a configuration-driven CUDA Trainer,
and paired candidate-versus-best Arena evaluation with confidence-aware model
promotion and immutable reports.

The authoritative game rules are recorded in [`game_rule.md`](game_rule.md).
Architecture and stage gates are documented in
[`STAGE1_DESIGN.md`](STAGE1_DESIGN.md).

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

Run the configured Stage 10 optimizer/CUDA AMP benchmark:

```powershell
.\.venv\Scripts\python -m benchmark.trainer
```

The report measures real forward, composite loss, backward, gradient clipping,
AdamW, and scheduler work. Pass `--cpu-only` to force float32 CPU training.

Run the configured Stage 11 paired Arena benchmark:

```powershell
.\.venv\Scripts\python -m benchmark.evaluation
```

The benchmark loads two full models, runs color-swapped opening pairs through
multiprocess MCTS, and reports complete game, simulation, and centralized
inference throughput. Pass `--cpu-only` to force CPU inference.

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

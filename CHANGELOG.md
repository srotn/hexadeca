# Changelog

All notable public changes to Hexadeca are recorded here.

## [0.1.0] - 2026-09-29

### Added

- Public 16×16 `hexadeca-v1` rules and a runnable game engine.
- Residual policy/value network with legal-action masking and score heads.
- PUCT search with FPU, virtual loss, batch leaf selection and batched inference.
- Deterministic self-play, Replay training, checkpoint management and paired arena evaluation.
- Optional C++20 Native acceleration for board, scoring, features and search.
- Tests and documentation for rules, architecture and combinatorial complexity.

### Scope

- This release contains the 16×16 research system only.
- The 20×20 research branch, model checkpoints, replay buffers, logs and local run artifacts are excluded.

### Notes

- Complexity figures in the documentation are rule-derived combinatorial analysis, not runtime benchmarks.
- Relative Elo values are internal evaluation-scale results and are not comparable with ratings from other games.

# MCTS Module

Stage 7 provides a correctness-first neural Monte Carlo tree search with legal
PUCT expansion, alternating-perspective backup, configurable Dirichlet root
noise, reversible virtual loss, exact simulation counts, and batched policy-
value inference. Replay policy targets retain normalized root visit counts,
while the configured temperature affects action sampling only. Terminal leaves
use the official scoring engine; search never uses rollouts or a manually
authored evaluation function.

`MctsSearch` is intentionally single-owner. It uses virtual-loss reservations
to form independent leaf batches while preserving committed statistics. Stage
9 will run multiple self-play workers, and Stage 12 may replace profiled search
hotspots with parallel C++ without changing these observable contracts.

# Hexadeca Architecture

Hexadeca is organized as a deterministic game engine surrounded by a neural-search and self-play training loop. The public release targets the 16×16 `hexadeca-v1` rules only.

## Runtime flow

```text
Board state
    ↓
Feature encoder ──→ residual policy/value network ──→ batched leaf evaluation
    ↑                                                    ↓
    └────────────── PUCT / MCTS ← visit counts, FPU, virtual loss
                         ↓
                   selected action
                         ↓
                   self-play record
                         ↓
                Replay sampling and training
                         ↓
                     checkpoint
                         ↓
              paired arena / relative Elo
```

## Rules and state

The Python engine owns legal-move generation, immutable move semantics, terminal detection and distance-layer scoring. The action identifier is `row * 16 + column` in the range 0–255. A state includes the board, player to move, move history, legal mask and terminal status. A Zobrist hash is used for fast state identity; exact caches use the complete board representation and player-to-move information.

The optional C++20 Native path mirrors the board, scoring, feature and search operations for throughput-sensitive experiments. It is an acceleration layer, not a second rule definition.

## Network

The current 16×16 model consumes 16 planes: board occupancy, legal moves, side to move, recent board history, latest-move indicators and normalized coordinates.

The main trunk is a 3×3 convolution followed by ten residual blocks with 128 channels. The policy branch maps to two spatial channels and then 256 action logits. The value branch maps to one spatial channel, a 256-unit hidden layer and three outputs: Black score, White score and a current-player result logit.

The policy is masked by legal actions outside the model. Result labels use win=1, draw=0.5 and loss=0 from the current player's perspective. Score heads are normalized by the 256-square board size. D4 augmentation is implemented as data augmentation; it is not a formal claim of strict architectural equivariance.

## Search

MCTS uses PUCT priors from the policy branch and neural values at leaf nodes. The implementation supports First Play Urgency, virtual loss for parallel workers, batch leaf selection and centralized batched inference. Training self-play can add Dirichlet root noise and temperature sampling. Evaluation protocols normally disable root noise and use deterministic action selection.

The exact endgame path solves WDL outcomes for small remaining action sets with memoization. It is a local solver and does not solve the 16×16 game or optimize the final score margin among all winning moves.

## Training and evaluation

Self-play produces positions with search visit targets and terminal result/score targets. Replay sampling feeds the optimizer, and checkpoints preserve model and optimizer state. Arena evaluation compares checkpoints under a fixed search budget with color-swapped opening pairs. Relative Elo is an internal scale for a specified protocol; it is not comparable to ratings in other games.

The public source intentionally excludes checkpoint weights, replay databases, logs, temporary run directories and the separate 20×20 research branch. Users can reproduce the rules, model construction, search components and tests without downloading private or machine-specific artifacts.

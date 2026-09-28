# Hexadeca-v1 Complexity Analysis

The figures in this document describe the combinatorial scale of the current 16×16 `hexadeca-v1` rules. They are rule-derived counts and approximations, not program benchmarks, measured runtime, or claims that all states have been enumerated by the implementation.

## Summary

| Quantity | Value |
|---|---:|
| Maximum game length | 64 plies |
| Action space | 256 |
| Reachable colored states | ≈ 1.56896 × 10^46 |
| Uncolored legal occupied-set states | ≈ 4.42220 × 10^34 |
| Full game-tree history nodes | ≈ 2.04243 × 10^104 |
| State-space information complexity | ≈ 153.46 bits |
| Game-tree information complexity | ≈ 346.51 bits |
| State-space peak | 44 plies |
| Peak states at 44 plies | ≈ 2.09534 × 10^45 |
| Game-tree peak | 63 plies |
| Opening branching factor | 256 |
| Effective average branching factor | ≈ 42.63 |

The state-space peak and game-tree peak occur at different depths: around 44 plies for distinct states and around 63 plies for historical game-tree nodes.

## Late-game layers

| Layer | Approximate states |
|---|---:|
| ≤1 ply from theoretical maximum | ≈ 2.896 × 10^34 |
| ≤4 plies | ≈ 2.175 × 10^38 |
| ≤8 plies | ≈ 6.304 × 10^41 |
| ≤10 plies | ≈ 1.127 × 10^43 |
| ≤12 plies | ≈ 1.112 × 10^44 |
| ≤16 plies | ≈ 2.373 × 10^45 |
| 64-piece terminal-layer states | ≈ 4.673 × 10^32 |

## Interpretation

The action space is small enough for a flat policy head with 256 logits, but the number of reachable configurations is enormous. Multiple move histories can transpose to the same colored board, so historical game-tree nodes and distinct state-space nodes measure different forms of complexity. This is why the game-tree estimate is much larger than the colored-state estimate.

The opening branching factor is exactly 256. The effective average branching factor summarizes the rule-constrained game tree across its depth; it should not be interpreted as a constant branching factor at every ply. The legal-move count depends on the geometry and overlap of forbidden neighborhoods.

The 64-ply bound follows from partitioning the board into 64 disjoint 2×2 blocks. It is an upper bound on legal play length, not a statement that every legal position at every depth is reachable or that every terminal game lasts 64 plies.

The information-complexity values use base-2 logarithms of the corresponding state-space and game-tree estimates. Because the underlying quantities are combinatorial approximations, the displayed bit values should be read with the same approximation status.

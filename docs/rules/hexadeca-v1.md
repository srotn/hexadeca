# Hexadeca-v1 Formal Rules

This document defines the current official Hexadeca rule set. The public implementation targets this version and does not silently change it when implementation details or research configurations change.

## 1. Board and turn order

- The board is a finite, non-wrapping 16×16 grid.
- Rows and columns are indexed from 0 through 15.
- There are two colors, Black and White. Black moves first.
- Players alternate turns.
- A move places exactly one stone of the current player's color on one empty square.
- Stones never move, get captured, get removed, or get replaced.
- There is no Pass action, random event, hidden information, or simultaneous action.

## 2. Legal moves

Let `S` be the set of occupied squares. A candidate square `x=(r,c)` is legal exactly when it is on the board, empty, and for every occupied square `s=(sr,sc)`:

```text
max(abs(r - sr), abs(c - sc)) > 1
```

Thus every stone forbids its own square and its eight in-bounds neighbors. Legality does not depend on stone color.

The empty board has 256 legal moves. After the first move, the next legal-move count is 252 for a corner, 250 for a non-corner edge square, and 247 for an interior square. Later forbidden neighborhoods may overlap, so a fixed subtraction is not valid.

## 3. Termination

The game ends immediately when the legal-move set is empty. The current player cannot pass and the other player does not receive an extra turn. All remaining empty squares are forbidden at termination.

## 4. Distance-layer scoring

All 256 board squares are scored, including occupied squares and forbidden empty squares. Each square contributes at most one point to one side, or contributes no point.

For each square `x`, compute the squared Euclidean distance to every stone:

```text
d2(x, s) = (x.row - s.row)^2 + (x.column - s.column)^2
```

Group stones with equal distance into layers and inspect layers from nearest to farthest:

1. Count Black stones `B` and White stones `W` in the current layer.
2. If `B > W`, award this square to Black and add one point.
3. If `W > B`, award this square to White and add one point.
4. If `B == W`, skip that layer and inspect the next layer independently.
5. If every layer is tied, the square is neutral.

Counts are not accumulated between layers, and a majority never awards more than one point for a square. An occupied square has a unique distance-zero stone and therefore scores for that stone's color.

## 5. Result

Let `SB` and `SW` be the final Black and White scores, and `U` the number of neutral squares:

```text
SB + SW + U = 256
```

Black wins when `SB > SW`, White wins when `SW > SB`, and the game is a draw when `SB == SW`. The canonical rules contain no komi.

## 6. Derived properties

- The game is finite and has no cycles because each move adds one permanent stone.
- A 2×2 partition gives an upper bound of 64 plies, since each 2×2 block can contain at most one stone.
- The rule is invariant under the dihedral symmetries of the square: rotations and reflections preserve legality and scoring.
- Arbitrary translations are not global symmetries because the board has finite boundaries.
- Swapping colors swaps geometric scoring, but does not change the fixed Black-first protocol into a separate legal game state.
- These rules do not constitute a proof of the optimal first-player result or a complete solution of the 16×16 game.

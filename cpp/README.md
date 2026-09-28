# Native Module

Stage 12.1 contains the profile-validated C++20/pybind11 acceleration layer.
The `native._hexadeca_native` extension owns the hot board state, incremental
legality, Zobrist hashing, strict squared-Euclidean terminal scoring, feature
packing, and PUCT tree traversal. Python remains responsible for immutable
public DTOs, AlphaZero orchestration, and centralized PyTorch GPU inference.

The extension receives board and MCTS parameters from validated Python config;
it does not embed an alternate ruleset or search configuration. `mcts.engine`
selects `"native"` for production or `"reference"` for diagnostics and parity
benchmarks.

`NativeSearchTree.leaf_states(...)` is the preferred inference boundary for a
batch of pending leaves.  It copies all native boards while the GIL is released
and materializes one ordered Python payload only after the copy completes;
selection, virtual-loss reservation, expansion, and backup remain in C++.

Build it during the normal package installation:

```powershell
python -m pip install -e ".[dev]"
```

Windows requires the MSVC C++20 build tools and a Windows SDK. Linux cloud
deployments require a C++20 compiler. In both cases pybind11 is declared as a
build dependency.

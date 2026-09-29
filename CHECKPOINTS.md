# Checkpoint quick evaluation

Hexadeca supports checkpoint-based interactive evaluation on its 16x16 board,
so you can open the monitoring UI without retraining. This is a research
project; use compatible checkpoints with known provenance.

## Published checkpoints

Five original 16x16 checkpoint bundles are published through **Git LFS**.
They are the same training snapshots used in the README's five-game visual
demonstration. Their IDs identify training iterations, not strength rankings.

| Checkpoint ID | State file size | Suggested use |
|---|---:|---|
| `iteration-001360` | 38,179,937 bytes | Earlier snapshot / comparison baseline |
| `iteration-001910` | 38,179,937 bytes | Intermediate snapshot |
| `iteration-002880` | 38,179,937 bytes | Intermediate snapshot |
| `iteration-003800` | 38,179,937 bytes | Later snapshot |
| `iteration-004520` | 38,179,937 bytes | Default interactive demo |

Total state data: **190,899,685 bytes (about 191 MB / 182 MiB)**, plus small
JSON manifests. All use `hexadeca-v1`, the `hexadeca-v1-16p` encoder, and a
128-channel, 10-block residual network. The bundles retain their original
optimizer/scheduler state and SHA-256 checksums; interactive evaluation loads
only the model. Replay buffers and the full training history are not included.
The published bundles are covered by the repository's MIT license.

## Install, verify, launch

Requirements: Python 3.11-3.13, Git, Git LFS, and a POSIX shell (Linux/macOS or
Git Bash on Windows). The existing editable build compiles the C++20 extension,
so a C++20 compiler and Python development headers must be available.

```bash
# Clone with Git LFS installed, then enter the repository:
git clone https://github.com/srotn/hexadeca.git
cd hexadeca
```

For an automatic environment setup and launch:

```bash
bash scripts/download-checkpoints.sh
bash scripts/quickstart.sh iteration-004520
```

Or install dependencies explicitly before verifying checkpoints and launching:

```bash
# From the repository root:
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
bash scripts/download-checkpoints.sh
bash scripts/quickstart.sh iteration-004520
```

With compatible weights already present, you can simply run:

```bash
bash scripts/quickstart.sh iteration-004520
```

The helper creates `.venv` if needed, activates it, and installs this checkout
in editable mode when necessary. No development extras are needed. It checks
weights before environment setup, so missing models do not trigger a large
dependency installation. Git Bash also supports `.venv/Scripts/activate`.

Default checkpoint: `iteration-004520`. Default port: `5555`. Open
**http://localhost:5555** once the server reports startup complete. Stop with
Ctrl+C. The CLI binds locally and uses the existing inference-only checkpoint
loading path; launching it starts no training job.

After installation and checkpoint preparation, the direct command is:

```bash
source .venv/bin/activate
python -m monitoring --checkpoint iteration-004520
```

Choose another checkpoint or port:

```bash
bash scripts/quickstart.sh iteration-001360 5566
# Equivalent after activating the environment:
python -m monitoring --checkpoint iteration-001360 --bind-port 5566
```

## Checkpoint layout and Git LFS

The current runtime requires both files for each checkpoint:

```text
checkpoints/
  bundles/
    iteration-001360/
      manifest.json
      state.pt
    iteration-001910/
      manifest.json
      state.pt
    iteration-002880/
      manifest.json
      state.pt
    iteration-003800/
      manifest.json
      state.pt
    iteration-004520/
      manifest.json
      state.pt
```

Alias files and parent checkpoints are unnecessary for an explicit ID. The
downloader checks all five published bundles; quickstart only requires the
selected one. The existing
`CheckpointManager` validates the manifest, network compatibility, and SHA-256
checksum before loading tensor data.

A flat `checkpoints/iteration-001360.pt` or `checkpoints/iteration-004520.pt`
**alone is not a loadable bundle**. The downloader reports flat artifacts if
present, but needs a complete bundle before reporting readiness. Obtain the
matching manifest and state file; do not fabricate metadata. The `.pth` and
`.tar` LFS patterns are storage rules, not additional runtime loading formats.

Checkpoint files may be large and should be managed with **Git LFS**. Root
`.pt`, `.pth`, `.tar` files and canonical `bundles/*/state.pt` use LFS; small
manifests remain ordinary JSON. Existing ignore rules keep local training
artifacts out of accidental commits; only the five listed bundles are tracked.
Use a Git clone and `git lfs pull` to obtain the binaries; a source archive
may contain LFS pointers rather than weights.

## Troubleshooting

- **Missing or incomplete checkpoint:** run
  `bash scripts/download-checkpoints.sh` from an up-to-date clone of `main`.
  If tracked manifests were deleted locally, restore the affected bundle from
  Git before retrying. Custom checkpoint IDs require their own complete bundle.
- **Git LFS unavailable:** install Git LFS and rerun the downloader. It runs
  `git lfs install --local` and `git lfs pull` from this repository.
- **Small text file instead of weights:** likely an LFS pointer. Both scripts
  reject pointers; rerun the downloader with LFS access.
- **LFS download fails:** check network access and the error from Git LFS.
  GitHub bandwidth or storage limits can also block downloads; retry after the
  limit or access issue is resolved. Do not replace weights with pointer text.
- **Python or build failure:** use Python 3.11-3.13 with `venv`, `pip`, and C++20
  build tools. Select an interpreter when creating `.venv` with
  `PYTHON=python3.12 bash scripts/quickstart.sh`. Rename an incomplete `.venv`
  before recreating it.
- **Port in use:** run `bash scripts/quickstart.sh iteration-004520 5566` and
  open `http://localhost:5566`.
- **Checksum or network mismatch:** obtain the matching original manifest
  and state file for the public `hexadeca-v1` 16x16 network. Do not edit the
  checksum to silence the error.
- **Slow search or limited GPU memory:** lower the search budget in the UI.
  For CPU use, create a local TOML profile with `[monitoring]` and
  `training_device = "cpu"`, then pass `--profile path/to/profile.toml` to
  the direct monitoring command.

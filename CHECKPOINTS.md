# Checkpoint quick evaluation

Hexadeca supports checkpoint-based interactive evaluation on its 16x16 board,
so you can open the monitoring UI without retraining. This is a research
project; use compatible checkpoints with known provenance.

**Availability:** this source release does not currently publish checkpoint
binaries through Git LFS. The downloader fetches published LFS objects and
checks local files; it cannot fetch unpublished weights. For now, obtain
complete compatible bundles from the maintainer and use the layout below.
Adding LFS attributes does not upload any model files.

## Install, verify, launch

Requirements: Python 3.11-3.13, Git, Git LFS, and a POSIX shell (Linux/macOS or
Git Bash on Windows). The existing editable build compiles the C++20 extension,
so a C++20 compiler and Python development headers must be available.

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
    iteration-004520/
      manifest.json
      state.pt
```

Alias files are unnecessary for an explicit ID. The downloader checks both
example bundles; quickstart only requires the selected one. The existing
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
artifacts out of accidental commits. A maintainer must explicitly publish
approved bundles before `git lfs pull` can fetch them.

## Troubleshooting

- **Missing or incomplete checkpoint:** run
  `bash scripts/download-checkpoints.sh`. If no weights are published for this
  checkout, obtain both bundle files from the maintainer. A successful
  `git lfs pull` with no tracked weights does not mean a model was downloaded.
- **Git LFS unavailable:** install Git LFS and rerun the downloader. It runs
  `git lfs install --local` and `git lfs pull` from this repository.
- **Small text file instead of weights:** likely an LFS pointer. Both scripts
  reject pointers; rerun the downloader with LFS access.
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

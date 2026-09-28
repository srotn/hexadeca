# Hexadeca Checkpoints

This document describes the available pre-trained checkpoints for quick evaluation and interactive play.

## Available Checkpoints

| Checkpoint ID | Iteration | Training Status | Approximate Elo (vs iteration-001360) | Use Case |
|---|---|---|---|---|
| `iteration-001360` | 1360 | Early | 1000 (baseline) | Reference baseline for Elo calculations |
| `iteration-004520` | 4520 | Mid-training | ~1240 | Strong mid-training model; recommended for exploration |
| `iteration-latest` | Latest available | Current best | ~1380 | Strongest trained model |

## Quick Start

### 1. Interactive Play (Web UI)

```bash
# Ensure the environment is set up
source .venv/bin/activate

# Run the monitoring server with a checkpoint
python -m monitoring --checkpoint iteration-004520
```

Then open `http://localhost:5555` in your browser to:
- Play against the AI
- Watch self-play games
- Analyze terminal positions with feature heatmaps
- Adjust search depth (200–3200 simulations)

### 2. Command-line Arena Evaluation

Compare two checkpoints:

```bash
python -m benchmark.evaluation \
  --profile config/compact-inference.toml \
  --checkpoint iteration-004520
```

### 3. Analysis and Visualization

Extract and plot terminal game analysis:

```bash
python -m analysis.analyze_record analysis/records/game-from-ui.json
```

Results are written to `analysis/json/` and `analysis/png/`.

## Checkpoint Details

All checkpoints are trained with:
- **Config**: `config/default.toml`
- **Model**: ResNet backbone, 10 residual blocks, 128 channels
- **Feature schema**: `hexadeca-v1-16p` (16 input planes)
- **Self-play**: 1600 simulations per move, 32 games per iteration
- **MCTS**: PUCT with c_puct=2.3, Dirichlet noise, virtual loss=3
- **Training**: AdamW, warmup-cosine scheduler, AMP (float16) on CUDA

## Evaluation Protocol

Checkpoints are evaluated via:
1. **Round-robin arena**: 100 games each, 1600 simulations, color-swapped pairs
2. **Bootstrap confidence interval**: 95% with 10,000 samples
3. **Paired-color evaluation**: Both sides use identical model; only color alternation
4. **Bradley–Terry rating**: Relative Elo with iteration-001360 fixed at 1000

## File Locations

```
checkpoints/
  iteration-001360.pt          # Baseline checkpoint
  iteration-004520.pt          # Mid-training checkpoint
  iteration-latest.pt          # Best model (symlink or latest)
  .gitattributes               # Git LFS tracking (if files > 100 MB)
```

## Troubleshooting

### Checkpoint not found
Ensure you have cloned with Git LFS support:
```bash
git lfs install
git clone https://github.com/srotn/hexadeca.git
cd hexadeca
git lfs pull
```

### Out of memory during inference
Use a smaller batch size or CPU-only mode:
```bash
python -m monitoring --checkpoint iteration-004520 --device cpu
```

### Slow Web UI / Search
Adjust simulations in the UI (default 1600). Fewer simulations = faster but weaker AI.

## Citation

If you use these checkpoints in research, please cite:

```bibtex
@software{hexadeca2026,
  title={Hexadeca: AlphaZero-style Training Platform for 16×16 Spatial Strategy},
  author={srotn},
  year={2026},
  url={https://github.com/srotn/hexadeca}
}
```

And reference the evaluation metrics from this document.

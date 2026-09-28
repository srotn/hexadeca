#!/bin/bash
# quickstart.sh
# One-command setup and launch for interactive play.
# Usage: bash scripts/quickstart.sh [checkpoint_name] [port]

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${1:-iteration-004520}"
PORT="${2:-5555}"

echo "═══════════════════════════════════════════════════════════"
echo "Hexadeca Quickstart"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Step 1: Setup Python environment
if [ ! -d "$PROJECT_ROOT/.venv" ]; then
    echo "Step 1/3: Creating virtual environment..."
    cd "$PROJECT_ROOT"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -e ".[dev]"
    echo "✓ Virtual environment ready"
else
    echo "Step 1/3: Activating existing virtual environment..."
    source "$PROJECT_ROOT/.venv/bin/activate"
    echo "✓ Virtual environment activated"
fi

echo ""

# Step 2: Verify checkpoint
echo "Step 2/3: Verifying checkpoint..."
if [ ! -f "$PROJECT_ROOT/checkpoints/$CHECKPOINT.pt" ]; then
    echo "✗ Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "Available checkpoints:"
    ls -1 "$PROJECT_ROOT/checkpoints"/*.pt 2>/dev/null | xargs -I {} basename {} .pt || echo "  (none found; run scripts/download-checkpoints.sh)"
    exit 1
fi
echo "✓ Checkpoint ready: $CHECKPOINT"

echo ""

# Step 3: Launch monitoring server
echo "Step 3/3: Launching monitoring server on port $PORT..."
echo ""
echo "┌─────────────────────────────────────────────────────────┐"
echo "│ Server is starting...                                   │"
echo "│                                                         │"
echo "│ Open your browser:                                      │"
echo "│   http://localhost:$PORT                                │"
echo "│                                                         │"
echo "│ Press Ctrl+C to stop                                    │"
echo "└─────────────────────────────────────────────────────────┘"
echo ""

cd "$PROJECT_ROOT"
python -m monitoring --checkpoint "$CHECKPOINT" --bind-port "$PORT"

#!/bin/bash
# download-checkpoints.sh
# Utility script to ensure Git LFS checkpoints are available locally.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CHECKPOINTS_DIR="$PROJECT_ROOT/checkpoints"

echo "Ensuring Git LFS is initialized..."
git lfs install

echo "Pulling LFS files..."
cd "$PROJECT_ROOT"
git lfs pull

echo "Verifying checkpoints..."
if [ -f "$CHECKPOINTS_DIR/iteration-001360.pt" ]; then
    echo "✓ iteration-001360.pt found"
else
    echo "✗ iteration-001360.pt not found"
    exit 1
fi

if [ -f "$CHECKPOINTS_DIR/iteration-004520.pt" ]; then
    echo "✓ iteration-004520.pt found"
else
    echo "✗ iteration-004520.pt not found (optional)"
fi

echo ""
echo "Checkpoints ready. Quick start:"
echo "  source .venv/bin/activate"
echo "  python -m monitoring --checkpoint iteration-001360"
echo ""

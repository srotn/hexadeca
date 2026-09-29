#!/bin/sh
# Fetch published LFS artifacts, then check all five demo bundles.
set -eu

case $0 in /*|[A-Za-z]:/*) SCRIPT_PATH=$0 ;; *) SCRIPT_PATH=./$0 ;; esac
PROJECT_ROOT=$(CDPATH= cd -P "$(dirname "$SCRIPT_PATH")/.." && pwd)
cd "$PROJECT_ROOT"

if ! command -v git >/dev/null 2>&1; then
    printf '%s\n' 'Error: Git is required.' >&2
    exit 1
fi
if ! git lfs version >/dev/null 2>&1; then
    printf '%s\n' 'Error: install Git LFS, then rerun this script.' >&2
    exit 1
fi

printf '%s\n' 'Initializing Git LFS for this repository...'
git lfs install --local
printf '%s\n' 'Pulling published LFS artifacts...'
if ! git lfs pull; then
    printf '%s\n' \
        'Error: checkpoint download failed. Check Git LFS network access and GitHub quota.' \
        'See CHECKPOINTS.md; then rerun: bash scripts/download-checkpoints.sh' >&2
    exit 1
fi

valid_file() {
    [ -f "$1" ] && [ -s "$1" ] && [ -r "$1" ] &&
        ! LC_ALL=C grep -q '^version https://git-lfs.github.com/spec/v1' "$1"
}

missing=0
for checkpoint in iteration-001360 iteration-001910 iteration-002880 iteration-003800 iteration-004520; do
    # Flat artifacts may be distributed too, but the runtime needs metadata.
    if [ -f "checkpoints/$checkpoint.pt" ]; then
        if valid_file "checkpoints/$checkpoint.pt"; then
            printf 'Found flat artifact: checkpoints/%s.pt\n' "$checkpoint"
        else
            printf 'Incomplete or LFS pointer artifact: checkpoints/%s.pt\n' "$checkpoint" >&2
            missing=1
        fi
    fi
    bundle=checkpoints/bundles/$checkpoint
    if valid_file "$bundle/manifest.json" && valid_file "$bundle/state.pt"; then
        printf 'Ready: %s\n' "$bundle"
    else
        printf 'Missing, empty, unreadable, or unhydrated bundle: %s\n' "$bundle" >&2
        missing=1
    fi
done

if [ "$missing" -ne 0 ]; then
    printf '%s\n' \
        'A complete bundle needs manifest.json and state.pt.' \
        'Git LFS only downloads artifacts published for this checkout.' \
        'Use an up-to-date clone of main; restore any locally deleted tracked bundle files.' \
        'See CHECKPOINTS.md; then rerun: bash scripts/download-checkpoints.sh' >&2
    exit 1
fi
printf '%s\n' 'Checkpoint files verified. Quick start:' \
    '  bash scripts/quickstart.sh iteration-004520'

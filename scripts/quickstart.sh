#!/bin/sh
# Usage: bash scripts/quickstart.sh [checkpoint_id] [port]
set -eu

case $0 in /*|[A-Za-z]:/*) SCRIPT_PATH=$0 ;; *) SCRIPT_PATH=./$0 ;; esac
PROJECT_ROOT=$(CDPATH= cd -P "$(dirname "$SCRIPT_PATH")/.." && pwd)
cd "$PROJECT_ROOT"
if [ "$#" -gt 2 ]; then
    printf '%s\n' 'Usage: bash scripts/quickstart.sh [checkpoint_id] [port]' >&2
    exit 2
fi
CHECKPOINT=${1-iteration-004520}
PORT=${2-5555}
case $CHECKPOINT in
    ''|[!A-Za-z0-9]*|*[!A-Za-z0-9._-]*)
        printf '%s\n' 'Error: use a checkpoint ID such as iteration-004520 (without .pt).' >&2
        exit 2 ;;
esac
if [ "${#CHECKPOINT}" -gt 128 ]; then
    printf '%s\n' 'Error: checkpoint ID must be at most 128 characters.' >&2
    exit 2
fi
case $PORT in
    ''|*[!0-9]*)
        printf '%s\n' 'Error: port must be an integer from 1 to 65535.' >&2
        exit 2 ;;
esac
if [ "${#PORT}" -gt 5 ] || [ "$PORT" -lt 1 ] || [ "$PORT" -gt 65535 ]; then
    printf '%s\n' 'Error: port must be an integer from 1 to 65535.' >&2
    exit 2
fi

# Check weights before creating a venv or installing dependencies.
BUNDLE=checkpoints/bundles/$CHECKPOINT
for file in "$BUNDLE/manifest.json" "$BUNDLE/state.pt"; do
    if [ ! -f "$file" ] || [ ! -s "$file" ] || [ ! -r "$file" ] ||
        LC_ALL=C grep -q '^version https://git-lfs.github.com/spec/v1' "$file"; then
        printf 'Checkpoint missing or incomplete: %s\nExpected: %s\n' "$CHECKPOINT" "$file" >&2
        if [ -f "checkpoints/$CHECKPOINT.pt" ]; then
            printf '%s\n' 'A standalone .pt is not a loadable bundle; obtain its matching manifest and state.pt.' >&2
        fi
        printf '%s\n' 'Run: bash scripts/download-checkpoints.sh' 'See CHECKPOINTS.md for availability and layout.' >&2
        exit 1
    fi
done

if [ ! -d .venv ]; then
    if [ -n "${PYTHON:-}" ]; then
        :
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON=python3
    else
        PYTHON=python
    fi
    if ! command -v "$PYTHON" >/dev/null 2>&1; then
        printf '%s\n' 'Error: install Python 3.11-3.13 or set PYTHON to its executable.' >&2
        exit 1
    fi
    printf '%s\n' 'Creating .venv...'
    "$PYTHON" -m venv .venv
fi

# POSIX activation; Git Bash with Windows Python uses Scripts/.
if [ -f .venv/bin/activate ]; then
    . ./.venv/bin/activate
elif [ -f .venv/Scripts/activate ]; then
    . ./.venv/Scripts/activate
else
    printf '%s\n' 'Error: .venv is incomplete. Rename it and rerun to create a new environment.' >&2
    exit 1
fi
if ! python -c 'import sys; sys.exit(not (3, 11) <= sys.version_info[:2] < (3, 14))'; then
    printf '%s\n' 'Error: Python 3.11-3.13 is required.' >&2
    exit 1
fi

# A wheel or a package from another checkout is not this editable install.
if ! python - <<'PY'
import json
import sys
import sysconfig
from importlib.metadata import distributions
from importlib.util import find_spec
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname

try:
    # Ignore the source tree's egg-info; inspect the active venv installation.
    installed = distributions(path=[sysconfig.get_path('purelib'), sysconfig.get_path('platlib')])
    package = next(item for item in installed if item.metadata['Name'].lower().replace('_', '-') == 'hexadeca-alpha-zero')
    info = json.loads(package.read_text('direct_url.json') or '{}')
    source = Path(url2pathname(urlsplit(info.get('url', '')).path)).resolve()
    ready = info.get('dir_info', {}).get('editable') and source == Path.cwd()
    ready = ready and all(find_spec(name) for name in ('torch', 'fastapi', 'uvicorn', 'psutil', 'tensorboard'))
except (StopIteration, ValueError, OSError):
    ready = False
sys.exit(0 if ready else 1)
PY
then
    printf '%s\n' 'Installing this checkout and its runtime dependencies...'
    python -m pip install -e .
fi

printf 'Starting Hexadeca with %s\nOpen http://localhost:%s\nPress Ctrl+C to stop.\n' "$CHECKPOINT" "$PORT"
exec python -m monitoring --checkpoint "$CHECKPOINT" --bind-port "$PORT"

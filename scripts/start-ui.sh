#!/usr/bin/env bash
# Start the browser UI (installed with: bash scripts/install.sh ui). Ctrl+C stops it.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  echo "No .venv found: run  bash scripts/install.sh ui  first." >&2
  exit 1
fi
exec .venv/bin/python -m simulator ui "$@"

#!/usr/bin/env bash
# Install Scientific Data Simulator into ./.venv (Linux and macOS).
#
#   scripts/install.sh [PROFILE]
#
# PROFILE: core | standard (default) | all | dev | ui
#   core      simulate, store CSV/NPY, interactive plots
#   standard  + static plot export, Parquet, HDF5, physical units
#   all       + dataset validation
#   dev       + tests and their dependencies (for contributors)
#   ui        standard + the browser UI (start it with: sds ui)
# Set PYTHON=/path/to/python3 to choose the interpreter (3.9 or newer).
# Works with the bash 3.2 that macOS ships.
set -euo pipefail

PROFILE="${1:-standard}"
case "$PROFILE" in
  core) EXTRAS="" ;;
  standard|all|dev) EXTRAS="[$PROFILE]" ;;
  ui) EXTRAS="[standard]" ;;
  -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
  *) echo "Unknown profile '$PROFILE' (core, standard, all, dev, ui)" >&2; exit 2 ;;
esac

cd "$(dirname "$0")/.."

find_python() {
  # versioned names first: on macOS, /usr/bin/python3 is the old system Python (and on a fresh
  # Mac running it opens the developer-tools installer), so it is used only as a last resort
  for candidate in "${PYTHON:-}" python3.13 python3.12 python3.11 python3.10 python3 python python3.9; do
    [ -z "$candidate" ] && continue
    path="$(command -v "$candidate" 2>/dev/null)" || continue
    if [ "$(uname -s)" = "Darwin" ] && [ "$path" = "/usr/bin/python3" ] && [ -z "${PYTHON:-}" ] &&
       ! xcode-select -p >/dev/null 2>&1; then
      continue
    fi
    if "$path" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1; then
      echo "$path"
      return 0
    fi
  done
  return 1
}

if ! PY="$(find_python)"; then
  echo "Python 3.9 or newer was not found." >&2
  echo "  macOS: install it from python.org, with Homebrew (brew install python) or uv." >&2
  echo "  Linux: install python3 and python3-venv with your package manager." >&2
  exit 1
fi
echo "Using $("$PY" -c 'import sys; print(sys.executable, sys.version.split()[0])')"

if [ -d .venv ] && ! { [ -x .venv/bin/python ] && .venv/bin/python -m pip --version >/dev/null 2>&1; }; then
  if [ -f .venv/pyvenv.cfg ] && [ ! -d .venv/Scripts ]; then
    echo "Removing an incomplete .venv from an earlier attempt ..."
    rm -rf .venv
  else
    echo ".venv exists but is not a working Linux/macOS environment (made on Windows?)." >&2
    echo "Remove it, or use another folder, then run this again." >&2
    exit 1
  fi
fi
if [ ! -d .venv ]; then
  echo "Creating .venv ..."
  if ! "$PY" -m venv .venv || ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
    rm -rf .venv
    echo "Could not create the virtual environment (on Debian/Ubuntu: sudo apt install python3-venv)." >&2
    exit 1
  fi
fi

echo "Installing profile '$PROFILE' ..."
.venv/bin/python -m pip install --upgrade pip >/dev/null
.venv/bin/python -m pip install -e ".$EXTRAS"
if [ "$PROFILE" = "ui" ]; then
  .venv/bin/python -m pip install -e ./ui
fi
echo "$PROFILE" > .venv/sds-profile

echo
.venv/bin/python -m simulator doctor || true
echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "Then try:             sds examples list"
if [ "$PROFILE" = "ui" ]; then
  echo "Start the UI:         sds ui      (or: bash scripts/start-ui.sh)"
fi

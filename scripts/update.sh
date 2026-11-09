#!/usr/bin/env bash
# Update Scientific Data Simulator (Linux and macOS): pull the latest code (a git checkout on a
# branch with an upstream), reinstall the profile chosen at install time, and run the doctor.
#
#   scripts/update.sh [PROFILE]     # PROFILE overrides the remembered one
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x .venv/bin/python ]; then
  echo "No .venv found: run scripts/install.sh first." >&2
  exit 1
fi
PROFILE="${1:-$(cat .venv/sds-profile 2>/dev/null || echo standard)}"
case "$PROFILE" in
  core) EXTRAS="" ;;
  standard|all|dev) EXTRAS="[$PROFILE]" ;;
  *) echo "Unknown profile '$PROFILE' (core, standard, all, dev)" >&2; exit 2 ;;
esac

if [ -d .git ] && command -v git >/dev/null 2>&1; then
  if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "You have uncommitted changes; not pulling. Commit or stash them, then run again." >&2
  elif git rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
    echo "Pulling the latest code ..."
    git pull --ff-only
  else
    echo "This branch has no upstream; skipping git pull."
  fi
else
  echo "Not a git checkout; updating the installed packages only."
fi

echo "Reinstalling profile '$PROFILE' ..."
.venv/bin/python -m pip install --upgrade pip >/dev/null
.venv/bin/python -m pip install -e ".$EXTRAS"
echo "$PROFILE" > .venv/sds-profile
echo
.venv/bin/python -m simulator doctor || true

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
  echo "Creating automap virtual environment..."
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi

if [[ ! -f checkpoints/note_policy.pt ]]; then
  echo "No checkpoint yet; training on the approved map first..."
  .venv/bin/python -m automap train --epochs 60
fi

exec .venv/bin/python -m automap ui "$@"

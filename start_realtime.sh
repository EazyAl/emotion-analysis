#!/usr/bin/env bash

set -euo pipefail

echo "🚀 Starting real-time emotion analysis server"
echo "---------------------------------------------"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -d "venv" ]]; then
  echo "• Activating virtual environment"
  # shellcheck disable=SC1091
  source "venv/bin/activate"
fi

python realtime_server.py



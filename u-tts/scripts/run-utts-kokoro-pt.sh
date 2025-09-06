#!/usr/bin/env bash
set -euo pipefail

# Minimal launcher for UTTS (Kokoro PyTorch) on macOS/Linux
# - Uses Python 3.12 venv by default at ~/venvs/kokoro_pt
# - Requires: pip install "kokoro==0.9.4" "misaki[ja]" soundfile unidic-lite fugashi

VENV_DIR=${KOKORO_PT_VENV:-"$HOME/venvs/kokoro_pt"}
PORT=${UTTS_PORT:-9051}
HOST=${UTTS_HOST:-0.0.0.0}

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "[UTTS] venv not found: $VENV_DIR" >&2
  echo "Create it with:" >&2
  echo "/opt/homebrew/bin/python3.12 -m venv $VENV_DIR && source $VENV_DIR/bin/activate && pip install -U pip && pip install 'kokoro==0.9.4' 'misaki[ja]' soundfile unidic-lite fugashi" >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
UTTS_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

source "$VENV_DIR/bin/activate"
export ENABLE_KOKORO_PT=${ENABLE_KOKORO_PT:-true}
export UTTSDATA=${UTTSDATA:-"$UTTS_DIR"}

echo "[UTTS] Starting with Kokoro-PT at http://$HOST:$PORT"
echo "[UTTS] ENABLE_KOKORO_PT=$ENABLE_KOKORO_PT  UTTSDATA=$UTTSDATA"
python -m uvicorn server:app --host "$HOST" --port "$PORT"


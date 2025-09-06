#!/usr/bin/env bash
set -euo pipefail

# Minimal launcher for UTTS (Kokoro ONNX) on macOS/Linux
# - Uses venv at ~/venvs/kokoro_onnx by default
# - Requires: pip install kokoro-onnx "misaki[ja]" soundfile

VENV_DIR=${KOKORO_ONNX_VENV:-"$HOME/venvs/kokoro_onnx"}
PORT=${UTTS_PORT:-9051}
HOST=${UTTS_HOST:-0.0.0.0}

if [ -z "${KOKORO_ONNX_MODEL:-}" ] || [ -z "${KOKORO_ONNX_VOICES:-}" ]; then
  echo "[UTTS] Set KOKORO_ONNX_MODEL and KOKORO_ONNX_VOICES to model paths" >&2
  echo "Example:" >&2
  echo "  export KOKORO_ONNX_MODEL=\"$HOME/models/kokoro/kokoro-v1.0.onnx\"" >&2
  echo "  export KOKORO_ONNX_VOICES=\"$HOME/models/kokoro/voices-v1.0.bin\"" >&2
  exit 1
fi

if [ ! -f "$VENV_DIR/bin/activate" ]; then
  echo "[UTTS] venv not found: $VENV_DIR" >&2
  echo "Create it with:" >&2
  echo "/opt/homebrew/bin/python3.12 -m venv $VENV_DIR && source $VENV_DIR/bin/activate && pip install -U pip && pip install kokoro-onnx 'misaki[ja]' soundfile" >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
UTTS_DIR=$(cd "$SCRIPT_DIR/.." && pwd)

source "$VENV_DIR/bin/activate"
export UTTSDATA=${UTTSDATA:-"$UTTS_DIR"}

echo "[UTTS] Starting with Kokoro-ONNX at http://$HOST:$PORT"
echo "[UTTS] MODEL=$KOKORO_ONNX_MODEL  VOICES=$KOKORO_ONNX_VOICES  UTTSDATA=$UTTSDATA"
python -m uvicorn server:app --host "$HOST" --port "$PORT"


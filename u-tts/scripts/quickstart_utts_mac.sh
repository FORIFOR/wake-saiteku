#!/usr/bin/env bash
set -euo pipefail

# Quickstart for macOS: install deps, launch UTTS, synth test WAV, and play it.

if [[ "${ZSH_VERSION:-}" != "" ]]; then
  set -o no_nomatch || true
fi

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)

echo "[UTTS] Quickstart (macOS)"

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Please install Homebrew first: https://brew.sh" >&2
  exit 1
fi

PY=${PYTHON:-python3}
PIP="$PY -m pip"

echo "[1/5] Installing Python packages (FastAPI/Uvicorn/Piper)"
$PIP install --upgrade pip wheel >/dev/null
$PIP install fastapi 'uvicorn[standard]' pydantic piper-tts >/dev/null

echo "[2/5] Installing Open JTalk + voices via Homebrew"
brew install open-jtalk >/dev/null || true
brew install open-jtalk-mecab-naist-jdic >/dev/null || true
brew install hts-voice-nitech-jp-atr503-m001 >/dev/null || true
brew install sox >/dev/null || true

export UTTSDATA="$HOME/u-tts"
export OPENJTALK_DICT_DIR="$(brew --prefix)/share/open_jtalk/open_jtalk_dic_utf_8-1.11"
export OPENJTALK_VOICE="$(brew --prefix)/share/hts-voice/nitech_jp_atr503_m001/nitech_jp_atr503_m001.htsvoice"
mkdir -p "$UTTSDATA/models/piper" "$UTTSDATA/cache"

echo "[3/5] Launching UTTS server (background)"
cd "$ROOT"
pushd "$ROOT" >/dev/null
cd "$ROOT"

cd "$ROOT"/../ 2>/dev/null || true
cd "$ROOT" 2>/dev/null || true

pushd "$ROOT" >/dev/null

cd "$ROOT"/.. 2>/dev/null || true
cd "$ROOT" 2>/dev/null || true

cd "$ROOT"/../.. 2>/dev/null || true
cd "$ROOT" 2>/dev/null || true

cd "$ROOT"/.. 2>/dev/null || true
cd "$ROOT" 2>/dev/null || true

cd "$ROOT"/u-tts

# Start background server
UTTS_PORT=${UTTS_PORT:-9051}
UTTS_HOST=${UTTS_HOST:-127.0.0.1}
(
  export UTTSDATA OPENJTALK_DICT_DIR OPENJTALK_VOICE
  exec uvicorn server:app --host "$UTTS_HOST" --port "$UTTS_PORT"
) >/tmp/utts.quickstart.log 2>&1 &
UTTS_PID=$!
trap 'kill $UTTS_PID 2>/dev/null || true' EXIT

echo "[4/5] Waiting for /health ..."
for i in $(seq 1 40); do
  if curl -sS "http://$UTTS_HOST:$UTTS_PORT/health" | grep -q '"status":"ok"'; then
    echo "[UTTS] /health OK"
    break
  fi
  sleep 0.5
  if ! kill -0 $UTTS_PID 2>/dev/null; then
    echo "[UTTS] Server died. See /tmp/utts.quickstart.log" >&2
    exit 1
  fi
  if [[ $i -eq 40 ]]; then
    echo "[UTTS] Timeout waiting for server" >&2
    exit 1
  fi
done

echo "[5/5] Synthesizing using OpenJTalk (out.wav)"
curl -sS -X POST "http://$UTTS_HOST:$UTTS_PORT/tts" \
  -H "Content-Type: application/json" \
  -d '{"text":"本日は晴天なり。","engine":"openjtalk","speed":1.1}' \
  --output out.wav
ls -lh out.wav || true

if command -v afplay >/dev/null 2>&1; then
  echo "[UTTS] Playing out.wav (afplay)"
  afplay out.wav || true
elif command -v play >/dev/null 2>&1; then
  echo "[UTTS] Playing out.wav (sox)"
  play out.wav || true
fi

echo "[UTTS] Done. Server logs: /tmp/utts.quickstart.log"


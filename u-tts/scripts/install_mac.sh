#!/usr/bin/env bash
set -euo pipefail

echo "[UTTS] Installing dependencies for macOS (Piper + OpenJTalk)"

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Please install Homebrew first: https://brew.sh" >&2
  exit 1
fi

python3 -m venv .venv || true
source .venv/bin/activate
pip install --upgrade pip wheel
pip install piper-tts fastapi uvicorn[standard] pydantic

brew install open-jtalk open-jtalk-mecab-naist-jdic hts-voice-nitech-jp-atr503-m001 sox

mkdir -p "$HOME/u-tts/models/piper" "$HOME/u-tts/cache"

cat << EOF

Done.
Set environment variables (add to your shell profile):
  export UTTSDATA=$HOME/u-tts
  export OPENJTALK_DICT_DIR=$(brew --prefix)/share/open_jtalk/open_jtalk_dic_utf_8-1.11
  export OPENJTALK_VOICE=$(brew --prefix)/share/hts-voice/mei/mei_normal.htsvoice
  export PYTORCH_ENABLE_MPS_FALLBACK=1  # for optional XTTS on Apple silicon

Start server:
  cd u-tts && uvicorn server:app --host 0.0.0.0 --port 9051

Place a Piper model into: $HOME/u-tts/models/piper (e.g., ja_JP-*.onnx + .onnx.json)
EOF

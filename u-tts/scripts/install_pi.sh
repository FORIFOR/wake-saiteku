#!/usr/bin/env bash
set -euo pipefail

echo "[UTTS] Installing dependencies for Raspberry Pi (Piper + OpenJTalk)"
echo "This script assumes a Debian/Ubuntu-based system with apt."

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found. Please install Python 3.9+" >&2
  exit 1
fi

if ! command -v pip3 >/dev/null 2>&1; then
  echo "pip3 not found. Installing..." >&2
  sudo apt update && sudo apt install -y python3-pip
fi

echo "[1/3] Install Piper (pip)"
pip3 install --upgrade pip wheel
pip3 install piper-tts

echo "[2/3] Install Open JTalk packages"
sudo apt update
sudo apt install -y open-jtalk open-jtalk-mecab-naist-jdic hts-voice-nitech-jp-atr503-m001 sox

echo "[3/3] Create directories"
mkdir -p "$HOME/u-tts/models/piper" "$HOME/u-tts/cache"

cat << EOF

Done.
Next steps:
  export UTTSDATA=$HOME/u-tts
  export OPENJTALK_DICT_DIR=/var/lib/mecab/dic/open-jtalk/naist-jdic
  export OPENJTALK_VOICE=/usr/share/hts-voice/mei/mei_normal.htsvoice

Start server:
  cd u-tts
  python3 -m venv .venv && source .venv/bin/activate
  pip install fastapi uvicorn[standard] pydantic
  cd u-tts && uvicorn server:app --host 0.0.0.0 --port 9051

Place a Piper model into: $HOME/u-tts/models/piper (e.g., ja_JP-*.onnx + .onnx.json)
EOF

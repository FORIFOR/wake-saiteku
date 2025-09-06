#!/usr/bin/env bash
set -euo pipefail

log() { echo "[UTTS] $*"; }

log "Installing Python packages (fastapi, uvicorn, piper-tts, pydantic)"
pip install -U pip wheel >/dev/null
pip install fastapi 'uvicorn[standard]' piper-tts pydantic

log "Installing Homebrew formulae (open-jtalk, sox)"
brew install open-jtalk || true
brew install sox || true

export UTTSDATA="${UTTSDATA:-$HOME/u-tts}"

# Auto-detect OpenJTalk dictionary and voice if not provided
if [[ -z "${OPENJTALK_DICT_DIR:-}" ]]; then
  DETECTED_DICT=$(find "$(brew --prefix)" -type d -name 'open_jtalk_dic_*' 2>/dev/null | head -n1 || true)
  if [[ -n "$DETECTED_DICT" ]]; then
    export OPENJTALK_DICT_DIR="$DETECTED_DICT"
  fi
fi

if [[ -z "${OPENJTALK_VOICE:-}" ]]; then
  DETECTED_VOICE=$(find "$(brew --prefix)" -type f -name '*.htsvoice' 2>/dev/null | grep -E 'nitech|mei' | head -n1 || true)
  if [[ -n "$DETECTED_VOICE" ]]; then
    export OPENJTALK_VOICE="$DETECTED_VOICE"
  fi
fi

if [[ -z "${OPENJTALK_DICT_DIR:-}" || -z "${OPENJTALK_VOICE:-}" ]]; then
  log "OpenJTalk dict/voice not found automatically. You can set them manually, e.g.:"
  log "  export OPENJTALK_DICT_DIR=\"\$(find \"\$(brew --prefix)\" -type d -name 'open_jtalk_dic_*' | head -n1)\""
  log "  export OPENJTALK_VOICE=\"\$(find \"\$(brew --prefix)\" -type f -name '*.htsvoice' | head -n1)\""
  log "Continuing; OpenJTalk self-test may fail unless these are set."
fi

log "Self-test (this tries Piper/OpenJTalk synthesis; non-fatal)"
python u-tts/scripts/selftest.py || true

log "Starting UTTS on 127.0.0.1:9051"
(cd u-tts && nohup uvicorn server:app --host 127.0.0.1 --port 9051 >/tmp/utts.log 2>&1 & echo $! >/tmp/utts.pid)
sleep 2

log "Requesting TTS (OpenJTalk)"
json='{"text":"音声合成のテストです。","engine":"openjtalk","speed":1.1}'
curl -s -X POST http://127.0.0.1:9051/tts -H 'Content-Type: application/json' -d "$json" -o /tmp/utts_test.wav

log "Playing /tmp/utts_test.wav"
afplay /tmp/utts_test.wav

log "Done. To stop UTTS: kill $(cat /tmp/utts.pid 2>/dev/null || echo '<pid>') or pkill -f 'uvicorn server:app --host 127.0.0.1 --port 9051'"

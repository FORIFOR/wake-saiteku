# Unified TTS Service (UTTS)

FastAPI-based TTS microservice that exposes a common API across Raspberry Pi 5 and macOS.

Endpoints
- GET `/health`: {"status":"ok","engines":["piper","openjtalk", ...]}
- GET `/voices?engine=piper|openjtalk|kokoro_onnx|kokoro_pt|xtts`
- POST `/tts` (JSON) → audio/wav

Request body
```
{
  "text": "おはようございます。",
  "engine": "piper",          // default: piper -> openjtalk fallback
  "lang": "ja",               // optional
  "voice": "ja_test",         // piper model stem; openjtalk uses env voice
  "speed": 1.0,                // 0.5–2.0 typical
  "pitch": 0.0,                // semitones; applied via sox when present
  "gain_db": 0.0,
  "sample_rate": 22050,        // 8000/16000/22050/24000/48000
  "cache": true,
  "reference_wav": null        // XTTS etc. (future, mac only)
}
```

Response headers
- `X-Engine`: selected engine
- `X-Voice`: voice id
- `X-Sample-Rate`: sample rate

Setup
1) Python env
- `python3 -m venv .venv && source .venv/bin/activate`
- `pip install --upgrade pip wheel`
- `pip install fastapi uvicorn[standard] pydantic piper-tts`

2) Install Open JTalk
- macOS: `brew install open-jtalk open-jtalk-mecab-naist-jdic hts-voice-nitech-jp-atr503-m001 sox`
- Pi: `sudo apt install -y open-jtalk open-jtalk-mecab-naist-jdic hts-voice-nitech-jp-atr503-m001 sox`

3) Env vars
- `export UTTSDATA=$HOME/u-tts`
- macOS: `export OPENJTALK_DICT_DIR=$(brew --prefix)/share/open_jtalk/open_jtalk_dic_utf_8-1.11`
- macOS: `export OPENJTALK_VOICE=$(brew --prefix)/share/hts-voice/mei/mei_normal.htsvoice`
- Pi: `export OPENJTALK_DICT_DIR=/var/lib/mecab/dic/open-jtalk/naist-jdic`
- Pi: `export OPENJTALK_VOICE=/usr/share/hts-voice/mei/mei_normal.htsvoice`

4) Models
- Piper: place models in `u-tts/models/piper` (e.g., `ja_test.onnx` + `ja_test.onnx.json`)

Run
- `cd u-tts && uvicorn server:app --host 0.0.0.0 --port 9051`

Service
- Pi (systemd): see `u-tts/config/utts.service` (adjust `User`, paths)
- macOS (launchd): `u-tts/config/com.saiteku.utts.plist` → `~/Library/LaunchAgents/`

Notes
- Pitch uses sox `pitch` (semitones → cents). If sox isn't present, pitch/gain are ignored.
- Open JTalk lists a single voice from `OPENJTALK_VOICE`. Add more wrappers if you maintain a voice directory.
- Kokoro/XTTS can be added later as plug-ins; server keeps API stable.

## Kokoro (日本語TTS) の有効化

Kokoroは品質確認向けのPyTorch版と、軽量で運用向けのONNX版を別venvで用意するのが安全です。詳細手順は `_docs/tts-kokoro-ja.md` を参照してください。

### Kokoro-ONNX を使う

1) 依存導入（別venv推奨）

```
pip install fastapi uvicorn[standard] pydantic
pip install kokoro-onnx "misaki[ja]" soundfile
```

2) モデル配置

```
export KOKORO_ONNX_MODEL=$HOME/models/kokoro/kokoro-v1.0.onnx
export KOKORO_ONNX_VOICES=$HOME/models/kokoro/voices-v1.0.bin
```

3) サーバー起動

```
UTTSDATA=$PWD uvicorn server:app --host 0.0.0.0 --port 9051
```

利用例:

```
curl -X POST 'http://127.0.0.1:9051/tts' \
  -H 'Content-Type: application/json' \
  -d '{"text":"こんにちは、日本語テストです。","engine":"kokoro_onnx","voice":"jf_alpha"}' \
  --output out.wav
```

### Kokoro (PyTorch) を使う

1) 依存導入（Python 3.12 venv 推奨）

```
pip install fastapi uvicorn[standard] pydantic
pip install "kokoro==0.9.4" "misaki[ja]" soundfile unidic-lite fugashi
```

2) 有効化フラグ

```
export ENABLE_KOKORO_PT=true
```

3) サーバー起動（上と同様）

利用例:

```
curl -X POST 'http://127.0.0.1:9051/tts' \
  -H 'Content-Type: application/json' \
  -d '{"text":"こんにちは、日本語テストです。","engine":"kokoro_pt","voice":"jf_alpha"}' \
  --output out.wav
```

クライアント再生（wake-saiteku）を使う場合は `.env` に `ENABLE_TTS_PLAYBACK=true` を設定し、 `SERVER_TTS_URL=http://<UTTSサーバーIP>:9051/tts` を指定してください。

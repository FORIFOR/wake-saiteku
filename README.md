Wake Saiteku
=================

「もしもしサイテク」で起動する軽量な日本語音声アシスタント。

特徴
- Wake Word検知: オフラインで「もしもし」「サイテク」を検知
- STT: サーバー側で faster-whisper による高精度文字起こし
- LLM: OpenAI互換API（LM Studio / llama.cpp / Ollama 等）に対応
- TTS: 任意。内蔵（sherpa-onnx）または別プロセス UTTS（Piper/OpenJTalk）
- 対応環境: Raspberry Pi 5（クライアント）/ macOS・Linux（サーバー）

ディレクトリ
- `server/`: FastAPI サーバー（STT + LLM + 任意TTS）
- `client/`: Wake検知 + 録音 + サーバー呼び出し
- `u-tts/`: 追加TTSサービス（Piper/OpenJTalk/Kokoro）。任意で使用
- `utils/`, `config/`, `scripts/`: ユーティリティと設定

前提
- Python 3.11+ 推奨
- マイク（USB推奨）/ スピーカー（任意、TTS再生用）
- Raspberry Pi 5 は `sudo apt install -y python3-venv libportaudio2` を推奨

クイックスタート
- 仮想環境作成: `python3 -m venv venv && source venv/bin/activate`
- 依存導入:
  - サーバー: `pip install -r requirements-server.txt`
  - クライアント: `pip install -r requirements-client.txt`
- サーバー起動（PC側）:
  - 環境例: `FW_MODEL=tiny FW_DEVICE=cpu FW_COMPUTE=int8 LM_URL=http://127.0.0.1:1234/v1/chat/completions`
  - 実行: `make run-server`（または `python server/server.py`）
- クライアント起動（Pi 5 など）:
  - サーバーURL: `export SERVER_URL=http://<サーバーIP>:8000/inference`
  - マイク（任意）: `export AUDIO_INPUT_DEVICE=2`（一覧: `python -c "import sounddevice as sd; print(sd.query_devices())"`）
  - 実行: `make run-client`（または `python client/client.py`）

Raspberry Pi 5 メモ
- Pi はクライアント運用が簡単（サーバーは PC 側）。
- 事前に `sudo apt update && sudo apt install -y python3-venv libportaudio2`。
- Pi でサーバーも動かす場合は `FW_MODEL=tiny` + `FW_COMPUTE=int8` を推奨。

TTS（任意）
- 内蔵TTSを使う（サーバー `/tts`）:
  - サーバー側で `TTS_ENABLED=true` とモデルパス（VITS 等）を設定。
  - クライアントで `ENABLE_TTS_PLAYBACK=true`（`SERVER_TTS_URL` 未指定なら `/inference` から `/tts` を自動導出）。
- 軽量な外部TTS（Pi 向け Piper/OpenJTalk）:
  - `bash u-tts/scripts/install_pi.sh`
  - `cd u-tts && uvicorn server:app --host 0.0.0.0 --port 9051`
  - クライアントに `ENABLE_TTS_PLAYBACK=true` と `SERVER_TTS_URL=http://<UTTS_IP>:9051/tts`

主な環境変数（例）
- サーバー
  - `FW_MODEL=tiny|base|small|…`（faster-whisper）
  - `FW_DEVICE=cpu|cuda`, `FW_COMPUTE=int8|fp16|…`
  - `LM_URL=http://…/v1/chat/completions`, `LM_MODEL=local-model`
  - `LLM_ROUTING=primary|local|auto`（フォールバック方針）
- クライアント
  - `SERVER_URL=http://<HOST>:8000/inference`（`/v1/audio/inference` も可）
  - `AUDIO_INPUT_DEVICE=<index or name>`
  - `ENABLE_TTS_PLAYBACK=true`（`SERVER_TTS_URL=http://<HOST>:9051/tts` など）

トラブルシュート
- デバイスエラー: `AUDIO_INPUT_DEVICE` を指定し、`sounddevice` の一覧で確認
- 接続不可: `SERVER_URL` のホスト/ポート、LAN/ファイアウォール設定を確認
- 応答遅延: `FW_MODEL=tiny` にする／LLM を小型化・ローカル化

ライセンス
- MIT License

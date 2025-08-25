# Wake Saiteku - 音声アシスタントシステム

「もしもしサイテク」で起動する音声アシスタントシステム

## 概要

Wake Saitekuは、Wake Word検知による音声アシスタントシステムです。「もしもしサイテク」という呼びかけで起動し、音声による対話が可能です。

### 特徴

- **Wake Word検知**: オフラインで「もしもしサイテク」を検知
- **高精度音声認識**: faster-whisperによる日本語音声認識
- **LLM連携**: ローカルLLMまたはリモートLLMによる応答生成
- **オフラインフォールバック**: ネットワーク障害時も動作継続
- **マルチプラットフォーム**: Raspberry Pi 5、Mac mini対応

## システム構成

```
┌─────────────────┐        ┌─────────────────┐
│   端末デバイス    │  HTTP  │   リモートPC     │
│ (Raspberry Pi 5) │ ──────▶│  (Mac mini等)    │
│                 │        │                 │
│ - Wake Word検知  │        │ - Whisper STT   │
│ - 音声録音       │        │ - LLM処理        │
│ - オフライン処理  │        │ - 応答生成       │
└─────────────────┘        └─────────────────┘
```

## 必要要件

### ハードウェア
- **端末**: Raspberry Pi 5 (4GB以上推奨) または Mac
- **サーバー**: Mac mini または Linux PC
- **マイク**: USB/3.5mmマイク

### ソフトウェア
- Python 3.8以上
- Git
- (Raspberry Pi) PortAudio

## クイックスタート

### 1. リポジトリのクローン

```bash
git clone https://github.com/yourusername/wake-saiteku.git
cd wake-saiteku
```

### 2. セットアップ

```bash
# セットアップスクリプトを実行
./scripts/setup.sh
```

セットアップ時に以下を選択:
- 1: サーバーのみ（リモートPC用）
- 2: クライアントのみ（端末用）
- 3: 両方

### 3. 設定ファイルの編集

#### サーバー側 (config/server.env)

```bash
# Whisper設定
FW_MODEL=small      # small/medium/large
FW_COMPUTE=int8     # int8/fp16 (Apple Siliconはint8_float16推奨)

# LLM設定
LM_URL=http://localhost:1234/v1/chat/completions
LM_MODEL=your-model-name
```

#### クライアント側 (config/client.env)

```bash
# サーバーのIPアドレスを設定
SERVER_URL=http://192.168.1.100:8000/inference

# オフラインフォールバック
LOCAL_STT_ENABLED=true
```

### 4. 起動

#### サーバー起動（リモートPC）

```bash
source venv/bin/activate
source config/server.env
python server/server.py
```

#### クライアント起動（端末）

```bash
source venv/bin/activate
source config/client.env
python client/client.py
```

## 使い方

1. クライアントを起動すると「Wake Word待機中」と表示されます
2. 「もしもしサイテク」と話しかけます
3. ビープ音の後、用件を話します
4. 自動的に音声の終了を検知し、応答が表示されます

## 詳細設定

### LM Studio連携

1. [LM Studio](https://lmstudio.ai/)をダウンロード
2. 日本語対応モデル（例: Qwen2.5-1.5B-Instruct）をロード
3. Developer → Local Serverでサーバーを起動
4. LANアクセスを許可

### オフライン動作（Raspberry Pi）

llama.cpp のセットアップ:

```bash
# ビルド
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
LLAMA_OPENBLAS=1 make -j

# モデルダウンロード（GGUF形式）
mkdir ~/models
cd ~/models
# Hugging Faceから適切なモデルをダウンロード

# サーバー起動
~/llama.cpp/llama-server -m ~/models/model.gguf --port 8081
```

### Systemdサービス化（Raspberry Pi）

```bash
# サービスファイルをコピー
sudo cp config/saiteku-client.service /etc/systemd/system/

# サービスを有効化
sudo systemctl daemon-reload
sudo systemctl enable --now saiteku-client
```

### LaunchAgent（macOS, サーバー常駐）

サーバーをmacOSで常駐させる場合は LaunchAgent を使います。

```bash
# Python 3.11 環境を作成（初回のみ）
/opt/homebrew/bin/python3.11 -m venv venv311
source venv311/bin/activate
pip install --upgrade pip
pip install -r requirements-server.txt || \
  pip install fastapi uvicorn[standard] soundfile requests python-multipart "faster-whisper==0.10.0"

# LaunchAgent を作成/読み込み（デフォルト: 127.0.0.1:9050）
bash scripts/install-macos-launchd.sh

# 起動確認
launchctl list | grep com.saiteku.server
curl -sS http://127.0.0.1:9050/

# ログ
tail -f logs/server.log logs/launchd-server.err
```

注: 既存の 8000/8010 などが使用中の場合に備え、LaunchAgentは 9050 を既定ポートにしています。クライアント側 `SERVER_URL` を `http://<MacのIP>:9050/inference` に設定してください。

## トラブルシューティング

### 音声が認識されない

- マイクの接続と音量を確認
- `arecord -l` (Linux) または システム環境設定 (Mac) で確認

### Wake Wordが反応しない

- はっきりと「もしもし」「サイテク」を発音
- 環境ノイズが多い場合は感度調整
- `LOG_LEVEL=DEBUG` で部分認識ログを確認（どの文字列に聞こえているかを把握）
- `WAKE_TIMEOUT_S` を延長（例: `export WAKE_TIMEOUT_S=4.0`）
- `WAKE_REQUIRE_BOTH=false` でいずれかの語で起動できるようにする（検証用途）
- 入力デバイスを指定: `export AUDIO_INPUT_DEVICE="<index or name>"`、一覧は `python -c 'import sounddevice as sd; print(sd.query_devices())'`

### サーバーに接続できない

- ファイアウォール設定を確認
- SERVER_URLが正しいか確認
- `ping` でネットワーク疎通を確認

## ログの見方

- 出力先: `logs/client.log`, `logs/server.log`（日次ローテーション、7日保持）
- 相関ID: 1回の発話処理ごとに `YYYYMMDDTHHMMSSZ-XXXXXXXX` 形式で採番し、クライアント⇄サーバー間で `X-Interaction-ID` ヘッダーで伝播します。
- 代表ログ例:
  - クライアント: `[ID] オンライン成功 roundtrip=1.23s transcript_len=10 reply_len=42`
  - サーバー: `[ID] 処理完了 STT=0.85s LLM=0.30s TOTAL=1.20s`
- 環境変数:
  - `LOG_TO_FILE=true|false`（既定: true）
  - `LOG_DIR=logs`（既定: logs）
  - `LOG_LEVEL=INFO|DEBUG|...`（既定: INFO）
- 実行例:
  - `tail -f logs/client.log`
  - `tail -f logs/server.log`
  - systemd利用時は `journalctl -u saiteku-client -f` / `journalctl -u saiteku-server -f`

## アーキテクチャ詳細

### コンポーネント

#### Wake Word検知
- Vosk: オフライン音声認識
- 軽量モデル（50MB）でRaspberry Piでも高速動作

#### VAD（Voice Activity Detection）
- WebRTC VAD: 音声区間検出
- 無音検出による自動録音停止

#### STT（Speech to Text）
- faster-whisper: 高精度日本語認識
- CTranslate2による高速化

#### LLM
- OpenAI互換API対応
- LM Studio / llama.cpp / Ollama等と連携可能

## パフォーマンス

### 処理時間目安

| 処理 | Raspberry Pi 5 | Mac mini M2 |
|------|---------------|-------------|
| Wake Word検知 | <100ms | <50ms |
| STT (3秒音声) | 2-3秒 | 0.5-1秒 |
| LLM応答 (1.5B) | 3-5秒 | 1-2秒 |

### メモリ使用量

- クライアント: 300-500MB
- サーバー: 1-4GB（モデルサイズによる）

## 今後の拡張

- [ ] TTS（Text-to-Speech）による音声応答
- [ ] カスタムWake Word学習
- [ ] マルチユーザー対応
- [ ] スマートホーム連携
- [ ] Web UIダッシュボード

## ライセンス

MIT License

## 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずissueを開いて変更内容を議論してください。

## サポート

問題が発生した場合は、GitHubのIssuesページで報告してください。

#!/usr/bin/env markdown
# sherpa-ONNX 設定ガイド

このプロジェクトでは、Voskに加えて sherpa-ONNX をローカルSTTやWake検出に利用できます。

## 1) 依存の導入

- 共通: `pip install -r requirements-client.txt`（`python-dotenv` を含む）
- sherpa-ONNX 本体: `pip install sherpa-onnx`（任意、利用時のみ）

## 2) .env を用意

```
cp config/.env.example .env
```

- Chat API, Wake（sherpa）, Local STT（sherpa）など、該当部分のコメントアウトを外し、モデルパスを実環境に合わせて編集してください。
- クライアントは `.env` を自動読み込みします（`python-dotenv`）。

## 3) 代表的な設定例

WakeをWhisper tinyで、Local STTを自動（設定があればsherpa、無ければVosk）にする例:

```
WAKE_BACKEND=sherpa
WAKE_SHERPA_MODEL_TYPE=whisper
WAKE_SHERPA_ENCODER=/opt/models/sherpa/whisper-tiny/encoder.onnx
WAKE_SHERPA_DECODER=/opt/models/sherpa/whisper-tiny/decoder.onnx
WAKE_SHERPA_TOKENS=/opt/models/sherpa/whisper-tiny/tokens.txt
WAKE_SHERPA_NUM_THREADS=2
WAKE_SHERPA_PROVIDER=cpu

LOCAL_STT_BACKEND=auto
SHERPA_MODEL_TYPE=whisper
SHERPA_ENCODER=/opt/models/sherpa/whisper-tiny/encoder.onnx
SHERPA_DECODER=/opt/models/sherpa/whisper-tiny/decoder.onnx
SHERPA_TOKENS=/opt/models/sherpa/whisper-tiny/tokens.txt
SHERPA_NUM_THREADS=2
SHERPA_PROVIDER=cpu
```

## 4) 起動と確認

```
python client/client.py
```

- 起動ログに以下が表示されます:
  - `Wakeバックエンド: SherpaWakeWordDetector` もしくは `VoskWakeWordDetector`
  - `ローカルSTTバックエンド: SherpaLocalSTT` もしくは `VoskLocalSTT`
- Chat APIを有効にしていれば、応答は「🌀 ストリーミング応答: 」として逐次出力されます。

## 備考

- モデルのAPIやファイル構成は sherpa-ONNX のバージョンや配布元によって差異があります。利用するモデルのREADMEを参照してください。
- sherpaの初期化に失敗した場合は自動でVoskへフォールバックします（WARNログが出ます）。
- Raspberry Pi 4/5 では sherpa-ONNX（Whisper tiny/Paraformer）が実用的な精度と速度を両立しやすいです。


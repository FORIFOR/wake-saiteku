#!/usr/bin/env python3
"""
Wake Saiteku Server - リモートPC側のSTTとLLM処理サーバー
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import soundfile as sf
import tempfile
import os
import requests
import logging
import logging.handlers
from typing import Dict, Any, Optional
import json
from datetime import datetime
import time

# ========== ロギング設定 ==========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ファイルロギング設定（日次ローテーション）
def _configure_file_logging():
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))
    if os.getenv("LOG_TO_FILE", "true").lower() == "true":
        log_dir = os.getenv("LOG_DIR", "logs")
        try:
            os.makedirs(log_dir, exist_ok=True)
            handler = logging.handlers.TimedRotatingFileHandler(
                os.path.join(log_dir, "server.log"), when="midnight", backupCount=7, encoding="utf-8"
            )
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
        except Exception as e:
            logger.warning(f"ログファイルの設定に失敗: {e}")

_configure_file_logging()

# ========== 環境変数から設定を読み込み ==========
FW_MODEL = os.getenv("FW_MODEL", "small")
FW_COMPUTE = os.getenv("FW_COMPUTE", "int8")
FW_DEVICE = os.getenv("FW_DEVICE", "cpu")

# LLM設定（LM StudioまたはOpenAI互換API）
LM_URL = os.getenv("LM_URL", "http://localhost:1234/v1/chat/completions")
LM_API_KEY = os.getenv("LM_API_KEY", "lm-studio")
LM_MODEL = os.getenv("LM_MODEL", "local-model")
LM_TIMEOUT = int(os.getenv("LM_TIMEOUT", "60"))

# ========== Whisperモデルの初期化 ==========
logger.info(f"Whisperモデルを初期化中: model={FW_MODEL}, device={FW_DEVICE}, compute_type={FW_COMPUTE}")
try:
    model = WhisperModel(
        FW_MODEL,
        device=FW_DEVICE,
        compute_type=FW_COMPUTE
    )
    logger.info("Whisperモデルの初期化完了")
except Exception as e:
    logger.error(f"Whisperモデルの初期化に失敗: {e}")
    raise

# ========== FastAPIアプリケーション ==========
app = FastAPI(title="Wake Saiteku Server", version="1.0.0")

# CORS設定（必要に応じて）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """ヘルスチェックエンドポイント"""
    return {
        "status": "online",
        "service": "Wake Saiteku Server",
        "whisper_model": FW_MODEL,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/inference")
async def inference(request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    音声ファイルを受け取り、STTとLLM処理を実行
    
    Returns:
        Dict containing transcript and LLM reply
    """
    start_time = datetime.utcnow()
    t0 = time.perf_counter()
    wav_path = None
    interaction_id = request.headers.get("x-interaction-id") or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    logp = f"[{interaction_id}]"
    
    try:
        # ファイル形式チェック
        if not file.filename.lower().endswith('.wav'):
            logger.warning(f"非WAVファイルが送信されました: {file.filename}")
        
        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            data = await file.read()
            tmp.write(data)
            wav_path = tmp.name
            logger.info(f"{logp} 音声ファイル保存: {wav_path} ({len(data)} bytes)")
        
        # Whisperで文字起こし
        logger.info(f"{logp} Whisper文字起こし開始 model={FW_MODEL}")
        stt_start = time.perf_counter()
        segments, info = model.transcribe(
            wav_path,
            language="ja",
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                threshold=0.5,
                min_speech_duration_ms=250,
                speech_pad_ms=400
            )
        )
        stt_dur = time.perf_counter() - stt_start
        
        # テキストを結合
        text = "".join(seg.text for seg in segments).strip()
        logger.info(f"{logp} 文字起こし完了 ({stt_dur:.2f}s): '{text}'")
        
        if not text:
            logger.warning(f"{logp} 音声から文字が検出されませんでした")
            return {
                "transcript": "",
                "reply": "音声が認識できませんでした。もう一度お話しください。",
                "processing_time": (datetime.utcnow() - start_time).total_seconds(),
                "interaction_id": interaction_id,
                "timings": {"stt": round(stt_dur, 2), "llm": 0.0, "total": round(time.perf_counter() - t0, 2)}
            }
        
        # LLMで応答生成
        llm_start = time.perf_counter()
        reply = await generate_llm_reply(text, interaction_id)
        llm_dur = time.perf_counter() - llm_start
        
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        total = time.perf_counter() - t0
        logger.info(f"{logp} 処理完了 STT={stt_dur:.2f}s LLM={llm_dur:.2f}s TOTAL={total:.2f}s")
        
        return {
            "transcript": text,
            "reply": reply,
            "processing_time": processing_time,
            "interaction_id": interaction_id,
            "timings": {"stt": round(stt_dur, 2), "llm": round(llm_dur, 2), "total": round(total, 2)}
        }
        
    except Exception as e:
        logger.error(f"{logp} 推論エラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 一時ファイルの削除
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
                logger.debug(f"一時ファイル削除: {wav_path}")
            except Exception as e:
                logger.warning(f"一時ファイル削除失敗: {e}")

async def generate_llm_reply(prompt: str, interaction_id: Optional[str] = None) -> str:
    """
    LLMを使用して応答を生成
    
    Args:
        prompt: ユーザーの入力テキスト
        
    Returns:
        LLMからの応答テキスト
    """
    payload = {
        "model": LM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "あなたは「サイテク」という名前の親切なアシスタントです。日本語で簡潔に、わかりやすく応答してください。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 500,
        "stream": False
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    if LM_API_KEY and LM_API_KEY != "lm-studio":
        headers["Authorization"] = f"Bearer {LM_API_KEY}"
    
    try:
        if interaction_id:
            headers["X-Interaction-ID"] = interaction_id
        logp = f"[{interaction_id}]" if interaction_id else ""
        logger.info(f"{logp} LLMリクエスト送信: {LM_URL}")
        response = requests.post(
            LM_URL,
            headers=headers,
            json=payload,
            timeout=LM_TIMEOUT
        )
        response.raise_for_status()
        
        result = response.json()
        reply = result["choices"][0]["message"]["content"]
        logger.info(f"{logp} LLM応答受信: '{reply[:100]}...'")
        return reply
        
    except requests.exceptions.Timeout:
        logger.error(f"{logp} LLMタイムアウト")
        return f"申し訳ありません、応答の生成に時間がかかっています。受信した内容: {prompt}"
        
    except requests.exceptions.ConnectionError:
        logger.error(f"{logp} LLM接続エラー")
        return f"(LLMサーバーに接続できません) 受信した内容: {prompt}"
        
    except Exception as e:
        logger.error(f"{logp} LLMエラー: {e}")
        return f"(エラーが発生しました) 受信した内容: {prompt}"

@app.post("/tts")
async def text_to_speech(text: str) -> Dict[str, str]:
    """
    テキストを音声に変換（将来の拡張用）
    """
    return {
        "status": "not_implemented",
        "message": "TTS機能は今後実装予定です"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("SERVER_PORT", "8000"))
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    
    logger.info(f"サーバー起動: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

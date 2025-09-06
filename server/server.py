#!/usr/bin/env python3
"""
Wake Saiteku Server - リモートPC側のSTTとLLM処理サーバー
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form, Body
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
# Note: Import faster-whisper lazily to avoid heavy deps in test environments
import tempfile
import os
import requests
import logging
import logging.handlers
from typing import Dict, Any, Optional, Iterable
import json
from datetime import datetime
import time
import io
import wave

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

# 追加: ローカルLLMとルーティング切替
LM_LOCAL_URL = os.getenv("LM_LOCAL_URL", "http://127.0.0.1:8010/v1/chat/completions")
LM_LOCAL_API_KEY = os.getenv("LM_LOCAL_API_KEY", "")
# primary | local | auto
LLM_ROUTING = os.getenv("LLM_ROUTING", "primary").strip().lower()

# ========== Whisperモデルの初期化 ==========
if os.getenv("SKIP_WHISPER_INIT", "false").lower() == "true":
    model = None  # type: ignore[assignment]
    logger.info("SKIP_WHISPER_INIT=true のためWhisper初期化をスキップします")
else:
    logger.info(f"Whisperモデルを初期化中: model={FW_MODEL}, device={FW_DEVICE}, compute_type={FW_COMPUTE}")
    try:
        from faster_whisper import WhisperModel  # type: ignore
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

def _choose_llm_endpoint(incoming_auth: Optional[str] = None, backend_override: Optional[str] = None):
    """Select LLM endpoint and headers based on override/header/env policy."""
    backend = (backend_override or "").strip().lower() or LLM_ROUTING
    if backend not in {"primary", "local", "auto"}:
        backend = "primary"

    def build_headers(use_primary: bool):
        headers = {"Content-Type": "application/json"}
        if incoming_auth:
            headers["Authorization"] = incoming_auth
        else:
            key = (LM_API_KEY if use_primary else LM_LOCAL_API_KEY)
            if key and key != "lm-studio":
                headers["Authorization"] = f"Bearer {key}"
        return headers

    if backend == "local":
        return LM_LOCAL_URL, build_headers(False), False
    if backend == "primary":
        return LM_URL, build_headers(True), False
    # auto: prefer primary, enable fallback
    return LM_URL, build_headers(True), True


async def generate_llm_reply(prompt: str, interaction_id: Optional[str] = None, request: Optional[Request] = None) -> str:
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
    
    incoming_auth = request.headers.get("authorization") if request else None
    backend_override = request.headers.get("x-llm-backend") if request else None
    url, headers, allow_fallback = _choose_llm_endpoint(incoming_auth, backend_override)
    if interaction_id:
        headers["X-Interaction-ID"] = interaction_id
    logp = f"[{interaction_id}]" if interaction_id else ""

    try:
        logger.info(f"{logp} LLMリクエスト送信: {url} backend={backend_override or LLM_ROUTING}")
        response = requests.post(url, headers=headers, json=payload, timeout=LM_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        reply = result["choices"][0]["message"]["content"]
        logger.info(f"{logp} LLM応答受信: '{reply[:100]}...'")
        return reply
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        if allow_fallback:
            # try local fallback
            try:
                url2, headers2, _ = LM_LOCAL_URL, {**headers}, False
                if LM_LOCAL_API_KEY:
                    headers2["Authorization"] = f"Bearer {LM_LOCAL_API_KEY}"
                logger.warning(f"{logp} LLM一次失敗({type(e).__name__}); ローカルにフォールバック: {url2}")
                r2 = requests.post(url2, headers=headers2, json=payload, timeout=LM_TIMEOUT)
                r2.raise_for_status()
                res2 = r2.json()
                return res2["choices"][0]["message"]["content"]
            except Exception as e2:
                logger.error(f"{logp} LLMフォールバックも失敗: {e2}")
        # failure messages
        if isinstance(e, requests.exceptions.Timeout):
            logger.error(f"{logp} LLMタイムアウト")
            return f"申し訳ありません、応答の生成に時間がかかっています。受信した内容: {prompt}"
        else:
            logger.error(f"{logp} LLM接続エラー")
            return f"(LLMサーバーに接続できません) 受信した内容: {prompt}"
    except Exception as e:
        logger.error(f"{logp} LLMエラー: {e}")
        return f"(エラーが発生しました) 受信した内容: {prompt}"

# ========== TTS（Text-to-Speech）サポート ==========
_TTS_ENABLED = os.getenv("TTS_ENABLED", "false").lower() == "true"
_TTS_BACKEND = os.getenv("TTS_BACKEND", "sherpa").strip().lower()

_tts_engine = None  # type: ignore[var-annotated]

def _init_tts_if_needed() -> None:
    global _tts_engine
    if _tts_engine is not None:
        return
    if not _TTS_ENABLED:
        logger.info("TTSは無効化されています (TTS_ENABLED=false)")
        return
    if _TTS_BACKEND != "sherpa":
        logger.warning(f"未対応のTTSバックエンド: {_TTS_BACKEND}. 現在は 'sherpa' のみ対応")
        return
    try:
        import sherpa_onnx as so  # type: ignore
    except Exception as e:
        logger.error(f"sherpa-onnx のインポートに失敗: {e}")
        return

    model_type = os.getenv("TTS_MODEL_TYPE", "vits").strip().lower()
    num_threads = int(os.getenv("TTS_NUM_THREADS", "1"))
    provider = os.getenv("TTS_PROVIDER", "cpu")
    max_sent = int(os.getenv("TTS_MAX_NUM_SENTENCES", "1"))
    silence_scale = float(os.getenv("TTS_SILENCE_SCALE", "0.2"))

    try:
        if model_type == "vits":
            # 必須パス
            vits_model = os.getenv("TTS_VITS_MODEL", "").strip()
            vits_tokens = os.getenv("TTS_VITS_TOKENS", "").strip()
            if not vits_model or not vits_tokens:
                raise RuntimeError("TTS_VITS_MODEL と TTS_VITS_TOKENS を設定してください")
            vits_lex = os.getenv("TTS_VITS_LEXICON", os.getenv("TTS_LEXICON", "").strip())
            # 生成速度/品質パラメータ（任意）
            ns = float(os.getenv("TTS_VITS_NOISE_SCALE", "0.667"))
            nsw = float(os.getenv("TTS_VITS_NOISE_SCALE_W", "0.8"))
            ls = float(os.getenv("TTS_VITS_LENGTH_SCALE", "1.0"))

            vits_cfg = so.OfflineTtsVitsModelConfig(
                model=vits_model,
                tokens=vits_tokens,
                lexicon=vits_lex,
                noise_scale=ns,
                noise_scale_w=nsw,
                length_scale=ls,
            )
            model_cfg = so.OfflineTtsModelConfig(
                vits=vits_cfg,
                num_threads=num_threads,
                provider=provider,
            )
        elif model_type == "kokoro":
            kokoro_model = os.getenv("TTS_KOKORO_MODEL", "").strip()
            kokoro_tokens = os.getenv("TTS_KOKORO_TOKENS", "").strip()
            if not kokoro_model or not kokoro_tokens:
                raise RuntimeError("TTS_KOKORO_MODEL と TTS_KOKORO_TOKENS を設定してください")
            kokoro_cfg = so.OfflineTtsKokoroModelConfig(
                model=kokoro_model,
                tokens=kokoro_tokens,
            )
            model_cfg = so.OfflineTtsModelConfig(
                kokoro=kokoro_cfg,
                num_threads=num_threads,
                provider=provider,
            )
        elif model_type == "matcha":
            matcha_model = os.getenv("TTS_MATCHA_MODEL", "").strip()
            matcha_tokens = os.getenv("TTS_MATCHA_TOKENS", "").strip()
            if not matcha_model or not matcha_tokens:
                raise RuntimeError("TTS_MATCHA_MODEL と TTS_MATCHA_TOKENS を設定してください")
            matcha_cfg = so.OfflineTtsMatchaModelConfig(
                model=matcha_model,
                tokens=matcha_tokens,
            )
            model_cfg = so.OfflineTtsModelConfig(
                matcha=matcha_cfg,
                num_threads=num_threads,
                provider=provider,
            )
        elif model_type == "kitten":
            kitten_model = os.getenv("TTS_KITTEN_MODEL", "").strip()
            kitten_tokens = os.getenv("TTS_KITTEN_TOKENS", "").strip()
            if not kitten_model or not kitten_tokens:
                raise RuntimeError("TTS_KITTEN_MODEL と TTS_KITTEN_TOKENS を設定してください")
            kitten_cfg = so.OfflineTtsKittenModelConfig(
                model=kitten_model,
                tokens=kitten_tokens,
            )
            model_cfg = so.OfflineTtsModelConfig(
                kitten=kitten_cfg,
                num_threads=num_threads,
                provider=provider,
            )
        else:
            raise RuntimeError(f"未知のTTSモデルタイプ: {model_type}")

        tts_cfg = so.OfflineTtsConfig(
            model=model_cfg,
            max_num_sentences=max_sent,
            silence_scale=silence_scale,
        )
        _tts_engine = so.OfflineTts(tts_cfg)
        logger.info(
            f"TTS初期化完了 backend=sherpa type={model_type} sr={_tts_engine.sample_rate}Hz speakers={_tts_engine.num_speakers}"
        )
    except Exception as e:
        logger.error(f"TTS初期化失敗: {e}")
        _tts_engine = None


def _wav_bytes_from_float32(audio, sr: int, channels: int = 1) -> bytes:
    import numpy as _np  # lazy import to keep server import lightweight
    audio = _np.asarray(audio)
    if audio.dtype != _np.float32:
        audio = audio.astype(_np.float32, copy=False)
    # [-1,1] -> int16
    pcm = _np.clip(audio, -1.0, 1.0)
    pcm_i16 = (pcm * 32767.0).astype(_np.int16, copy=False)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(pcm_i16.tobytes())
    return buf.getvalue()


@app.post("/tts")
async def text_to_speech(request: Request):
    """
    テキストを音声(WAV)に変換して返す。

    リクエスト(JSON): {"text": "こんにちは" [, "sid": 0, "speed": 1.0]}
    レスポンス: audio/wav バイナリ
    """
    _init_tts_if_needed()
    if _tts_engine is None:
        return JSONResponse(
            status_code=501,
            content={
                "status": "not_implemented",
                "message": "TTSが有効化されていないか初期化に失敗しました。TTS_ENABLED=true とモデルパスを設定してください。",
            },
        )

    try:
        payload = {}
        # form / query fallback
        try:
            payload = await request.json()
        except Exception:
            pass
        if not payload:
            # try query param ?text=...
            payload = {"text": request.query_params.get("text", "")}  # type: ignore[assignment]
        text = (payload.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="text が空です")

        sid = int(payload.get("sid", 0) or 0)
        speed = float(payload.get("speed", 1.0) or 1.0)

        # 生成
        res = _tts_engine.generate(text, sid=sid, speed=speed)
        # res は numpy.ndarray(float32) か、samples 属性を持つオブジェクト
        samples = getattr(res, "samples", res)
        sr = getattr(_tts_engine, "sample_rate")
        if isinstance(sr, property):
            # shouldn't happen for instances, but guard anyway
            sr_val = 22050
        else:
            sr_val = int(sr)  # type: ignore[arg-type]

        # Convert samples to WAV bytes; the helper will handle dtype conversion
        wav_bytes = _wav_bytes_from_float32(samples, sr_val)
        return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS生成エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== v1 API: Audio inference (non-streaming) ==========
def _detect_web_search(transcript: str, prompt: str = "") -> bool:
    keys = ["検索", "調べて", "ニュース", "天気", "価格", "レビュー", "最新"]
    t = (transcript or "") + " " + (prompt or "")
    return any(k in t for k in keys)


def _build_chat_messages(transcript: str, prompt: Optional[str]) -> Any:
    extra = (prompt or "").strip()
    if extra:
        content = f"以下の音声認識結果に基づき、指示に従って応答してください。\n指示: {extra}\n---\n音声認識結果: {transcript}"
    else:
        content = f"以下の音声認識結果に基づき、簡潔に応答してください。\n音声認識結果: {transcript}"
    return [
        {
            "role": "system",
            "content": "あなたは『サイテク』という親切な日本語アシスタントです。簡潔で具体的に答えてください。",
        },
        {"role": "user", "content": content},
    ]


@app.post("/v1/audio/inference")
async def v1_audio_inference(
    request: Request,
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(default=None),
) -> Dict[str, Any]:
    interaction_id = request.headers.get("x-interaction-id") or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    logp = f"[{interaction_id}]"
    start_time = datetime.utcnow()
    t0 = time.perf_counter()
    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            data = await file.read()
            tmp.write(data)
            wav_path = tmp.name
            logger.info(f"{logp} /v1/audio/inference 受信 bytes={len(data)} path={wav_path}")

        stt_start = time.perf_counter()
        segments, info = model.transcribe(
            wav_path,
            language="ja",
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                threshold=0.5,
                min_speech_duration_ms=250,
                speech_pad_ms=400,
            ),
        )
        stt_dur = time.perf_counter() - stt_start
        text = "".join(seg.text for seg in segments).strip()
        logger.info(f"{logp} STT完了 ({stt_dur:.2f}s): '{text}'")
        if not text:
            total_ms = int((time.perf_counter() - t0) * 1000)
            asr_ms = int(stt_dur * 1000)
            return {
                "object": "audio.inference",
                "transcript": "",
                "reply": "音声が認識できませんでした。",
                "timings": {"asr_ms": asr_ms, "llm_ms": 0, "total_ms": total_ms},
            }

        # Build chat payload for LLM
        payload = {
            "model": LM_MODEL,
            "messages": _build_chat_messages(text, prompt),
            "temperature": 0.7,
            "max_tokens": 500,
            "stream": False,
        }
        # Auto-enable web_search for certain keywords
        if _detect_web_search(text, prompt or ""):
            payload["web_search"] = True
            try:
                payload["search_k"] = int(os.getenv("WEB_SEARCH_K", "3"))
            except Exception:
                pass

        # Choose LLM backend
        incoming_auth = request.headers.get("authorization")
        backend_override = request.headers.get("x-llm-backend")
        url, headers, allow_fallback = _choose_llm_endpoint(incoming_auth, backend_override)
        if interaction_id:
            headers["X-Interaction-ID"] = interaction_id

        llm_start = time.perf_counter()
        r = requests.post(url, headers=headers, json=payload, timeout=LM_TIMEOUT)
        r.raise_for_status()
        result = r.json()
        reply = result["choices"][0]["message"]["content"]
        # Fallback on connection/timeout if policy is auto
        # (Non-streaming path only)
        llm_dur = time.perf_counter() - llm_start
        total = time.perf_counter() - t0
        logger.info(f"{logp} /v1/audio/inference 完了 STT={stt_dur:.2f}s LLM={llm_dur:.2f}s TOTAL={total:.2f}s")
        # API shape alignment
        return {
            "object": "audio.inference",
            "transcript": text,
            "reply": reply,
            "timings": {
                "asr_ms": int(stt_dur * 1000),
                "llm_ms": int(llm_dur * 1000),
                "total_ms": int(total * 1000),
            },
        }
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
        if allow_fallback:
            try:
                url2, headers2, _ = _choose_llm_endpoint(incoming_auth, "local")
                r2 = requests.post(url2, headers=headers2, json=payload, timeout=LM_TIMEOUT)
                r2.raise_for_status()
                res2 = r2.json()
                reply2 = res2["choices"][0]["message"]["content"]
                total = time.perf_counter() - t0
                return {
                    "object": "audio.inference",
                    "transcript": text,
                    "reply": reply2,
                    "timings": {
                        "asr_ms": int(stt_dur * 1000),
                        "llm_ms": int((time.perf_counter() - llm_start) * 1000),
                        "total_ms": int(total * 1000),
                    },
                }
            except Exception as e2:
                logger.error(f"/v1/audio/inference フォールバック失敗: {e2}")
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        logger.error(f"{logp} /v1/audio/inference エラー: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except Exception:
                pass


# ========== v1 API: Audio inference (streaming) ==========
def _sse_gen_from_openai_stream(resp) -> Iterable[bytes]:
    for raw in resp.iter_lines(decode_unicode=True):
        if raw is None:
            continue
        line = raw.strip()
        if not line:
            continue
        # proxy as-is
        if not line.startswith("data:"):
            yield ("data: " + line + "\n\n").encode("utf-8")
        else:
            yield (line + "\n\n").encode("utf-8")


@app.post("/v1/audio/inference/stream")
async def v1_audio_inference_stream(
    request: Request,
    file: UploadFile = File(...),
    prompt: Optional[str] = Form(default=None),
):
    interaction_id = request.headers.get("x-interaction-id") or datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    logp = f"[{interaction_id}]"
    wav_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            data = await file.read()
            tmp.write(data)
            wav_path = tmp.name
            logger.info(f"{logp} /v1/audio/inference/stream 受信 bytes={len(data)} path={wav_path}")

        # STT first
        segments, info = model.transcribe(
            wav_path,
            language="ja",
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                threshold=0.5,
                min_speech_duration_ms=250,
                speech_pad_ms=400,
            ),
        )
        text = "".join(seg.text for seg in segments).strip()
        logger.info(f"{logp} STT完了(stream): '{text}'")

        # Build and forward streaming request to LLM
        payload = {
            "model": LM_MODEL,
            "messages": _build_chat_messages(text, prompt),
            "temperature": 0.7,
            "max_tokens": 500,
            "stream": True,
        }
        if _detect_web_search(text, prompt or ""):
            payload["web_search"] = True
            try:
                payload["search_k"] = int(os.getenv("WEB_SEARCH_K", "3"))
            except Exception:
                pass
        incoming_auth = request.headers.get("authorization")
        backend_override = request.headers.get("x-llm-backend")
        url, headers, _ = _choose_llm_endpoint(incoming_auth, backend_override)
        if interaction_id:
            headers["X-Interaction-ID"] = interaction_id

        def stream():
            # send transcript first as an sse event
            init = json.dumps({"transcript": text}, ensure_ascii=False)
            yield ("data: " + init + "\n\n").encode("utf-8")
            try:
                with requests.post(url, headers=headers, json=payload, timeout=LM_TIMEOUT, stream=True) as resp:
                    resp.raise_for_status()
                    for chunk in _sse_gen_from_openai_stream(resp):
                        yield chunk
                # done
                yield b"data: [DONE]\n\n"
            except Exception as e:
                err = json.dumps({"error": str(e)}, ensure_ascii=False)
                yield ("data: " + err + "\n\n").encode("utf-8")

        return StreamingResponse(stream(), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"{logp} /v1/audio/inference/stream エラー: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
            except Exception:
                pass


# ========== v1 API: chat completions (proxy with optional streaming) ==========
@app.post("/v1/chat/completions")
async def v1_chat_completions(request: Request, body: Dict[str, Any] = Body(...)):
    # body may include: model, messages, stream, web_search, search_k, ...
    stream = bool(body.get("stream"))
    incoming_auth = request.headers.get("authorization")
    backend_override = request.headers.get("x-llm-backend")
    url, headers, _ = _choose_llm_endpoint(incoming_auth, backend_override)
    try:
        if stream:
            def proxy():
                try:
                    with requests.post(url, headers=headers, json=body, timeout=LM_TIMEOUT, stream=True) as r:
                        r.raise_for_status()
                        for chunk in _sse_gen_from_openai_stream(r):
                            yield chunk
                    yield b"data: [DONE]\n\n"
                except Exception as e:
                    err = json.dumps({"error": str(e)}, ensure_ascii=False)
                    yield ("data: " + err + "\n\n").encode("utf-8")
            return StreamingResponse(proxy(), media_type="text/event-stream")
        else:
            r = requests.post(url, headers=headers, json=body, timeout=LM_TIMEOUT)
            r.raise_for_status()
            return JSONResponse(status_code=r.status_code, content=r.json())
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="LLM timeout")
    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=502, detail="LLM connection error")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("SERVER_PORT", "8000"))
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    
    logger.info(f"サーバー起動: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

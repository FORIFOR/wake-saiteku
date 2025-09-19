# serve_kokoro.py
import os, io, pathlib, numpy as np, soundfile as sf, uvicorn
from fastapi import Body
from fastapi.responses import StreamingResponse, JSONResponse
from server import app
from kokoro_onnx import Kokoro

# モデルパス
M = os.path.expanduser(os.getenv("KOKORO_ONNX_MODEL",  "~/models/kokoro/kokoro-v1.0.onnx"))
V = os.path.expanduser(os.getenv("KOKORO_ONNX_VOICES", "~/models/kokoro/voices-v1.0.bin"))
CACHE = os.path.expanduser(os.getenv("KOKORO_CACHE_DIR", "~/Projects/wake-saiteku/u-tts/cache"))
pathlib.Path(CACHE).mkdir(parents=True, exist_ok=True)

if not (os.path.exists(M) and os.path.exists(V)):
    raise FileNotFoundError(f"model or voices not found: {M}, {V}")

kokoro = Kokoro(model_path=M, voices_path=V)  # ここがコア

@app.get("/voices")
def voices(engine: str = "kokoro_onnx"):
    if engine != "kokoro_onnx":
        return JSONResponse(status_code=400, content={"detail": f"engine not available: {engine}"})
    out = []
    for vid in kokoro.get_voices():
        out.append({"id": vid, "path": V, "lang": "ja"})
    return out

@app.post("/tts")
def tts(payload: dict = Body(...)):
    # 入力
    text   = payload.get("text")
    voice  = payload.get("voice") or payload.get("speaker") or "jf_alpha"
    lang   = payload.get("lang", "ja")
    speed  = payload.get("speed", 1.0)

    if not text:
        return JSONResponse(status_code=400, content={"detail": "text is required"})

    # 旧 misaki 系の前処理は使わず、そのまま Kokoro に渡す
    try:
        # バージョン差異に備えて create のみ使用
        wav, sr = kokoro.create(text, voice=voice, lang=lang, speed=speed)
    except TypeError:
        # lang/speed を受けない旧シグネチャ向けフォールバック
        wav, sr = kokoro.create(text, voice=voice)

    # WAV 化して返す
    buf = io.BytesIO()
    sf.write(buf, np.asarray(wav, dtype=np.float32), int(sr), format="WAV")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=9051, log_level="info")

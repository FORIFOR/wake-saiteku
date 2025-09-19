# mini_kokoro_server.py
from fastapi import FastAPI, Body, Response
from pydantic import BaseModel
from kokoro import KPipeline
import numpy as np
import soundfile as sf
import io

app = FastAPI(title="Mini Kokoro TTS")

# 日本語パイプラインを1度だけロード
pipe = KPipeline(lang_code='j')

# よく使う日本語ボイス（存在が確認できるもの）
J_VOICES = ["jf_alpha", "jm_kumo", "jf_gongitsune", "jf_nezumi", "jf_tebukuro"]

class SynthesisIn(BaseModel):
    text: str
    voice: str | None = None   # 省略時は jf_alpha
    speed: float | None = 1.0

@app.get("/voices")
def voices():
    # Kokoroはプログラムから声一覧を取る公APIが無いので、既知IDを返す
    return [{"id": v, "lang": "ja"} for v in J_VOICES]

@app.post("/tts")
def tts(inp: SynthesisIn):
    voice = inp.voice or "jf_alpha"
    speed = inp.speed or 1.0
    # 合成
    chunks = []
    for _, _, audio in pipe(inp.text, voice=voice, speed=speed):
        chunks.append(audio)
    wav = np.concatenate(chunks)
    # WAVにして返す
    buf = io.BytesIO()
    sf.write(buf, wav, 24000, format="WAV")
    return Response(content=buf.getvalue(), media_type="audio/wav")

#!/usr/bin/env python3
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field
import importlib.util

# Dynamically load engine modules to support hyphenated directory name
ENG_DIR = Path(__file__).resolve().parent / "engines"

def _load_class(path: Path, qualname: str):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if not spec or not spec.loader:
        raise ImportError(f"Cannot load module at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, qualname)

PiperEngine = _load_class(ENG_DIR / "piper_engine.py", "PiperEngine")
OpenJTalkEngine = _load_class(ENG_DIR / "openjtalk_engine.py", "OpenJTalkEngine")
try:
    KokoroOnnxEngine = _load_class(ENG_DIR / "kokoro_onnx_engine.py", "KokoroOnnxEngine")
except Exception:
    KokoroOnnxEngine = None  # type: ignore
try:
    KokoroPT = _load_class(ENG_DIR / "kokoro_pt_engine.py", "KokoroPT")
except Exception:
    KokoroPT = None  # type: ignore

# Optional placeholders (future)
XTTSEngine = None  # type: ignore


class TTSReq(BaseModel):
    text: str = Field(..., min_length=1)
    engine: Optional[str] = None  # piper|openjtalk|kokoro|xtts
    lang: Optional[str] = None
    voice: Optional[str] = None
    speed: float = 1.0
    pitch: float = 0.0
    gain_db: float = 0.0
    sample_rate: int = 22050
    cache: bool = True
    reference_wav: Optional[str] = None


def _env_path(name: str, default: Optional[Path] = None) -> Path:
    v = os.environ.get(name)
    return Path(v) if v else (default if default is not None else Path.cwd())


# Directories
UTTSDATA = _env_path("UTTSDATA", Path(__file__).resolve().parent)
CACHE_DIR = UTTSDATA / "cache"
MODELS_DIR = UTTSDATA / "models"
PIPER_MODELS = MODELS_DIR / "piper"
for d in (CACHE_DIR, PIPER_MODELS):
    d.mkdir(parents=True, exist_ok=True)


# Build engines map
engines: Dict[str, Any] = {}

# Piper (if voices found)
try:
    piper = PiperEngine(models_dir=str(PIPER_MODELS), cache_dir=str(CACHE_DIR))
    if len(piper.list_voices()) > 0:
        engines["piper"] = piper
except Exception:
    pass

# OpenJTalk (requires dict + voice)
OJ_DICT = os.environ.get("OPENJTALK_DICT_DIR")
OJ_VOICE = os.environ.get("OPENJTALK_VOICE")
if OJ_DICT and OJ_VOICE:
    try:
        oj = OpenJTalkEngine(dict_dir=OJ_DICT, voice_path=OJ_VOICE, cache_dir=str(CACHE_DIR))
        engines["openjtalk"] = oj
    except Exception:
        pass

# Optional engines (plug-in if available)
if KokoroOnnxEngine is not None:
    KOKORO_ONNX_MODEL = os.environ.get("KOKORO_ONNX_MODEL")
    KOKORO_ONNX_VOICES = os.environ.get("KOKORO_ONNX_VOICES")
    if KOKORO_ONNX_MODEL and KOKORO_ONNX_VOICES:
        try:
            engines["kokoro_onnx"] = KokoroOnnxEngine(
                model_path=KOKORO_ONNX_MODEL,
                voices_path=KOKORO_ONNX_VOICES,
                cache_dir=str(CACHE_DIR),
            )
        except Exception:
            pass

if KokoroPT is not None and os.environ.get("ENABLE_KOKORO_PT", "false").lower() == "true":
    try:
        engines["kokoro_pt"] = KokoroPT(cache_dir=str(CACHE_DIR))
    except Exception:
        pass
if XTTSEngine is not None:
    # engines["xtts"] = XTTSEngine(...)
    pass


app = FastAPI(title="Unified TTS Service", version="0.1.0")


@app.get("/health", response_class=PlainTextResponse)
def health() -> str:
    return json.dumps({"status": "ok", "engines": list(engines.keys())}, ensure_ascii=False)


@app.get("/voices")
def voices(engine: str = Query(..., description="piper|openjtalk|kokoro_onnx|kokoro_pt|xtts")) -> List[Dict[str, Any]]:
    if engine not in engines:
        raise HTTPException(status_code=400, detail=f"engine not available: {engine}")
    try:
        return engines[engine].list_voices()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts")
def tts(req: TTSReq):
    # resolve default engine: piper -> openjtalk -> kokoro (if available)
    engine_name = req.engine
    if not engine_name:
        for cand in ("piper", "openjtalk", "kokoro_onnx", "kokoro_pt"):
            if cand in engines:
                engine_name = cand
                break
    if not engine_name or engine_name not in engines:
        raise HTTPException(status_code=400, detail="no available engine")

    # attempt synthesis; optional fallback piper->openjtalk if engine unspecified
    candidates = [engine_name]
    if req.engine is None:
        # fallback order
        order = ["piper", "openjtalk", "kokoro_onnx", "kokoro_pt"]
        for cand in order:
            if cand in engines and cand not in candidates:
                candidates.append(cand)

    last_err: Optional[str] = None
    for ename in candidates:
        eng = engines[ename]
        try:
            if ename == "piper":
                voice_id = req.voice or (eng.list_voices()[0]["id"])
                wavpath = eng.synth(
                    text=req.text,
                    voice_id=voice_id,
                    sample_rate=int(req.sample_rate),
                    speed=float(req.speed),
                    gain_db=float(req.gain_db),
                    pitch=float(req.pitch),
                    cache=bool(req.cache),
                )
            elif ename == "openjtalk":
                wavpath = eng.synth(
                    text=req.text,
                    sample_rate=int(req.sample_rate),
                    speed=float(req.speed),
                    pitch=float(req.pitch),
                    gain_db=float(req.gain_db),
                    cache=bool(req.cache),
                )
            elif ename == "kokoro_onnx":
                voice_id = req.voice or (eng.list_voices()[0]["id"])
                wavpath = eng.synth(
                    text=req.text,
                    voice=voice_id,
                    sample_rate=int(req.sample_rate),
                    speed=float(req.speed),
                    gain_db=float(req.gain_db),
                    pitch=float(req.pitch),
                    cache=bool(req.cache),
                )
            elif ename == "kokoro_pt":
                voice_id = req.voice or (eng.list_voices()[0]["id"])
                wavpath = eng.synth(
                    text=req.text,
                    voice=voice_id,
                    sample_rate=int(req.sample_rate),
                    speed=float(req.speed),
                    gain_db=float(req.gain_db),
                    pitch=float(req.pitch),
                    cache=bool(req.cache),
                )
            else:
                raise HTTPException(status_code=400, detail=f"unsupported engine: {ename}")
            # Add helpful headers
            if ename in ("piper", "kokoro_onnx", "kokoro_pt"):
                voice_hdr = voice_id
            else:
                voice_hdr = Path(getattr(eng, "voice_path", "")).stem or "default"
            headers = {
                "X-Engine": ename,
                "X-Voice": voice_hdr,
                "X-Sample-Rate": str(req.sample_rate),
            }
            return FileResponse(wavpath, media_type="audio/wav", filename="out.wav", headers=headers)
        except HTTPException:
            raise
        except Exception as e:
            last_err = str(e)
            continue

    raise HTTPException(status_code=500, detail=f"synthesis failed: {last_err or 'unknown error'}")


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("UTTS_HOST", "0.0.0.0")
    port = int(os.environ.get("UTTS_PORT", "9051"))
    uvicorn.run(app, host=host, port=port)

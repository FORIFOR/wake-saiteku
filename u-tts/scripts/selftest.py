#!/usr/bin/env python3
import os
import sys
import json
import shutil
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
ENG_DIR = ROOT / "engines"


def _load_class(path: Path, qualname: str):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if not spec or not spec.loader:
        raise ImportError(f"Cannot load module at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, qualname)


def main() -> int:
    print("[UTTS] Self-test start")
    uttsdata = Path(os.environ.get("UTTSDATA", ROOT))
    cache = uttsdata / "cache"
    models = uttsdata / "models"
    piper_models = models / "piper"
    cache.mkdir(parents=True, exist_ok=True)
    piper_models.mkdir(parents=True, exist_ok=True)

    PiperEngine = _load_class(ENG_DIR / "piper_engine.py", "PiperEngine")
    OpenJTalkEngine = _load_class(ENG_DIR / "openjtalk_engine.py", "OpenJTalkEngine")

    ok_count = 0

    # Piper check
    try:
        if shutil.which("piper") is None:
            print("[Piper] piper CLI not found. Install with: pip install piper-tts")
        pe = PiperEngine(models_dir=str(piper_models), cache_dir=str(cache))
        voices = pe.list_voices()
        if voices:
            vid = voices[0]["id"]
            print(f"[Piper] Found voices: {[v['id'] for v in voices]} (using '{vid}')")
            wav = pe.synth("テスト合成。これはPiperのテストです。", voice_id=vid, sample_rate=22050, speed=1.0)
            print(f"[Piper] OK -> {wav}")
            ok_count += 1
        else:
            print(f"[Piper] No models in {piper_models}. Place *.onnx + *.onnx.json")
    except Exception as e:
        print(f"[Piper] FAIL: {e}")

    # OpenJTalk check
    try:
        dict_dir = os.environ.get("OPENJTALK_DICT_DIR")
        voice = os.environ.get("OPENJTALK_VOICE")
        if not dict_dir or not voice:
            print("[OpenJTalk] Set OPENJTALK_DICT_DIR and OPENJTALK_VOICE to test.")
        else:
            if shutil.which("open_jtalk") is None:
                print("[OpenJTalk] open_jtalk not found. Install via apt/brew.")
            oe = OpenJTalkEngine(dict_dir=dict_dir, voice_path=voice, cache_dir=str(cache))
            wav = oe.synth("テスト合成。こちらはOpen JTalkのテストです。", sample_rate=22050, speed=1.1)
            print(f"[OpenJTalk] OK -> {wav}")
            ok_count += 1
    except Exception as e:
        print(f"[OpenJTalk] FAIL: {e}")

    if ok_count == 0:
        print("[UTTS] No TTS engine produced audio. Please install piper/open_jtalk and set env/models.")
        return 1
    print(f"[UTTS] Self-test done (engines OK: {ok_count})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


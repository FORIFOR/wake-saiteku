import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple


class KokoroOnnxEngine:
    """
    Kokoro-ONNX engine with misaki[ja] G2P.

    - Requires: pip install kokoro-onnx "misaki[ja]" soundfile
    - Models: model_path (kokoro-v*.onnx), voices_path (voices-*.bin)
    - Caching: cache_dir/<sha1>.wav
    - Text handling: chunk by Japanese punctuation (。！？) ~60-80 chars
    - G2P: misaki ja → is_phonemes=True to avoid non-Japanese G2P
    - Post FX: resample/pitch/gain via sox if available
    """

    def __init__(self, model_path: str, voices_path: str, cache_dir: str):
        try:
            from kokoro_onnx import Kokoro  # type: ignore
            from misaki import ja  # type: ignore
        except Exception as e:
            raise RuntimeError(f"kokoro-onnx / misaki が見つかりません: {e}")
        self._Kokoro = Kokoro
        self._ja = ja
        self.model = Path(model_path)
        self.voices = Path(voices_path)
        if not self.model.exists() or not self.voices.exists():
            raise RuntimeError(f"Kokoro-ONNX モデル/ボイスが見つかりません: {self.model} / {self.voices}")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # lazy init heavy objects
        self._g2p = None
        self._tts = None

    def _load(self):
        if self._g2p is None:
            self._g2p = self._ja.G2P()
        if self._tts is None:
            self._tts = self._Kokoro(str(self.model), str(self.voices))

    def list_voices(self) -> List[Dict[str, Any]]:
        # Voices bin is opaque; expose known defaults commonly distributed
        return [
            {"id": v, "path": str(self.voices), "lang": "ja"}
            for v in [
                "jf_alpha",
                "jm_kumo",
            ]
        ]

    def _hash_key(self, text: str, voice: str, sample_rate: int, speed: float, gain_db: float, pitch: float) -> str:
        key = f"kokoro-onnx|{voice}|{sample_rate}|{speed:.3f}|{gain_db:.2f}|{pitch:.2f}|{text}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()

    @staticmethod
    def _chunk_text(text: str, max_len: int = 80) -> List[str]:
        import re
        # split by 。！？!? retaining punctuation
        parts = re.findall(r"[^。！？!?]+[。！？!?]?", text)
        out: List[str] = []  # type: ignore[name-defined]
        buf = ""
        for p in parts:
            if len(buf) + len(p) <= max_len:
                buf += p
            else:
                if buf:
                    out.append(buf)
                buf = p
        if buf:
            out.append(buf)
        return [s.strip() for s in out if s.strip()]

    @staticmethod
    def _write_wav(path: Path, audio, sr: int) -> None:
        try:
            import soundfile as sf  # type: ignore
            sf.write(str(path), audio, sr)
        except Exception:
            # Fallback minimal WAV write
            import wave
            import numpy as np
            a = (np.asarray(audio).astype(np.float32))
            pcm = (a.clip(-1.0, 1.0) * 32767.0).astype("<i2")
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(int(sr))
                wf.writeframes(pcm.tobytes())

    def synth(
        self,
        text: str,
        voice: str = "jf_alpha",
        sample_rate: int = 24000,
        speed: float = 1.0,
        gain_db: float = 0.0,
        pitch: float = 0.0,
        cache: bool = True,
    ) -> str:
        self._load()
        key = self._hash_key(text, voice, sample_rate, speed, gain_db, pitch)
        outpath = self.cache_dir / f"{key}.wav"
        if cache and outpath.exists():
            return str(outpath)

        # Synthesize by chunks
        chunks = self._chunk_text(text)
        import numpy as np
        wavs: List[Any] = []
        for ch in chunks:
            phonemes, _ = self._g2p(ch)
            audio, sr = self._tts.create(phonemes, voice=voice, is_phonemes=True, speed=float(speed))
            wavs.append((audio, sr))

        # Concatenate with small pause
        if not wavs:
            raise RuntimeError("no audio generated")
        base_sr = wavs[0][1]
        audio_cat = np.concatenate([a for a, _ in wavs])

        # Write to temp, then post-process with sox if available
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = Path(tmp.name)
        try:
            self._write_wav(tmp_wav, audio_cat, base_sr)
            sox = shutil.which("sox")
            if sox and (int(sample_rate) != int(base_sr) or abs(gain_db) > 1e-6 or abs(pitch) > 1e-6):
                cmd = [sox, str(tmp_wav)]
                if sample_rate and int(sample_rate) != int(base_sr):
                    cmd.extend(["-r", str(int(sample_rate))])
                cmd.append(str(outpath))
                if abs(pitch) > 1e-6:
                    cents = float(pitch) * 100.0
                    cmd.extend(["pitch", f"{cents}"])
                if abs(gain_db) > 1e-6:
                    cmd.extend(["gain", f"{gain_db}"])
                res = os.spawnvp(os.P_WAIT, cmd[0], cmd)
                if res != 0 or not outpath.exists():
                    shutil.move(str(tmp_wav), str(outpath))
            else:
                shutil.move(str(tmp_wav), str(outpath))
            return str(outpath)
        finally:
            try:
                if tmp_wav.exists():
                    os.unlink(tmp_wav)
            except Exception:
                pass


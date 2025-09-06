import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List


class KokoroPT:
    """
    PyTorch Kokoro pipeline (quality check, quick setup).

    - Requires: python 3.12 venv, pip install kokoro==0.9.4 misaki[ja] soundfile fugashi unidic-lite
    - Uses KPipeline(lang_code='j') and voice like 'jf_alpha'
    - Caching: cache_dir/<sha1>.wav
    - Post FX: resample/pitch/gain via sox if available
    """

    def __init__(self, cache_dir: str):
        try:
            from kokoro import KPipeline  # type: ignore
        except Exception as e:
            raise RuntimeError(f"kokoro (PyTorch) が見つかりません: {e}")
        self._KPipeline = KPipeline
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            # Japanese pipeline
            self._pipe = self._KPipeline(lang_code='j')

    def list_voices(self) -> List[Dict[str, Any]]:
        # Kokoro voices depend on the package assets; expose common ones
        return [
            {"id": v, "lang": "ja"}
            for v in ["jf_alpha", "jm_kumo"]
        ]

    def _hash_key(self, text: str, voice: str, sample_rate: int, speed: float, gain_db: float, pitch: float) -> str:
        import hashlib
        key = f"kokoro-pt|{voice}|{sample_rate}|{speed:.3f}|{gain_db:.2f}|{pitch:.2f}|{text}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()

    def _write_wav(self, path: Path, audio, sr: int) -> None:
        try:
            import soundfile as sf  # type: ignore
            sf.write(str(path), audio, sr)
        except Exception:
            import wave, numpy as np
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

        # KPipeline returns generator; take first segment for simplicity
        _, _, audio = next(self._pipe(text, voice=voice))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = Path(tmp.name)
        try:
            # Kokoro defaults to 24kHz
            base_sr = 24000
            self._write_wav(tmp_wav, audio, base_sr)
            sox = shutil.which("sox")
            if sox and (int(sample_rate) != int(base_sr) or abs(gain_db) > 1e-6 or abs(pitch) > 1e-6):
                cmd = [sox, str(tmp_wav)]
                if int(sample_rate) != int(base_sr):
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


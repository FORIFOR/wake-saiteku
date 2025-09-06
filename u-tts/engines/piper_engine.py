import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
from typing import List, Dict, Any


@dataclass
class PiperVoice:
    id: str
    path: str
    config: Optional[str]
    meta: Dict[str, Any]


class PiperEngine:
    """
    Piper wrapper using the `piper` CLI (pip install piper-tts).

    - Models: models_dir/*.onnx with sidecar *.onnx.json
    - Caching: cache_dir/<sha1>.wav
    - Speed: mapped via --length_scale = 1.0 / speed
    - Sample rate: resampled with `sox` if available
    """

    def __init__(self, models_dir: str, cache_dir: str, bin_name: str = "piper"):
        self.models_dir = Path(models_dir)
        self.cache_dir = Path(cache_dir)
        self.bin = bin_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def list_voices(self) -> List[Dict[str, Any]]:
        voices: List[Dict[str, Any]] = []
        if not self.models_dir.exists():
            return voices
        for onnx in sorted(self.models_dir.glob("*.onnx")):
            cfg = onnx.with_suffix(".onnx.json")
            meta = {}
            if cfg.exists():
                try:
                    meta = json.loads(cfg.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            voices.append(
                {
                    "id": onnx.stem,
                    "path": str(onnx),
                    "config": str(cfg) if cfg.exists() else None,
                    "meta": meta,
                }
            )
        return voices

    def _hash_key(
        self,
        text: str,
        voice_id: str,
        sample_rate: int,
        speed: float,
        gain_db: float,
        pitch: float,
    ) -> str:
        key = f"piper|{voice_id}|{sample_rate}|{speed:.3f}|{gain_db:.2f}|{pitch:.2f}|{text}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()

    def _write_wav(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def synth(
        self,
        text: str,
        voice_id: str,
        sample_rate: int = 22050,
        speed: float = 1.0,
        gain_db: float = 0.0,
        pitch: float = 0.0,
        cache: bool = True,
    ) -> str:
        onnx = self.models_dir / f"{voice_id}.onnx"
        cfg = self.models_dir / f"{voice_id}.onnx.json"
        if not onnx.exists() or not cfg.exists():
            raise RuntimeError(f"Piper model not found: {voice_id}")

        key = self._hash_key(text, voice_id, sample_rate, speed, gain_db, pitch)
        outpath = self.cache_dir / f"{key}.wav"
        if cache and outpath.exists():
            return str(outpath)

        # Generate to a temp wav first (model native sample rate), then resample if needed
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = Path(tmp.name)

        try:
            # Piper CLI
            # length_scale is inverse of speed
            length_scale = 1.0 / max(0.1, float(speed))
            cmd = [
                self.bin,
                "--model",
                str(onnx),
                "--config",
                str(cfg),
                "--output_file",
                str(tmp_wav),
                "--length_scale",
                f"{length_scale:.3f}",
                "--sentence_silence",
                "0.2",
            ]
            res = subprocess.run(cmd, input=text.encode("utf-8"))
            if res.returncode != 0 or not tmp_wav.exists() or tmp_wav.stat().st_size == 0:
                raise RuntimeError("Piper synthesis failed")

            # Post-process: resample/pitch/gain with sox if available
            sox = shutil.which("sox")
            if sox:
                cmd2 = [sox, str(tmp_wav)]
                if sample_rate:
                    cmd2.extend(["-r", str(sample_rate)])
                cmd2.append(str(outpath))
                # pitch in semitones -> cents
                if abs(pitch) > 1e-6:
                    cents = float(pitch) * 100.0
                    cmd2.extend(["pitch", f"{cents}"])
                if abs(gain_db) > 1e-6:
                    cmd2.extend(["gain", f"{gain_db}"])
                res2 = subprocess.run(cmd2)
                if res2.returncode != 0 or not outpath.exists():
                    # Fallback to original
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

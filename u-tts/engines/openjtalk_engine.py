import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any


@dataclass
class OpenJTalkConfig:
    dict_dir: str
    voice_path: str


class OpenJTalkEngine:
    """
    Open JTalk wrapper using the `open_jtalk` CLI.

    - Requires system packages (dict + voice), paths via env or passed in
    - Caching: cache_dir/<sha1>.wav
    - Speed: `-r` option
    - Sample rate: `-s` option
    - Gain: applied with `sox gain` if available
    """

    def __init__(self, dict_dir: str, voice_path: str, cache_dir: str, bin_name: str = "open_jtalk"):
        self.dict_dir = dict_dir
        self.voice_path = voice_path
        self.cache_dir = Path(cache_dir)
        self.bin = bin_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def list_voices(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": Path(self.voice_path).stem,
                "path": self.voice_path,
                "lang": "ja",
            }
        ]

    def _hash_key(self, text: str, sample_rate: int, speed: float, pitch: float, gain_db: float) -> str:
        key = f"openjtalk|{self.voice_path}|{sample_rate}|{speed:.3f}|{pitch:.2f}|{gain_db:.2f}|{text}"
        return hashlib.sha1(key.encode("utf-8")).hexdigest()

    def synth(
        self,
        text: str,
        sample_rate: int = 22050,
        speed: float = 1.0,
        pitch: float = 0.0,
        gain_db: float = 0.0,
        cache: bool = True,
    ) -> str:
        key = self._hash_key(text, sample_rate, speed, pitch, gain_db)
        outpath = self.cache_dir / f"{key}.wav"
        if cache and outpath.exists():
            return str(outpath)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = Path(tmp.name)

        try:
            cmd = [
                self.bin,
                "-x",
                self.dict_dir,
                "-m",
                self.voice_path,
                "-r",
                f"{float(speed):.2f}",
                "-s",
                str(int(sample_rate)),
                "-ow",
                str(tmp_wav),
            ]
            # We keep pitch handling minimal; advanced pitch shaping requires additional DSP
            res = subprocess.run(cmd, input=text.encode("utf-8"))
            if res.returncode != 0 or not tmp_wav.exists() or tmp_wav.stat().st_size == 0:
                raise RuntimeError("OpenJTalk synthesis failed")

            # Apply pitch/gain with sox if available
            sox = shutil.which("sox")
            if sox and (abs(gain_db) > 1e-6 or abs(pitch) > 1e-6):
                cmd2 = [sox, str(tmp_wav), str(outpath)]
                if abs(pitch) > 1e-6:
                    cents = float(pitch) * 100.0
                    cmd2.extend(["pitch", f"{cents}"])
                if abs(gain_db) > 1e-6:
                    cmd2.extend(["gain", f"{gain_db}"])
                res2 = subprocess.run(cmd2)
                if res2.returncode != 0 or not outpath.exists():
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

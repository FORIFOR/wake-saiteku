"""
Local STT backends abstraction.

Supports:
- Vosk (existing)
- sherpa-onnx (optional; if installed and configured via env)

This module avoids importing heavy deps at module import time.
"""

from __future__ import annotations

import os
import wave
import tempfile
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SherpaConfig:
    model_type: str  # 'paraformer' | 'whisper' | 'transducer'
    tokens: Optional[str] = None
    model: Optional[str] = None  # paraformer
    encoder: Optional[str] = None  # whisper/transducer
    decoder: Optional[str] = None  # whisper/transducer
    joiner: Optional[str] = None  # transducer
    num_threads: int = 1
    provider: str = "cpu"  # 'cpu' or 'coreml' or 'cuda' (if available)
    # Whisper-specific
    language: str = "auto"
    task: str = "transcribe"  # or 'translate'


class BaseLocalSTT:
    def transcribe(self, pcm: np.ndarray, sample_rate: int) -> str:
        raise NotImplementedError


class SherpaLocalSTT(BaseLocalSTT):
    def __init__(self, cfg: SherpaConfig):
        # Lazy import
        try:
            import sherpa_onnx as so  # type: ignore
        except Exception as e:
            raise RuntimeError(f"sherpa-onnx をインポートできません: {e}")

        self.so = so
        self.cfg = cfg

        mt = (cfg.model_type or "").lower()
        if mt not in {"paraformer", "whisper", "transducer"}:
            raise ValueError(f"未知の model_type: {cfg.model_type}")

        # Prefer Python wrapper classmethods first (sherpa-onnx >=1.10)
        OR = getattr(self.so, "OfflineRecognizer", None)
        if OR is None:
            raise RuntimeError("sherpa_onnx.OfflineRecognizer が見つかりません")

        try:
            if mt == "whisper" and hasattr(OR, "from_whisper"):
                if not (cfg.encoder and cfg.decoder):
                    raise ValueError("Whisper には ENCODER/DECODER が必要です")
                lang = (cfg.language or "auto")
                if lang.lower() in {"", "auto", "autodetect", "auto_detect"}:
                    # Avoid invalid 'auto' for sherpa whisper. Default to Japanese.
                    lang = "ja"
                    logging.getLogger(__name__).warning(
                        "Whisper言語が'auto'のため'ja'に設定しました。SHERPA_LANGUAGEで変更できます。"
                    )
                self.recognizer = OR.from_whisper(
                    encoder=cfg.encoder,
                    decoder=cfg.decoder,
                    tokens=cfg.tokens or "",
                    language=lang,
                    task=cfg.task or "transcribe",
                    num_threads=cfg.num_threads,
                    provider=cfg.provider,
                )
                return
            if mt == "paraformer" and hasattr(OR, "from_paraformer"):
                if not (cfg.model and cfg.tokens):
                    raise ValueError("Paraformer には SHERPA_MODEL と SHERPA_TOKENS が必要です")
                self.recognizer = OR.from_paraformer(
                    paraformer=cfg.model,
                    tokens=cfg.tokens or "",
                    num_threads=cfg.num_threads,
                    provider=cfg.provider,
                )
                return
            if mt == "transducer" and hasattr(OR, "from_transducer"):
                if not (cfg.encoder and cfg.decoder and cfg.joiner and cfg.tokens):
                    raise ValueError("Transducer には ENCODER/DECODER/JOINER/TOKENS が必要です")
                self.recognizer = OR.from_transducer(
                    encoder=cfg.encoder,
                    decoder=cfg.decoder,
                    joiner=cfg.joiner,
                    tokens=cfg.tokens or "",
                    num_threads=cfg.num_threads,
                    provider=cfg.provider,
                )
                return
        except Exception:
            # Fall through to config-based initialization
            pass

        # Build model config dynamically, tolerating minor API differences.
        OfflineModelConfig = getattr(so, "OfflineModelConfig", None)
        if OfflineModelConfig is None:
            raise RuntimeError("sherpa_onnx.OfflineModelConfig が見つかりません")

        # Sub-configs
        ParaConfig = getattr(so, "OfflineParaformerModelConfig", None)
        TransConfig = getattr(so, "OfflineTransducerModelConfig", None)
        WhisperConfig = getattr(so, "OfflineWhisperModelConfig", None)

        # Prepare kwargs for OfflineModelConfig
        model_kwargs = {
            "tokens": cfg.tokens or "",
            "num_threads": cfg.num_threads,
            "provider": cfg.provider,
        }

        if mt == "paraformer":
            if not (cfg.model and cfg.tokens):
                raise ValueError("Paraformer には SHERPA_MODEL と SHERPA_TOKENS が必要です")
            if ParaConfig is None:
                raise RuntimeError("OfflineParaformerModelConfig が見つかりません")
            model_kwargs["paraformer"] = ParaConfig(model=cfg.model)
        elif mt == "transducer":
            if not (cfg.encoder and cfg.decoder and cfg.joiner and cfg.tokens):
                raise ValueError("Transducer には ENCODER/DECODER/JOINER/TOKENS が必要です")
            if TransConfig is None:
                raise RuntimeError("OfflineTransducerModelConfig が見つかりません")
            model_kwargs["transducer"] = TransConfig(
                encoder=cfg.encoder,
                decoder=cfg.decoder,
                joiner=cfg.joiner,
            )
        else:  # whisper
            if not (cfg.encoder and cfg.decoder):
                raise ValueError("Whisper には ENCODER/DECODER が必要です")
            if WhisperConfig is None:
                raise RuntimeError("OfflineWhisperModelConfig が見つかりません")
            # API差異に対応（language, taskが必要なバージョンに合わせる）
            try:
                model_kwargs["whisper"] = WhisperConfig(
                    encoder=cfg.encoder,
                    decoder=cfg.decoder,
                    language=cfg.language or "auto",
                    task=cfg.task or "transcribe",
                )
            except TypeError:
                # 古い版: encoder/decoderのみ
                model_kwargs["whisper"] = WhisperConfig(
                    encoder=cfg.encoder,
                    decoder=cfg.decoder,
                )

        OfflineRecognizer = getattr(self.so, "OfflineRecognizer")
        RecognizerCfg = getattr(self.so, "OfflineRecognizerConfig", None)
        decoding = os.getenv("SHERPA_DECODING_METHOD", "greedy_search")

        # Try 1: Preferred wrapper/factory
        if RecognizerCfg is not None:
            rec_cfg = RecognizerCfg(model_config=OfflineModelConfig(**model_kwargs), decoding_method=decoding)
            factory = getattr(self.so, "create_offline_recognizer", None)
            if factory is not None:
                try:
                    self.recognizer = factory(rec_cfg)  # type: ignore[misc]
                    return
                except Exception:
                    pass
            if hasattr(OfflineRecognizer, "from_config"):
                try:
                    self.recognizer = OfflineRecognizer.from_config(rec_cfg)  # type: ignore[attr-defined]
                    return
                except Exception:
                    pass
            try:
                self.recognizer = OfflineRecognizer(rec_cfg)
                return
            except TypeError:
                pass

        # Try 2: Direct kwargs
        try:
            self.recognizer = OfflineRecognizer(model_config=OfflineModelConfig(**model_kwargs), decoding_method=decoding)
            return
        except TypeError:
            pass

        # Try 3: Legacy positional
        try:
            self.recognizer = OfflineRecognizer(OfflineModelConfig(**model_kwargs))
            return
        except TypeError as e:
            raise RuntimeError("sherpa-onnx OfflineRecognizer の初期化に失敗しました。互換のあるバージョンへ変更してください。") from e

    def _save_wav(self, pcm: np.ndarray, sample_rate: int) -> str:
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        return path

    def transcribe(self, pcm: np.ndarray, sample_rate: int) -> str:
        """Transcribe PCM audio using sherpa-onnx OfflineRecognizer.

        Accepts int16 PCM (preferred) or float32 in [-1, 1].
        Avoids relying on sherpa_onnx.read_wave to improve compatibility
        across versions; feeds audio directly to the recognizer.
        """
        if pcm is None or pcm.size == 0:
            return ""

        # Ensure mono int16 -> float32 [-1, 1]
        if pcm.dtype == np.int16:
            audio = (pcm.astype(np.float32) / 32768.0).copy()
        elif pcm.dtype == np.float32:
            audio = pcm
        else:
            audio = pcm.astype(np.float32, copy=False)
            # Heuristic: if values look like int16 range, normalize
            if audio.max(initial=0) > 1.0 or audio.min(initial=0) < -1.0:
                audio = audio / 32768.0

        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio)
        # Some versions benefit from signalling end-of-input
        if hasattr(stream, "input_finished"):
            try:
                stream.input_finished()
            except Exception:
                pass
        self.recognizer.decode_stream(stream)
        res = getattr(getattr(stream, "result", None), "text", "")
        return (res or "").strip()


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "on"}


def _sherpa_config_from_env() -> Optional[SherpaConfig]:
    mt = os.getenv("SHERPA_MODEL_TYPE", "").strip().lower()
    if not mt:
        return None
    cfg = SherpaConfig(
        model_type=mt,
        tokens=os.getenv("SHERPA_TOKENS"),
        model=os.getenv("SHERPA_MODEL"),
        encoder=os.getenv("SHERPA_ENCODER"),
        decoder=os.getenv("SHERPA_DECODER"),
        joiner=os.getenv("SHERPA_JOINER"),
        num_threads=int(os.getenv("SHERPA_NUM_THREADS", "1")),
        provider=os.getenv("SHERPA_PROVIDER", "cpu"),
        language=os.getenv("SHERPA_LANGUAGE", "auto"),
        task=os.getenv("SHERPA_TASK", "transcribe"),
    )
    # Quick sanity of file existence when provided
    for p in [cfg.tokens, cfg.model, cfg.encoder, cfg.decoder, cfg.joiner]:
        if p and not os.path.exists(p):
            logger.warning(f"sherpa-onnx: モデルファイルが見つかりません: {p}")
    return cfg


def is_sherpa_available() -> bool:
    try:
        import importlib
        importlib.import_module("sherpa_onnx")
        return True
    except Exception:
        return False


def create_local_stt_engine(preferred: str, _unused=None) -> BaseLocalSTT:
    preferred = (preferred or "").lower()
    cfg = _sherpa_config_from_env()
    if not cfg:
        raise RuntimeError("ローカルSTTにsherpaを使用するため SHERPA_* の設定が必要です")
    if not is_sherpa_available():
        raise RuntimeError("sherpa-onnx がインストールされていません。pip install sherpa-onnx を実行してください")
    return SherpaLocalSTT(cfg)

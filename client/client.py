#!/usr/bin/env python3
"""
Wake Saiteku Client - 端末側のWake Word検知と音声送信クライアント
オンライン優先、オフラインフォールバック対応
"""

import os
import sys
import time
import json
import queue
import threading
import wave
import re
import requests
import random
import tempfile
import logging
import logging.handlers
import uuid
import subprocess
import shlex
import atexit
import shutil
from pathlib import Path
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import sounddevice as sd
import webrtcvad
import queue as _q

# --- プロジェクトルートを import path に追加（python client/client.py 実行対応）
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.wake_utils import recent_text_from_history, is_wake_in_text, squash_repeated_tokens, find_wake_match
from utils.text_utils import dedupe_transcript
from utils.stt_backends import create_local_stt_engine
try:
    from client.audio_device_manager import choose_devices
except ModuleNotFoundError:
    if __package__ in {None, ""}:
        from audio_device_manager import choose_devices  # type: ignore
    else:
        raise

# .env の自動読み込み（任意）
try:
    from dotenv import load_dotenv  # type: ignore
    
    # 1. プロジェクトルートの .env を読み込む (モデルパスなど)
    root_dotenv_path = Path(_PROJECT_ROOT) / ".env"
    if root_dotenv_path.exists():
        load_dotenv(dotenv_path=root_dotenv_path, override=True)

    # 2. config/client.env を読み込む (サーバーURLなど、上書き可能)
    client_dotenv_path = Path(_PROJECT_ROOT) / "config" / "client.env"
    if client_dotenv_path.exists():
        load_dotenv(dotenv_path=client_dotenv_path, override=True)

except Exception:
    # dotenv が無くても、そのまま続行
    pass

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
        default_dir = str((Path(__file__).resolve().parents[1] / "logs").resolve())
        log_dir = os.getenv("LOG_DIR", default_dir)
        try:
            os.makedirs(log_dir, exist_ok=True)
            handler = logging.handlers.TimedRotatingFileHandler(
                os.path.join(log_dir, "client.log"), when="midnight", backupCount=7, encoding="utf-8"
            )
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            logger.addHandler(handler)
        except Exception as e:
            logger.warning(f"ログファイルの設定に失敗: {e}")

_configure_file_logging()

# ========== 設定 ==========
@dataclass
class AudioConfig:
    SAMPLE_RATE: int = 16000
    FRAME_DUR_MS: int = 20
    CHANNELS: int = 1
    DTYPE: str = 'int16'
    INPUT_DEVICE: Optional[str] = os.getenv("AUDIO_INPUT_DEVICE")  # device index or name
    OUTPUT_DEVICE: Optional[str] = os.getenv("AUDIO_OUTPUT_DEVICE")  # device index or name
    
    @property
    def frame_length(self) -> int:
        return int(self.SAMPLE_RATE * self.FRAME_DUR_MS / 1000)

@dataclass
class WakeConfig:
    WAKE_TIMEOUT_S: float = float(os.getenv("WAKE_TIMEOUT_S", "4.0"))
    WAKE_REQUIRE_BOTH: bool = os.getenv("WAKE_REQUIRE_BOTH", "false").lower() == "true"
    WAKE_WORDS: List[Tuple[str, str]] = None
    
    def __post_init__(self):
        if self.WAKE_WORDS is None:
            self.WAKE_WORDS = [
                ("もしもし", r"もしもし"),
                ("サイテク", r"(サイテク|さいてく|ｻｲﾃｸ|さいテク|サイテック|さいてっく|サイトク|さいとく)")
            ]

@dataclass
class VADConfig:
    VAD_MODE: int = int(os.getenv("VAD_MODE", "2"))  # 0-3, 3が最も厳しい
    MIN_UTTERANCE_MS: int = int(os.getenv("MIN_UTTERANCE_MS", "300"))
    # 追加の互換ENV（WAKE_VAD_SILENCE_MS）があれば優先
    END_SILENCE_MS: int = int(os.getenv("WAKE_VAD_SILENCE_MS", os.getenv("END_SILENCE_MS", "800")))
    # 互換ENV: WAKE_MAX_RECORD_SEC（秒）
    MAX_RECORDING_MS: int = int(
        os.getenv("MAX_RECORDING_MS",
                 str(int(float(os.getenv("WAKE_MAX_RECORD_SEC", "10")) * 1000)))
    )

@dataclass
class ServerConfig:
    REMOTE_URL: str = os.getenv("SERVER_URL", "http://127.0.0.1:8000/inference")
    # Optional alternate URL for quick switching (e.g., legacy<->v1)
    ALT_URL: str = os.getenv("SERVER_URL_ALT", "")
    # auto | legacy | v1
    SERVER_API_MODE: str = os.getenv("SERVER_API_MODE", "auto").strip().lower()
    LOCAL_STT_ENABLED: bool = os.getenv("LOCAL_STT_ENABLED", "true").lower() == "true"
    LOCAL_LLM_URL: str = os.getenv("LLM_LOCAL_URL", "http://127.0.0.1:8081/v1/chat/completions")
    LOCAL_LLM_MODEL: str = os.getenv("LLM_LOCAL_MODEL", "local-model")
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "60"))
    # HTTP接続まわり詳細
    CONNECT_TIMEOUT: float = float(os.getenv("CONNECT_TIMEOUT", "3"))
    READ_TIMEOUT: float = float(os.getenv("READ_TIMEOUT", "60"))
    HTTP_RETRIES: int = int(os.getenv("HTTP_RETRIES", "5"))
    RETRY_BACKOFF: float = float(os.getenv("RETRY_BACKOFF", "0.6"))
    # LLM速度/品質調整
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "200"))
    LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", "0.9"))
    # 外部Chat API（OpenAI互換）設定（ストリーミング対応）
    # 1/true/yes/on も true とみなす
    PREFER_CHAT_API: bool = os.getenv("PREFER_CHAT_API", "true").lower() in {"1", "true", "yes", "on"}
    CHAT_API_BASE_URL: str = os.getenv("CHAT_API_BASE_URL", "")  # 例: http://localhost:8000/v1
    CHAT_API_KEY: str = os.getenv("CHAT_API_KEY", "")
    CHAT_API_MODEL: str = os.getenv("CHAT_API_MODEL", os.getenv("LLM_LOCAL_MODEL", "local-model"))
    # TTS（任意）
    # WAKE_SPEAK が指定された場合はそれを優先（1/true/yes/on）
    ENABLE_TTS_PLAYBACK: bool = (os.getenv("WAKE_SPEAK", os.getenv("ENABLE_TTS_PLAYBACK", "false")).lower() in {"1", "true", "yes", "on"})
    SERVER_TTS_URL: str = os.getenv("SERVER_TTS_URL", "")  # 未指定時は REMOTE_URL から導出
    TTS_ENGINE: str = os.getenv("TTS_ENGINE", "")  # kokoro_onnx|kokoro_pt|piper|openjtalk など
    TTS_VOICE: str = os.getenv("TTS_VOICE", "")   # 例: jf_alpha
    TTS_SAMPLE_RATE: int = int(os.getenv("TTS_SAMPLE_RATE", "24000"))
    TTS_SPEED: float = float(os.getenv("TTS_SPEED", "1.0"))
    AUTO_START_TTS: bool = os.getenv("WAKE_AUTO_START_TTS", os.getenv("AUTO_START_TTS", "true")).lower() in {"1", "true", "yes", "on"}
    TTS_AUTOSTART_CMD: str = os.getenv("TTS_AUTOSTART_CMD", "").strip()
    TTS_AUTOSTART_WAIT: float = float(os.getenv("TTS_AUTOSTART_WAIT", "5.0"))
    FORCE_OUTPUT_CHANNELS: Optional[int] = int(os.getenv("WAKE_FORCE_OUTPUT_CH", "0")) or None
    FALLBACK_OUTPUT_DEVICES: List[int] = field(default_factory=lambda: [
        int(i) for i in os.getenv("WAKE_FALLBACK_OUT_DEV", "").replace(" ", "").split(",") if i.strip().isdigit()
    ])
    FORCE_OUTPUT_SAMPLE_RATE: Optional[int] = int(os.getenv("WAKE_FORCE_OUTPUT_SR", "0")) or None
    STREAM_FALLBACK_ENABLED: bool = os.getenv("WAKE_STREAM_FALLBACK", "true").lower() in {"1", "true", "yes", "on"}
    # SSEトークンを逐次TTSするか
    TTS_STREAMING: bool = os.getenv("WAKE_TTS_STREAMING", "false").lower() in {"1", "true", "yes", "on"}
    # 音声APIのSSEストリーミングを使用するか（/v1/audio/inference/stream）
    AUDIO_SSE: bool = os.getenv("AUDIO_SSE", "false").lower() in {"1", "true", "yes", "on"}

# 設定インスタンス
audio_config = AudioConfig()
wake_config = WakeConfig()
vad_config = VADConfig()
server_config = ServerConfig()

_kokoro_proc: Optional[subprocess.Popen] = None


def _stop_autostart_tts() -> None:
    global _kokoro_proc
    if not _kokoro_proc:
        return
    try:
        if _kokoro_proc.poll() is None:
            _kokoro_proc.terminate()
            try:
                _kokoro_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _kokoro_proc.kill()
    except Exception as e:
        logger.debug(f"TTS自動起動プロセス停止で例外: {e}")
    finally:
        _kokoro_proc = None


atexit.register(_stop_autostart_tts)


def _tts_health_url(tts_url: str) -> str:
    try:
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(tts_url)
        path = parsed.path or ""
        if path.endswith("/tts"):
            path = path[: -len("/tts")] + "/voices"
        elif path:
            if not path.endswith("/"):
                path = path + "/"
            path = path + "voices"
        else:
            path = "/voices"
        return urlunparse((parsed.scheme or "http", parsed.netloc, path, "", "", ""))
    except Exception:
        return tts_url


def _check_tts_ready(tts_url: str) -> bool:
    health_url = _tts_health_url(tts_url)
    try:
        r = requests.get(health_url, timeout=1.5)
        return r.status_code == 200
    except requests.RequestException:
        return False


def _wait_for_tts(tts_url: str, timeout_s: float) -> bool:
    if timeout_s <= 0:
        return _check_tts_ready(tts_url)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if _check_tts_ready(tts_url):
            return True
        time.sleep(0.3)
    return _check_tts_ready(tts_url)


def _ensure_local_tts_server() -> None:
    global _kokoro_proc
    if not server_config.ENABLE_TTS_PLAYBACK or not server_config.AUTO_START_TTS:
        return
    tts_url = server_config.SERVER_TTS_URL.strip() or _derive_tts_url_from_inference(server_config.REMOTE_URL)
    try:
        from urllib.parse import urlparse

        parsed = urlparse(tts_url)
    except Exception as e:
        logger.warning(f"TTS自動起動: URL解析に失敗しました ({e})")
        return

    host = (parsed.hostname or "").lower()
    if host not in {"127.0.0.1", "localhost"}:
        return

    port = parsed.port or (443 if (parsed.scheme or "http").lower() == "https" else 80)

    if _check_tts_ready(tts_url):
        return

    if _kokoro_proc and _kokoro_proc.poll() is None:
        if _wait_for_tts(tts_url, max(0.0, server_config.TTS_AUTOSTART_WAIT)):
            return
        logger.warning("Kokoro TTS が応答しないため再起動します。")
        _stop_autostart_tts()

    u_tts_dir = Path(_PROJECT_ROOT) / "u-tts"
    if not u_tts_dir.exists():
        logger.warning("Kokoro自動起動をスキップしました (u-tts ディレクトリが見つかりません)")
        return

    env = os.environ.copy()
    existing_py_path = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(u_tts_dir) + (os.pathsep + existing_py_path if existing_py_path else "")

    if server_config.TTS_AUTOSTART_CMD:
        cmd_str = server_config.TTS_AUTOSTART_CMD.format(host=host, port=port)
        logger.info(f"Kokoro TTSサーバーを自動起動: {cmd_str}")
        try:
            _kokoro_proc = subprocess.Popen(cmd_str, shell=True, cwd=str(u_tts_dir), env=env)
        except Exception as e:
            logger.warning(f"Kokoro TTS自動起動に失敗しました: {e}")
            return
    else:
        python_cmd = os.getenv("TTS_AUTOSTART_PYTHON", "").strip()
        if not python_cmd:
            python_cmd = shutil.which("python3") or sys.executable
        log_level = os.getenv("TTS_AUTOSTART_LOG_LEVEL", "info")
        cmd_list = [
            python_cmd,
            "-m",
            "uvicorn",
            "mini_kokoro_server:app",
            "--host",
            host or "127.0.0.1",
            "--port",
            str(port),
            "--log-level",
            log_level,
        ]
        pretty_cmd = " ".join(shlex.quote(c) for c in cmd_list)
        logger.info(f"Kokoro TTSサーバーを自動起動: {pretty_cmd}")
        try:
            _kokoro_proc = subprocess.Popen(cmd_list, cwd=str(u_tts_dir), env=env)
        except Exception as e:
            logger.warning(f"Kokoro TTS自動起動に失敗しました: {e}")
            _kokoro_proc = None
            return

    if not _wait_for_tts(tts_url, max(0.0, server_config.TTS_AUTOSTART_WAIT)):
        logger.warning("Kokoro TTSが起動しませんでした。ログを確認してください。")


# ========== TTS ヘルパー ==========
def _resolve_output_device(preferred_idx: Optional[int]) -> Tuple[Optional[int], Optional[dict]]:
    candidates: List[int] = []
    if preferred_idx is not None and preferred_idx != -1:
        candidates.append(int(preferred_idx))
    for idx in server_config.FALLBACK_OUTPUT_DEVICES:
        if idx not in candidates:
            candidates.append(idx)
    if not candidates:
        try:
            dev = sd.query_devices(preferred_idx) if preferred_idx not in (None, -1) else None
        except Exception:
            dev = None
        return preferred_idx, dev

    for idx in candidates:
        try:
            dev = sd.query_devices(idx)
            max_out = int(dev.get("max_output_channels", dev.get("max_output_channels", 0)))
            if max_out <= 0:
                continue
            if idx != preferred_idx:
                logger.info(f"TTS出力デバイスをフォールバックします: idx={idx} name={dev.get('name')}")
            return idx, dev
        except Exception as e:
            logger.warning(f"出力デバイス idx={idx} の検査に失敗: {e}")
            continue

    try:
        dev = sd.query_devices(preferred_idx) if preferred_idx not in (None, -1) else None
    except Exception:
        dev = None
    return preferred_idx, dev


def _resample_1d(buffer: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr <= 0 or dst_sr <= 0 or src_sr == dst_sr or buffer.size == 0:
        return buffer.astype(np.float32, copy=False)
    duration = buffer.shape[0] / float(src_sr)
    new_len = max(1, int(round(duration * dst_sr)))
    t_src = np.linspace(0.0, duration, buffer.shape[0], endpoint=False, dtype=np.float64)
    t_dst = np.linspace(0.0, duration, new_len, endpoint=False, dtype=np.float64)
    resampled = np.interp(t_dst, t_src, buffer.astype(np.float64))
    return resampled.astype(np.float32)


def _resample_buffer(buffer: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if buffer.ndim == 1:
        return _resample_1d(buffer, src_sr, dst_sr)
    channels = [
        _resample_1d(buffer[:, ch], src_sr, dst_sr)
        for ch in range(buffer.shape[1])
    ]
    return np.column_stack(channels).astype(np.float32, copy=False)


def _prepare_playback_buffer(buffer: np.ndarray, sr: int, dev_out: Optional[dict]) -> Tuple[np.ndarray, int]:
    buf = np.asarray(buffer, dtype=np.float32)
    max_out = int((dev_out or {}).get("max_output_channels", 1)) or 1
    target_ch = server_config.FORCE_OUTPUT_CHANNELS or (2 if max_out >= 2 else 1)
    if target_ch > max_out and max_out > 0:
        logger.warning(f"指定チャンネル数 {target_ch} がデバイス上限 {max_out} を超えています。{max_out}ch に調整します。")
        target_ch = max_out
    if target_ch <= 1:
        if buf.ndim > 1:
            buf = buf[:, 0]
    else:
        if buf.ndim == 1:
            buf = np.tile(buf[:, None], (1, target_ch))
        elif buf.shape[1] != target_ch:
            if buf.shape[1] > target_ch:
                buf = buf[:, :target_ch]
            else:
                first = buf[:, 0:1]
                repeats = [first] + [buf[:, -1:]] * (target_ch - 1)
                buf = np.concatenate(repeats, axis=1)

    target_sr = server_config.FORCE_OUTPUT_SAMPLE_RATE or sr
    if target_sr and target_sr > 0 and target_sr != sr:
        buf = _resample_buffer(buf, sr, target_sr)
        sr = target_sr
    return buf.astype(np.float32, copy=False), sr


def _fallback_playback_buffer(buffer: np.ndarray, sr: int) -> Tuple[np.ndarray, int]:
    buf = buffer[:, 0] if buffer.ndim > 1 else buffer
    target_sr = server_config.FORCE_OUTPUT_SAMPLE_RATE or 16000
    if target_sr and target_sr != sr:
        buf = _resample_1d(buf, sr, target_sr)
        sr = target_sr
    return buf.astype(np.float32, copy=False), sr


# ========== TTS テストユーティリティ ==========
def _run_tts_startup_check(test_text: str, attempt: int, max_attempts: int) -> bool:
    tts_url = server_config.SERVER_TTS_URL
    payload = {"text": test_text}
    if server_config.TTS_ENGINE:
        payload["engine"] = server_config.TTS_ENGINE
    if server_config.TTS_VOICE:
        payload["voice"] = server_config.TTS_VOICE
    if server_config.TTS_SPEED:
        payload["speed"] = float(server_config.TTS_SPEED)
    if server_config.TTS_SAMPLE_RATE:
        payload["sample_rate"] = int(server_config.TTS_SAMPLE_RATE)

    try:
        logger.info(f"TTS疎通テスト {attempt}/{max_attempts}: {tts_url}")
        r = requests.post(tts_url, json=payload, timeout=10)
        if r.status_code != 200 or (r.headers.get("Content-Type") or "").split(";")[0] != "audio/wav":
            logger.warning(f"⚠️ TTSテスト失敗: status={r.status_code} content-type={r.headers.get('Content-Type')}")
            return False

        import io as _io
        import wave as _wave

        with _wave.open(_io.BytesIO(r.content), "rb") as wf:
            ch = wf.getnchannels()
            sr = wf.getframerate()
            n = wf.getnframes()
            pcm = np.frombuffer(wf.readframes(n), dtype=np.int16)
            if ch > 1:
                pcm = pcm.reshape(-1, ch).mean(axis=1).astype(np.int16, copy=False)

        try:
            dflt = sd.default.device
            out_idx_default = dflt[1] if isinstance(dflt, (tuple, list)) or hasattr(dflt, "__getitem__") else None
        except Exception:
            out_idx_default = None

        out_idx, dev_out = _resolve_output_device(out_idx_default)
        if out_idx is None and out_idx_default not in (None, -1):
            out_idx = out_idx_default

        play_buf = (pcm.astype(np.float32) / 32767.0).astype(np.float32)
        play_buf, sr = _prepare_playback_buffer(play_buf, sr, dev_out)

        device_arg = out_idx if out_idx not in (None, -1) else None
        try:
            sd.play(play_buf, samplerate=sr, blocking=False, device=device_arg)
        except Exception as e:
            logger.warning(f"TTS初期化テストの再生に失敗: {e}. 1ch/16kHzで再試行します。")
            fallback_buf, fallback_sr = _fallback_playback_buffer(play_buf, sr)
            try:
                sd.play(fallback_buf, samplerate=fallback_sr, blocking=False, device=device_arg)
            except Exception as e2:
                logger.warning(f"TTS初期化テストのフォールバック再生も失敗: {e2}")
                return False
        logger.info("✅ TTS初期化テスト完了")
        time.sleep(float(os.getenv("WAKE_TTS_TEST_WAIT", "1.0")))
        return True
    except Exception as e:
        logger.warning(f"⚠️ TTSテストエラー: {e}")
        return False
    finally:
        try:
            sd.stop()
        except Exception:
            pass


# ========== SSEテキスト抽出ユーティリティ ==========
def _extract_stream_reply_segments(obj: Any) -> List[str]:
    """Extract textual reply fragments from diverse streaming JSON payloads."""

    results: List[str] = []
    seen: set[str] = set()

    def add(text: Optional[str]) -> None:
        if not isinstance(text, str):
            return
        t = text.strip()
        if not t or t in seen:
            return
        seen.add(t)
        results.append(t)

    def collect(value: Any) -> None:
        if isinstance(value, str):
            add(value)
            return
        if isinstance(value, dict):
            for key in ("content", "text", "output_text", "reply", "value"):
                if key in value:
                    add(value.get(key))
            for key in (
                "content",
                "text",
                "output_text",
                "reply",
                "values",
                "data",
                "message",
                "messages",
                "delta",
                "output",
                "outputs",
                "segments",
                "parts",
                "choices",
            ):
                if key in value:
                    collect(value[key])
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    if isinstance(obj, dict):
        collect(obj.get("message"))
        collect(obj.get("response"))
        collect(obj.get("result"))
        collect(obj.get("output"))
        collect(obj.get("outputs"))
        collect(obj.get("data"))
        delta = obj.get("delta")
        if isinstance(delta, dict):
            collect(delta.get("message"))
            collect(delta.get("data"))
        choices = obj.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if isinstance(choice, dict):
                    collect(choice.get("message"))
                    collect(choice.get("content"))

    return results


# ========== Voskモデルは使用しません（sherpa-ONNX専用） ==========

# ========== 逐次TTS用の簡易キュー ==========
class _TTSWorker:
    def __init__(self, audio_proc: 'AudioProcessor'):
        self.audio_proc = audio_proc
        self.q: _q.Queue[str] = _q.Queue()
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()

    def enqueue(self, text: str):
        t = (text or '').strip()
        if t:
            self.q.put(t)

    def stop(self):
        self._stop.set()
        try:
            self.q.put_nowait('')
        except Exception:
            pass
        try:
            if self._th.is_alive():
                self._th.join(timeout=0.5)
        except Exception:
            pass

    def _loop(self):
        while not self._stop.is_set():
            try:
                text = self.q.get(timeout=0.2)
            except _q.Empty:
                continue
            if not text:
                continue
            try:
                speak_via_server_tts(text, self.audio_proc)
            except Exception:
                pass

# ========== オーディオ処理 ==========
class AudioProcessor:
    def __init__(self):
        self.q = queue.Queue()
        self.vad = webrtcvad.Vad(vad_config.VAD_MODE)
        self.stream = None
        self._last_level_log = 0.0
        self.last_frame_time = time.time()
        self._squelch_until = 0.0  # 再生音の取り込み抑制用

        # 入力ストリームの実際の dtype / samplerate を保持
        self._in_dtype: Optional[str] = None   # 'float32' or 'int16'
        self._in_samplerate: Optional[int] = None
        self._mode: str = "input"  # or "duplex"

        # 16k/20ms に整形するためのバッファ
        self._rs_outbuf = np.zeros(0, dtype=np.int16)

        # ダミーストリーム管理
        self._dummy_thread = None
        self._dummy_stop = threading.Event()

    @staticmethod
    def _resample_linear_i16(x: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        if from_sr == to_sr or x.size == 0:
            return x
        xf = x.astype(np.float32)
        n_in = xf.shape[0]
        n_out = int(round(n_in * (to_sr / float(from_sr))))
        if n_out <= 0:
            return np.zeros(0, dtype=np.int16)
        xp = np.linspace(0.0, 1.0, num=n_in, endpoint=False, dtype=np.float32)
        xq = np.linspace(0.0, 1.0, num=n_out, endpoint=False, dtype=np.float32)
        y = np.interp(xq, xp, xf)
        y = np.clip(y, -32768.0, 32767.0).astype(np.int16)
        return y

    def _handle_input(self, indata, status):
        if status:
            logger.warning(f"オーディオステータス: {status}")
        now = time.time()
        if now < self._squelch_until:
            self.last_frame_time = now
            return
        x = indata
        if x.ndim == 2 and x.shape[1] > 1:
            x = x.mean(axis=1)
        x = x.reshape(-1)

        # float32 -> int16 へ変換（CoreAudio互換のため float32 を優先）
        if self._in_dtype == 'float32' or (hasattr(x, 'dtype') and x.dtype.kind == 'f'):
            f32 = x.astype(np.float32, copy=False)
            pcm = np.clip(f32 * 32767.0, -32768.0, 32767.0).astype(np.int16)
        else:
            pcm = x.astype(np.int16, copy=False)

        # サンプルレートが16k以外なら軽量リサンプル
        sr_in = self._in_samplerate or audio_config.SAMPLE_RATE
        if sr_in != audio_config.SAMPLE_RATE:
            pcm = self._resample_linear_i16(pcm, sr_in, audio_config.SAMPLE_RATE)

        # 20ms フレームに整形してキューへ
        if pcm.size:
            self._rs_outbuf = np.concatenate([self._rs_outbuf, pcm])
            frame_len = audio_config.frame_length  # 320 @16k/20ms
            while self._rs_outbuf.size >= frame_len:
                frame = self._rs_outbuf[:frame_len]
                self._rs_outbuf = self._rs_outbuf[frame_len:]
                self.q.put(frame.reshape(-1, audio_config.CHANNELS))

        self.last_frame_time = now

    def _cb_input(self, indata, frames, time_info, status):
        self._handle_input(indata, status)

    def _cb_duplex(self, indata, outdata, frames, time_info, status):
        # 無音を常時出力して SCO/HFP を維持
        if outdata is not None:
            outdata.fill(0)
        self._handle_input(indata, status)

    def squelch(self, duration_sec: float):
        self._squelch_until = max(self._squelch_until, time.time() + max(0.0, duration_sec))

    def _open_stream(self, *, device_index: int, samplerate: int, dtype: str, duplex: bool, blocksize: int):
        self._in_dtype = dtype
        self._in_samplerate = samplerate
        self._mode = "duplex" if duplex else "input"
        # 実際に使用するサンプルレートを全体設定に反映（20msフレーム長やVAD等と整合）
        try:
            audio_config.SAMPLE_RATE = int(samplerate)
        except Exception:
            pass
        if duplex:
            self.stream = sd.Stream(
                device=(device_index, device_index),
                channels=(audio_config.CHANNELS, 1),
                samplerate=samplerate,
                dtype=dtype,
                blocksize=blocksize,
                callback=self._cb_duplex,
                latency='high',
            )
        else:
            self.stream = sd.InputStream(
                device=device_index,
                channels=audio_config.CHANNELS,
                samplerate=samplerate,
                dtype=dtype,
                blocksize=blocksize,
                callback=self._cb_input,
                latency='high',
            )
        self.stream.start()

    def start_stream(self):
        if self.stream is not None:
            return

        try:
            dflt = sd.default.device
            # sounddevice returns an _InputOutputPair (tuple-like). Safely extract input index.
            if isinstance(dflt, (tuple, list)) or hasattr(dflt, "__getitem__"):
                in_idx = dflt[0]
            else:
                in_idx = dflt
            if in_idx is None or (isinstance(in_idx, int) and in_idx < 0):
                raise RuntimeError("入力デバイスが未選択です")

            dinfo = sd.query_devices(in_idx)
            name = dinfo['name']
            ch = audio_config.CHANNELS
            # 列挙直後は不安定なため少し待つ
            time.sleep(0.5)

            # HFP互換を優先: 16k → 8k の順で試行（まずデュプレクス、次に入力単独）
            sr_candidates = [16000, 8000]

            last_err: Optional[Exception] = None
            for attempt in range(3):
                for sr in sr_candidates:
                    # まずデュプレクス（入出力同一デバイス）を試す
                    try:
                        # 常に入出力を同一デバイスに揃える（SCO維持のため）
                        sd.default.device = (in_idx, in_idx)
                        # duplex open
                        sd.check_input_settings(device=in_idx, channels=ch, samplerate=sr, dtype='int16')
                        sd.check_output_settings(device=in_idx, channels=1, samplerate=sr, dtype='int16')
                        self._open_stream(device_index=in_idx, samplerate=sr, dtype='int16', duplex=True, blocksize=None)  # type: ignore[arg-type]
                        logger.info(
                            f"🎤 入力開始 dev='{name}' hw_sr≈{sr}Hz → proc_sr={audio_config.SAMPLE_RATE}Hz "
                            f"block=default dtype=int16 mode=duplex"
                        )
                        self.last_frame_time = time.time()
                        return
                    except Exception as e:
                        last_err = e
                        # 次に入力単独
                        try:
                            sd.check_input_settings(device=in_idx, channels=ch, samplerate=sr, dtype='int16')
                            self._open_stream(device_index=in_idx, samplerate=sr, dtype='int16', duplex=False, blocksize=None)  # type: ignore[arg-type]
                            logger.info(
                                f"🎤 入力開始 dev='{name}' hw_sr≈{sr}Hz → proc_sr={audio_config.SAMPLE_RATE}Hz "
                                f"block=default dtype=int16 mode=input"
                            )
                            self.last_frame_time = time.time()
                            # 任意: HFP安定のための無音出力 keep-alive（環境変数で有効化）
                            self._maybe_start_keepalive()
                            return
                        except Exception as e2:
                            last_err = e2
                            time.sleep(0.5)
                            continue

            raise RuntimeError(f"どのサンプルレートでも開けませんでした: {last_err}")

        except Exception as e:
            logger.error(f"オーディオストリーム作成失敗: {e}")
            self._start_dummy_stream()

    def stop_stream(self):
        if self.stream:
            if self.stream == "dummy":
                self._dummy_stop.set()
                try:
                    if self._dummy_thread and self._dummy_thread.is_alive():
                        self._dummy_thread.join(timeout=1.0)
                except Exception:
                    pass
                self.stream = None
                logger.info("🔇 ダミーストリーム停止")
            else:
                try:
                    self.stream.stop()
                except Exception as e:
                    logger.warning(f"⚠️ ストリーム stop エラー: {e}")
                try:
                    self.stream.close()
                except Exception as e:
                    logger.warning(f"⚠️ ストリーム close エラー: {e}")
                self.stream = None
                logger.info("🎤 オーディオストリーム停止")
        self._rs_outbuf = np.zeros(0, dtype=np.int16)

    def get_audio_frame(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_queue(self):
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break

    def _start_dummy_stream(self):
        logger.info("🔇 ダミーストリームモードで起動（マイク入力無し）")
        self.stream = "dummy"
        self._dummy_stop.clear()

        def dummy_audio_generator():
            frame = np.zeros((audio_config.frame_length, audio_config.CHANNELS), dtype=np.int16)
            interval = audio_config.FRAME_DUR_MS / 1000.0
            while not self._dummy_stop.is_set():
                self.q.put(frame.copy())
                time.sleep(interval)

        self._dummy_thread = threading.Thread(target=dummy_audio_generator, daemon=True)
        self._dummy_thread.start()

    # ---- 任意: HFP安定用の無音出力 keep-alive ----
    def _maybe_start_keepalive(self) -> None:
        try:
            enable = os.getenv("WAKE_BT_KEEPALIVE", "0").lower() in {"1", "true", "yes", "on"}
            dflt = sd.default.device
            out_idx = dflt[1] if isinstance(dflt, (tuple, list)) or hasattr(dflt, "__getitem__") else None
            if not enable or out_idx in (None, -1):
                return
            # すでに動作中なら何もしない
            if getattr(self, "_keepalive_stream", None):
                return
            sr = int(audio_config.SAMPLE_RATE)
            self._keepalive_stop = threading.Event()
            self._keepalive_stream = sd.OutputStream(device=out_idx, channels=1, samplerate=sr, dtype='int16')
            self._keepalive_stream.start()
            def _loop():
                frame_n = max(1, int(sr * (audio_config.FRAME_DUR_MS/1000.0)))
                silent = np.zeros((frame_n,), dtype=np.int16)
                while not self._keepalive_stop.is_set():
                    try:
                        self._keepalive_stream.write(silent)
                    except Exception:
                        break
            self._keepalive_th = threading.Thread(target=_loop, daemon=True)
            self._keepalive_th.start()
            logger.info("🔄 BT keep-alive 出力開始（SCO維持）")
        except Exception as e:
            logger.debug(f"keep-alive開始失敗: {e}")

    def _stop_keepalive(self) -> None:
        try:
            if hasattr(self, "_keepalive_stop") and self._keepalive_stop:
                self._keepalive_stop.set()
            if getattr(self, "_keepalive_stream", None):
                try:
                    self._keepalive_stream.stop(); self._keepalive_stream.close()
                except Exception:
                    pass
        finally:
            self._keepalive_stream = None

# ========== Wake Word検知（sherpa-ONNX） ==========


class SherpaWakeWordDetector:
    def __init__(self):
        try:
            import sherpa_onnx as so
        except Exception as e:
            raise RuntimeError(f"sherpa-onnx が見つかりません: {e}")
        self.so = so
        self.text_history: List[Tuple[str, float]] = []
        self.last_text = ""
        self._last_decode_t = 0.0
        self._init_offline_recognizer()
        self._pcm_buffer = np.zeros(0, dtype=np.int16)

    def _env(self, name: str, default: Optional[str] = None) -> Optional[str]:
        # WAKE_SHERPA_* を優先、なければ SHERPA_* を参照
        return os.getenv(name) or os.getenv(name.replace("WAKE_", ""), default)

    def _init_offline_recognizer(self):
        so = self.so
        mt = (self._env("WAKE_SHERPA_MODEL_TYPE", "").strip().lower())
        if not mt:
            raise RuntimeError("WAKE_SHERPA_MODEL_TYPE（またはSHERPA_MODEL_TYPE）が未設定です")
        tokens = self._env("WAKE_SHERPA_TOKENS")
        model = self._env("WAKE_SHERPA_MODEL")
        enc = self._env("WAKE_SHERPA_ENCODER")
        dec = self._env("WAKE_SHERPA_DECODER")
        join = self._env("WAKE_SHERPA_JOINER")
        num_threads = int(self._env("WAKE_SHERPA_NUM_THREADS", "1"))
        provider = self._env("WAKE_SHERPA_PROVIDER", "cpu")
        language = self._env("WAKE_SHERPA_LANGUAGE", "auto") or "auto"
        task = self._env("WAKE_SHERPA_TASK", "transcribe") or "transcribe"

        # Prefer Python wrapper classmethods first (sherpa-onnx >=1.10)
        OR = getattr(so, "OfflineRecognizer", None)
        if OR is None:
            raise RuntimeError("sherpa_onnx.OfflineRecognizer が見つかりません")

        try:
            if mt == "whisper" and hasattr(OR, "from_whisper"):
                if not (enc and dec):
                    raise RuntimeError("Whisperの設定不足（ENCODER/DECODER）")
                lang = language
                if (lang or "").lower() in {"", "auto", "autodetect", "auto_detect"}:
                    # sherpa-onnx whisper does not accept 'auto'; default to Japanese here.
                    lang = "ja"
                    logger.warning("Whisper言語が'auto'のため'ja'に設定しました。WAKE_SHERPA_LANGUAGEで変更できます。")
                self.recognizer = OR.from_whisper(
                    encoder=enc,
                    decoder=dec,
                    tokens=tokens or "",
                    language=lang,
                    task=task,
                    num_threads=num_threads,
                    provider=provider,
                )
                return
            if mt == "paraformer" and hasattr(OR, "from_paraformer"):
                if not (model and tokens):
                    raise RuntimeError("Paraformerの設定不足（MODEL/TOKENS）")
                self.recognizer = OR.from_paraformer(
                    paraformer=model,
                    tokens=tokens or "",
                    num_threads=num_threads,
                    provider=provider,
                )
                return
            if mt == "transducer" and hasattr(OR, "from_transducer"):
                if not (enc and dec and join and tokens):
                    raise RuntimeError("Transducerの設定不足（ENCODER/DECODER/JOINER/TOKENS）")
                self.recognizer = OR.from_transducer(
                    encoder=enc,
                    decoder=dec,
                    joiner=join,
                    tokens=tokens or "",
                    num_threads=num_threads,
                    provider=provider,
                )
                return
        except Exception:
            # Fall through to config-based initialization
            pass

        # Config-based initialization for other versions
        OfflineModelConfig = getattr(so, "OfflineModelConfig", None)
        if OfflineModelConfig is None:
            raise RuntimeError("sherpa_onnx.OfflineModelConfig が見つかりません")
        Para = getattr(so, "OfflineParaformerModelConfig", None)
        Trans = getattr(so, "OfflineTransducerModelConfig", None)
        Whisp = getattr(so, "OfflineWhisperModelConfig", None)

        kwargs = {
            "tokens": tokens or "",
            "num_threads": num_threads,
            "provider": provider,
        }
        if mt == "paraformer":
            if not (model and tokens and Para):
                raise RuntimeError("Paraformerの設定不足（MODEL/TOKENS）")
            kwargs["paraformer"] = Para(model=model)
        elif mt == "transducer":
            if not (enc and dec and join and tokens and Trans):
                raise RuntimeError("Transducerの設定不足（ENCODER/DECODER/JOINER/TOKENS）")
            kwargs["transducer"] = Trans(encoder=enc, decoder=dec, joiner=join)
        else:  # whisper
            if not (enc and dec and Whisp):
                raise RuntimeError("Whisperの設定不足（ENCODER/DECODER）")
            try:
                kwargs["whisper"] = Whisp(encoder=enc, decoder=dec, language=language, task=task)
            except TypeError:
                kwargs["whisper"] = Whisp(encoder=enc, decoder=dec)

        OfflineRecognizer = getattr(so, "OfflineRecognizer")
        RecognizerCfg = getattr(so, "OfflineRecognizerConfig", None)
        ModelCfg = getattr(so, "OfflineModelConfig")
        decoding = os.getenv("WAKE_SHERPA_DECODING_METHOD", os.getenv("SHERPA_DECODING_METHOD", "greedy_search"))

        # Try 1: create_offline_recognizer/ from_config/ constructor patterns
        if RecognizerCfg is not None:
            rec_cfg = RecognizerCfg(model_config=ModelCfg(**kwargs), decoding_method=decoding)
            factory = getattr(so, "create_offline_recognizer", None)
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
                # Some versions may accept calling via the wrapper
                self.recognizer = OfflineRecognizer(rec_cfg)
                return
            except TypeError:
                pass

        # Try 2: Pass model_config directly as kwargs
        try:
            self.recognizer = OfflineRecognizer(model_config=ModelCfg(**kwargs), decoding_method=decoding)
            return
        except TypeError:
            pass

        # Try 3: Legacy positional model config
        try:
            self.recognizer = OfflineRecognizer(ModelCfg(**kwargs))
            return
        except TypeError as e:
            raise RuntimeError("sherpa-onnx OfflineRecognizer の初期化に失敗しました。互換のあるバージョンへ変更してください。") from e

    def reset(self):
        self.text_history.clear()
        self.last_text = ""
        self._last_decode_t = 0.0
        self._pcm_buffer = np.zeros(0, dtype=np.int16)

    def process_audio(self, pcm_data: bytes) -> bool:
        # バッファに追記し、最大で (WAKE_TIMEOUT_S + 0.5)s を保持
        pcm = np.frombuffer(pcm_data, dtype=np.int16)
        if pcm.size:
            self._pcm_buffer = np.concatenate([self._pcm_buffer, pcm])
            max_len = int((wake_config.WAKE_TIMEOUT_S + 0.5) * audio_config.SAMPLE_RATE)
            if self._pcm_buffer.size > max_len:
                self._pcm_buffer = self._pcm_buffer[-max_len:]

        now = time.time()
        # デコードは700ms間隔で実行（負荷抑制）
        if now - self._last_decode_t >= 0.7 and self._pcm_buffer.size > 0:
            # int16 -> float32 [-1,1]
            f32 = (self._pcm_buffer.astype(np.float32) / 32768.0).copy()
            stream = self.recognizer.create_stream()
            stream.accept_waveform(audio_config.SAMPLE_RATE, f32)
            self.recognizer.decode_stream(stream)
            text = getattr(stream.result, "text", "") or ""
            if text and text != self.last_text:
                if logger.isEnabledFor(logging.DEBUG) or os.getenv("WAKE_DEBUG_PARTIAL", "").lower() in {"1","true","yes","on"}:
                    logger.debug(f"Wake partial: {text}")
                self.text_history.append((text, now))
                self.last_text = text
            self._last_decode_t = now

        cutoff_time = now - wake_config.WAKE_TIMEOUT_S - 0.5
        self.text_history = [(t, ts) for t, ts in self.text_history if ts >= cutoff_time]
        return self._check_wake_words()

    def _check_wake_words(self) -> bool:
        current_time = time.time()
        recent_text = recent_text_from_history(self.text_history, current_time, wake_config.WAKE_TIMEOUT_S)
        ok = is_wake_in_text(recent_text, require_both=wake_config.WAKE_REQUIRE_BOTH)
        if ok:
            logger.info(f"Wake Word検出: {squash_repeated_tokens(recent_text)}")
        return ok

# ========== 音声録音 ==========
class SpeechRecorder:
    def __init__(self, audio_processor: AudioProcessor):
        self.audio_processor = audio_processor
        
    def record_speech(self) -> np.ndarray:
        """VADを使用して音声を録音"""
        logger.info("録音開始...")
        
        speech_frames = []
        voiced_ms = 0
        silence_ms = 0
        started = False
        
        start_time = time.time()
        
        while True:
            # オーディオフレーム取得
            pcm = self.audio_processor.get_audio_frame(timeout=1.0)
            if pcm is None:
                continue
            
            # VAD判定
            is_speech = self.audio_processor.vad.is_speech(
                pcm.tobytes(),
                audio_config.SAMPLE_RATE
            )
            
            speech_frames.append(pcm)
            
            if is_speech:
                voiced_ms += audio_config.FRAME_DUR_MS
                silence_ms = 0
                if not started and voiced_ms >= vad_config.MIN_UTTERANCE_MS:
                    started = True
                    logger.info("発話開始検出")
            else:
                if started:
                    silence_ms += audio_config.FRAME_DUR_MS
            
            # 終了条件
            if started and silence_ms >= vad_config.END_SILENCE_MS:
                logger.info(f"発話終了検出 (無音 {silence_ms}ms)")
                break
            
            # 最大録音時間
            if len(speech_frames) * audio_config.FRAME_DUR_MS > vad_config.MAX_RECORDING_MS:
                logger.warning("最大録音時間に到達")
                break
            
            # タイムアウト（発話が始まらない場合）
            if not started and (time.time() - start_time) > 5.0:
                logger.warning("発話が検出されませんでした")
                break
        
        if speech_frames:
            return np.concatenate(speech_frames, axis=0).reshape(-1)
        return np.array([], dtype=audio_config.DTYPE)

# ========== 音声処理 ==========
def save_wav(pcm_data: np.ndarray, filepath: str):
    """PCMデータをWAVファイルとして保存"""
    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(audio_config.CHANNELS)
        wf.setsampwidth(2)  # int16
        wf.setframerate(audio_config.SAMPLE_RATE)
        wf.writeframes(pcm_data.tobytes())

def _dbfs(pcm: np.ndarray) -> float:
    """簡易dBFS計算 (int16想定)"""
    if pcm.size == 0:
        return float("-inf")
    # 16-bit full scale
    rms = np.sqrt(np.mean((pcm.astype(np.float32) / 32768.0) ** 2))
    if rms <= 1e-8:
        return float("-inf")
    return 20.0 * np.log10(rms)

# ========== オフラインSTT ==========
def stt_offline_vosk(pcm_data: np.ndarray) -> str:
    """Voskを使用したオフライン音声認識"""
    logger.info("オフラインSTT処理中...")
    
    recognizer = vosk.KaldiRecognizer(vosk_model, audio_config.SAMPLE_RATE)
    recognizer.SetWords(True)
    
    # 音声データを処理
    recognizer.AcceptWaveform(pcm_data.tobytes())
    result = json.loads(recognizer.FinalResult())
    
    text = result.get("text", "").strip()
    logger.info(f"オフラインSTT結果: {text}")
    return text

# ========== 便利関数（通知音） ==========
def _fade_in_out(signal: np.ndarray, fade_ratio: float = 0.05) -> np.ndarray:
    n = len(signal)
    if n == 0:
        return signal
    fr = max(1, int(n * fade_ratio))
    window = np.ones(n, dtype=np.float32)
    window[:fr] = np.linspace(0.0, 1.0, fr, dtype=np.float32)
    window[-fr:] = np.linspace(1.0, 0.0, fr, dtype=np.float32)
    return (signal * window).astype(np.float32)

def play_wake_sound():
    """簡単な二音『ポロン』通知音を再生"""
    try:
        base_sr = audio_config.SAMPLE_RATE
        # 880Hz -> 660Hz の二音（各90ms）
        dur1 = 0.09
        dur2 = 0.09
        t1 = np.linspace(0, dur1, int(base_sr * dur1), endpoint=False)
        t2 = np.linspace(0, dur2, int(base_sr * dur2), endpoint=False)
        tone1 = 0.2 * np.sin(2 * np.pi * 880 * t1).astype(np.float32)
        tone2 = 0.2 * np.sin(2 * np.pi * 660 * t2).astype(np.float32)
        signal = np.concatenate([_fade_in_out(tone1), _fade_in_out(tone2)])

        try:
            dflt = sd.default.device
            out_idx_default = dflt[1] if isinstance(dflt, (tuple, list)) or hasattr(dflt, "__getitem__") else None
        except Exception:
            out_idx_default = None

        out_idx, dev_out = _resolve_output_device(out_idx_default)
        if out_idx in (None, -1):
            logger.info("通知音をスキップ（出力デバイス未設定）")
            return

        play_buf, sr = _prepare_playback_buffer(signal, base_sr, dev_out)
        device_arg = out_idx if out_idx not in (None, -1) else None
        try:
            sd.play(play_buf, samplerate=sr, blocking=True, device=device_arg)
        except Exception as e:
            logger.warning(f"通知音の再生に失敗（フォールバック前）: {e}")
            fb_buf, fb_sr = _fallback_playback_buffer(play_buf, sr)
            try:
                sd.play(fb_buf, samplerate=fb_sr, blocking=True, device=device_arg)
            except Exception as e2:
                logger.warning(f"通知音フォールバックも失敗: {e2}")
    except Exception as e:
        logger.warning(f"通知音の再生に失敗: {e}")

    

# ========== オフラインLLM ==========
def llm_local_reply(prompt: str, interaction_id: str = "") -> str:
    """ローカルLLMを使用した応答生成"""
    logger.info("ローカルLLM処理中...")
    
    payload = {
        "model": server_config.LOCAL_LLM_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "あなたは「サイテク」という名前のアシスタントです。日本語で簡潔に応答してください。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": server_config.LLM_TEMPERATURE,
        "max_tokens": server_config.LLM_MAX_TOKENS,
        "top_p": server_config.LLM_TOP_P
    }
    
    try:
        response = requests.post(
            server_config.LOCAL_LLM_URL,
            json=payload,
            headers={"X-Interaction-ID": interaction_id} if interaction_id else None,
            timeout=server_config.REQUEST_TIMEOUT
        )
        response.raise_for_status()
        
        result = response.json()
        reply = result["choices"][0]["message"]["content"]
        logger.info(f"ローカルLLM応答: {reply[:50]}...")
        return reply
        
    except Exception as e:
        logger.error(f"ローカルLLMエラー: {e}")
        return f"申し訳ありません、応答を生成できませんでした。入力: {prompt}"

def llm_streaming_chat_api(prompt: str, interaction_id: str = "") -> str:
    """外部Chat API（OpenAI互換）のストリーミングで応答を取得して逐次表示する。
    利用条件: CHAT_API_BASE_URL, CHAT_API_KEY が設定されていること。
    失敗時は例外を投げる（呼び出し側でフォールバック）。
    """
    if not server_config.CHAT_API_BASE_URL or not server_config.CHAT_API_KEY:
        raise RuntimeError("Chat API設定が不足しています")

    base = server_config.CHAT_API_BASE_URL.rstrip("/")
    def build_url(b: str, path: str = "/chat/completions") -> str:
        return b.rstrip("/") + path
    url = build_url(base, "/chat/completions")
    headers = {
        "Authorization": f"Bearer {server_config.CHAT_API_KEY}",
        "Content-Type": "application/json",
    }
    if interaction_id:
        headers["X-Interaction-ID"] = interaction_id

    payload = {
        "model": server_config.CHAT_API_MODEL,
        "messages": [
            {"role": "system", "content": "あなたは『サイテク』というアシスタントです。日本語で簡潔に答えてください。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": server_config.LLM_TEMPERATURE,
        "max_tokens": server_config.LLM_MAX_TOKENS,
        "top_p": server_config.LLM_TOP_P,
        "stream": True,
    }

    print("🌀 ストリーミング応答: ", end="", flush=True)
    full = []
    def stream_once(u: str) -> None:
        with requests.post(u, json=payload, headers=headers, stream=True, timeout=(5, server_config.REQUEST_TIMEOUT)) as r:
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if raw is None or raw == "":
                    continue
                line = raw.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if piece is None:
                    piece = choices[0].get("text")
                if not piece:
                    continue
                print(piece, end="", flush=True)
                full.append(piece)

    tried_alt = False
    try:
        stream_once(url)
    except requests.HTTPError as http_err:
        status = getattr(http_err.response, 'status_code', None)
        # 404/405などの場合、/v1 の有無を切り替えて再試行
        if status in (404, 405):
            tried_alt = True
            base2 = base
            if base2.rstrip('/').endswith('/v1'):
                base2 = base2.rstrip('/').rsplit('/v1', 1)[0]
            else:
                base2 = base2.rstrip('/') + '/v1'
            alt_url = build_url(base2, "/chat/completions")
            logger.warning(f"Chat API 404/405: {status}. 別パスで再試行: {alt_url}")
            try:
                stream_once(alt_url)
            except requests.HTTPError as http_err2:
                status2 = getattr(http_err2.response, 'status_code', None)
                # まだダメなら /responses も試す
                alt2 = build_url(base, "/responses")
                logger.warning(f"Chat API 再試行失敗: {status2}. 代替パスで再試行: {alt2}")
                stream_once(alt2)
        else:
            raise
    finally:
        print()
    reply = "".join(full).strip()
    logger.info(f"ストリーミング応答 終了 len={len(reply)}")
    return reply

# ========== サーバー通信 ==========
from requests.adapters import HTTPAdapter
try:
    from urllib3.util.retry import Retry  # type: ignore
except Exception:
    Retry = None  # 型: 無しでもセッション生成を続行

_http_session: Optional[requests.Session] = None

def _session_with_retries() -> requests.Session:
    global _http_session
    if _http_session is not None:
        return _http_session
    s = requests.Session()
    if Retry is not None:
        retry = Retry(
            total=server_config.HTTP_RETRIES,
            connect=server_config.HTTP_RETRIES,
            read=server_config.HTTP_RETRIES,
            status=server_config.HTTP_RETRIES,
            backoff_factor=server_config.RETRY_BACKOFF,
            status_forcelist=(429, 502, 503, 504),
            allowed_methods=frozenset(["POST"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
    s.headers.update({"User-Agent": "WakeSaitekuClient/1.0"})
    _http_session = s
    return s
def _toggle_inference_path(url: str) -> str:
    """Toggle between legacy '/inference' and '/v1/audio/inference' on the same host."""
    try:
        from urllib.parse import urlparse, urlunparse
        u = urlparse(url)
        path = u.path.rstrip('/')
        if path.endswith('/v1/audio/inference'):
            path = path[: -len('/v1/audio/inference')] + '/inference'
        elif path.endswith('/inference'):
            path = path[: -len('/inference')] + '/v1/audio/inference'
        else:
            # If ambiguous, prefer v1
            if not path.endswith('/v1'):
                path = path + '/v1/audio/inference'
            else:
                path = path + '/audio/inference'
        return urlunparse((u.scheme, u.netloc, path, "", "", ""))
    except Exception:
        # Fallback string manipulation
        if '/v1/audio/inference' in url:
            return url.replace('/v1/audio/inference', '/inference')
        if url.endswith('/inference'):
            return url[:-len('/inference')] + '/v1/audio/inference'
        return url.rstrip('/') + '/v1/audio/inference'


def _iter_candidate_urls() -> List[str]:
    urls: List[str] = []
    mode = server_config.SERVER_API_MODE
    primary = server_config.REMOTE_URL.strip()
    alt_env = server_config.ALT_URL.strip()
    if primary:
        urls.append(primary)
        if mode == 'auto':
            urls.append(_toggle_inference_path(primary))
    if alt_env:
        if alt_env not in urls:
            urls.append(alt_env)
        if mode == 'auto':
            toggled = _toggle_inference_path(alt_env)
            if toggled not in urls:
                urls.append(toggled)
    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for u in urls:
        if u and u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


def _to_stream_url(url: str) -> str:
    """/inference または /v1/audio/inference を /stream 付きへ変換（既に /stream ならそのまま）。"""
    u = url.rstrip('/')
    return u if u.endswith('/stream') else (u + '/stream')


def _to_non_stream_url(url: str) -> str:
    """/stream を含む推論URLを非ストリーミング版に変換。"""
    u = url.rstrip('/')
    if u.endswith('/stream'):
        u = u[: -len('/stream')]
    return u


def _call_non_streaming(session: requests.Session, file_path: str, interaction_id: str, base_url: str) -> Optional[dict]:
    url = _to_non_stream_url(base_url)
    try:
        with open(file_path, "rb") as f:
            files = {"file": ("utterance.wav", f, "audio/wav")}
            response = session.post(
                url,
                files=files,
                headers={
                    "X-Interaction-ID": interaction_id,
                    "User-Agent": "WakeSaitekuClient/1.0",
                },
                timeout=(server_config.CONNECT_TIMEOUT, server_config.READ_TIMEOUT)
            )
        response.raise_for_status()
        logger.info(f"非ストリーミング応答受信成功 url={url}")
        return response.json()
    except Exception as e:
        logger.warning(f"非ストリーミングフォールバックに失敗: {e}")
        return None


def send_to_server(audio_data: np.ndarray, interaction_id: str, stream: Optional[bool] = None) -> Tuple[bool, Optional[dict]]:
    """
    音声データをサーバーに送信
    
    Returns:
        (成功フラグ, レスポンスデータ)
    """
    candidates = _iter_candidate_urls()
    # 自動判定: REMOTE_URLが/stream で終わる or AUDIO_SSE=1 ならSSE
    use_stream = server_config.AUDIO_SSE or any(u.rstrip('/').endswith('/stream') for u in candidates)
    if stream is not None:
        use_stream = bool(stream)
    # ログ表示用の代表URL
    log_url = _to_stream_url(candidates[0]) if use_stream else candidates[0]
    logger.info(f"サーバーに送信中: {log_url} (候補={len(candidates)})")
    
    # 一時WAVファイル作成
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        save_wav(audio_data, tmp_path)
        
        last_err: Optional[Exception] = None
        sess = _session_with_retries()
        for i, base_url in enumerate(candidates):
            url = _to_stream_url(base_url) if use_stream else base_url
            try:
                if not use_stream:
                    with open(tmp_path, "rb") as f:
                        files = {"file": ("utterance.wav", f, "audio/wav")}
                        response = sess.post(
                            url,
                            files=files,
                            headers={
                                "X-Interaction-ID": interaction_id,
                                "User-Agent": "WakeSaitekuClient/1.0"
                            },
                            timeout=(server_config.CONNECT_TIMEOUT, server_config.READ_TIMEOUT)
                        )
                    response.raise_for_status()
                    data = response.json()
                    logger.info(f"サーバー応答受信成功 url={url}")
                    return True, data
                else:
                    # SSE ストリーミング受信
                    transcript = ""
                    reply_chunks: List[str] = []
                    # 逐次TTS用バッファ
                    seg_buf: List[str] = []
                    seg_min = int(os.getenv("WAKE_TTS_MIN_CHARS", "24"))
                    tts_worker = _TTSWorker(audio_processor) if (server_config.ENABLE_TTS_PLAYBACK and server_config.TTS_STREAMING) else None
                    timings: dict = {}
                    print("🌀 ストリーミング応答: ", end="", flush=True)
                    with open(tmp_path, "rb") as f:
                        files = {"file": ("utterance.wav", f, "audio/wav")}
                        with sess.post(
                            url,
                            files=files,
                            headers={
                                "X-Interaction-ID": interaction_id,
                                "Accept": "text/event-stream",
                                "Cache-Control": "no-cache",
                                "User-Agent": "WakeSaitekuClient/1.0",
                            },
                            stream=True,
                            timeout=(max(5.0, server_config.CONNECT_TIMEOUT), server_config.READ_TIMEOUT + 60.0),
                        ) as r:
                            r.raise_for_status()
                            for raw in r.iter_lines(decode_unicode=True):
                                if not raw:
                                    continue
                                line = raw.strip()
                                # 任意: SSE 生ログ（デバッグ用）
                                try:
                                    if os.getenv("WAKE_DEBUG_SSE", "").lower() in {"1", "true", "yes", "on"}:
                                        logger.info(f"[SSE] {line}")
                                except Exception:
                                    pass
                                if not line.startswith("data:"):
                                    continue
                                payload = line[5:].strip()
                                if payload == "[DONE]":
                                    break
                                try:
                                    obj = json.loads(payload)
                                except Exception:
                                    continue
                                # transcript（最初のイベント）
                                if isinstance(obj.get("transcript"), str):
                                    transcript = obj.get("transcript") or transcript
                                    continue
                                # timings（任意）
                                if obj.get("object") == "audio.timings":
                                    t = obj.get("timings") or {}
                                    if isinstance(t, dict):
                                        timings.update(t)
                                    continue
                                # OpenAI 互換 choices[].delta.content
                                choices = obj.get("choices") or []
                                if choices:
                                    delta = choices[0].get("delta") or {}
                                    piece = delta.get("content") or delta.get("output_text") or ""
                                    if piece:
                                        print(piece, end="", flush=True)
                                        reply_chunks.append(piece)
                                        if tts_worker is not None:
                                            seg_buf.append(piece)
                                            joined = "".join(seg_buf)
                                            if any(p in joined for p in ["。","．","!","！","?","？","\n"]) or len(joined) >= seg_min:
                                                tts_worker.enqueue(joined)
                                                seg_buf.clear()
                                    continue
                                # その他の簡易形式
                                if isinstance(obj.get("delta"), dict):
                                    piece = obj.get("delta", {}).get("content") or obj.get("delta", {}).get("output_text")
                                    if piece:
                                        print(piece, end="", flush=True)
                                        reply_chunks.append(piece)
                                        if tts_worker is not None:
                                            seg_buf.append(piece)
                                            joined = "".join(seg_buf)
                                            if any(p in joined for p in ["。","．","!","！","?","？","\n"]) or len(joined) >= seg_min:
                                                tts_worker.enqueue(joined)
                                                seg_buf.clear()
                                        continue
                                # 最終テキストが 'final' 等で来る場合
                                found_piece = False
                                for key in ("final", "output_text", "content", "text", "reply"):
                                    piece = obj.get(key)
                                    if isinstance(piece, str) and piece:
                                        print(piece, end="", flush=True)
                                        reply_chunks.append(piece)
                                        if tts_worker is not None:
                                            tts_worker.enqueue(piece)
                                        found_piece = True
                                        break
                                if not found_piece:
                                    extras = _extract_stream_reply_segments(obj)
                                    if extras:
                                        for piece in extras:
                                            print(piece, end="", flush=True)
                                            reply_chunks.append(piece)
                                            if tts_worker is not None:
                                                tts_worker.enqueue(piece)
                    print()
                    # 残りがあればTTSへ
                    if (server_config.ENABLE_TTS_PLAYBACK and server_config.TTS_STREAMING) and seg_buf:
                        try:
                            tts_worker.enqueue("".join(seg_buf))
                        except Exception:
                            pass
                    if tts_worker is not None:
                        try:
                            tts_worker.stop()
                        except Exception:
                            pass
                    logger.info(f"サーバー応答受信成功 url={url}")
                    result = {
                        "interaction_id": interaction_id,
                        "transcript": transcript,
                        "reply": "".join(reply_chunks).strip(),
                        "timings": timings or None,
                        "stream": True,
                    }
                    if server_config.STREAM_FALLBACK_ENABLED and not result["reply"]:
                        logger.info("SSE応答が空のため非ストリーミングAPIにフォールバックします。")
                        fallback = _call_non_streaming(sess, tmp_path, interaction_id, base_url)
                        if fallback:
                            return True, fallback
                    return True, result
            except requests.exceptions.Timeout as e:
                logger.warning(f"サーバータイムアウト url={url}")
                last_err = e
                continue
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"サーバー接続エラー url={url}")
                last_err = e
                # 候補URL切替前に短いバックオフ（サーバー起動直後の安定待ち）
                try:
                    backoff = min(2.0, 0.4 + i * 0.6) + random.uniform(0, 0.2)
                    time.sleep(backoff)
                except Exception:
                    pass
                continue
            except requests.HTTPError as e:
                status = getattr(getattr(e, 'response', None), 'status_code', None)
                logger.warning(f"HTTPエラー url={url} status={status}")
                last_err = e
                # try next candidate on 404/405/422 etc.
                continue
            except Exception as e:
                logger.warning(f"サーバー通信例外 url={url} err={e}")
                last_err = e
                continue
        # All candidates failed
        if last_err:
            logger.error(f"全URL失敗: {last_err}")
        return False, None
        
    except Exception as e:
        logger.error(f"サーバー通信エラー: {e}")
        return False, None
        
    finally:
        # 一時ファイル削除
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception as e:
                logger.warning(f"一時ファイル削除エラー: {e}")


# ========== サーバーTTS再生（任意） ==========
def _derive_tts_url_from_inference(inf_url: str) -> str:
    try:
        from urllib.parse import urlparse, urlunparse
        u = urlparse(inf_url)
        # /inference を /tts に
        path = u.path
        if path.endswith("/inference"):
            path = path[: -len("/inference")] + "/tts"
        else:
            if not path.endswith("/"):
                path = path + "/"
            path = path + "tts"
        return urlunparse((u.scheme, u.netloc, path, "", "", ""))
    except Exception:
        return inf_url.replace("/inference", "/tts")


def speak_via_server_tts(text: str, audio_processor: AudioProcessor) -> None:
    if not text:
        return
    _ensure_local_tts_server()
    # 出力デバイス未設定の場合は、TTS再生をスキップして安定性を優先
    try:
        dflt = sd.default.device
        out_idx_default = dflt[1] if isinstance(dflt, (tuple, list)) or hasattr(dflt, "__getitem__") else None
    except Exception:
        out_idx_default = None
    out_idx, dev_out = _resolve_output_device(out_idx_default)
    if out_idx in (None, -1):
        logger.info("TTS出力をスキップ（出力デバイス未設定）")
        return
    tts_url = server_config.SERVER_TTS_URL.strip() or _derive_tts_url_from_inference(server_config.REMOTE_URL)
    try:
        logger.info(f"サーバーTTS要求: {tts_url}")
        payload = {"text": text}
        if server_config.TTS_ENGINE:
            payload["engine"] = server_config.TTS_ENGINE
        if server_config.TTS_VOICE:
            payload["voice"] = server_config.TTS_VOICE
        if server_config.TTS_SPEED:
            payload["speed"] = float(server_config.TTS_SPEED)
        if server_config.TTS_SAMPLE_RATE:
            payload["sample_rate"] = int(server_config.TTS_SAMPLE_RATE)
        r = requests.post(tts_url, json=payload, timeout=server_config.REQUEST_TIMEOUT)
        if r.status_code != 200 or (r.headers.get("Content-Type") or "").split(";")[0] != "audio/wav":
            logger.warning(f"サーバーTTS失敗 status={r.status_code} content-type={r.headers.get('Content-Type')}")
            return
        data = r.content
        # WAVをデコード
        import io as _io
        import wave as _wave

        with _wave.open(_io.BytesIO(data), "rb") as wf:
            ch = wf.getnchannels()
            sr = wf.getframerate()
            sw = wf.getsampwidth()
            n = wf.getnframes()
            raw = wf.readframes(n)
        if sw != 2:
            logger.warning(f"TTS WAV サンプル幅が想定外: {sw}bytes")
            return
        # int16 -> float32 [-1,1]
        pcm = np.frombuffer(raw, dtype=np.int16)
        if ch > 1:
            pcm = pcm.reshape(-1, ch).mean(axis=1).astype(np.int16, copy=False)
        audio = (pcm.astype(np.float32) / 32767.0).astype(np.float32, copy=False)
        audio, sr = _prepare_playback_buffer(audio, sr, dev_out)
        frames = audio.shape[0] if audio.ndim > 1 else audio.size
        dur = frames / float(sr or audio_config.SAMPLE_RATE)
        # 再生音の回り込み抑制
        audio_processor.squelch(dur + 0.2)
        play_buf = audio
        device_arg = out_idx if out_idx not in (None, -1) else None
        try:
            sd.play(play_buf, samplerate=sr, blocking=False, device=device_arg)
        except Exception as e:
            logger.warning(f"サーバーTTS再生エラー（フォールバック前）: {e}")
            fb_buf, fb_sr = _fallback_playback_buffer(play_buf, sr)
            try:
                sd.play(fb_buf, samplerate=fb_sr, blocking=False, device=device_arg)
                sr = fb_sr
                play_buf = fb_buf
            except Exception as e2:
                logger.warning(f"サーバーTTS再生フォールバック失敗: {e2}")
                return
        logger.info(f"サーバーTTS再生 len={frames} sr={sr} dur={dur:.2f}s device={device_arg}")
    except Exception as e:
        logger.warning(f"サーバーTTS再生エラー: {e}")

# ========== メイン処理 ==========
def main():
    """メインループ"""
    # デバイス確定（TTSテストより前に！）
    print("⚙️  オーディオデバイスを選択中...")
    # 環境変数で強制指定があれば優先（名前/部分一致 または index）
    def _force_index(val: Optional[str], kind: str) -> Optional[int]:
        if not val:
            return None
        s = str(val).strip()
        if s == "":
            return None
        if s.isdigit():
            try:
                i = int(s)
                info = sd.query_devices(i)
                if (kind == 'input' and info.get('max_input_channels', 0) > 0) or \
                   (kind == 'output' and info.get('max_output_channels', 0) > 0):
                    return i
            except Exception:
                return None
        # 部分一致検索
        try:
            for i, d in enumerate(sd.query_devices()):
                nm = str(d.get('name',''))
                if s.lower() in nm.lower():
                    if (kind == 'input' and d.get('max_input_channels', 0) > 0) or \
                       (kind == 'output' and d.get('max_output_channels', 0) > 0):
                        return i
        except Exception:
            pass
        return None

    env_in = os.getenv('WAKE_IN_DEV') or os.getenv('AUDIO_INPUT_DEVICE')
    env_out = os.getenv('WAKE_OUT_DEV') or os.getenv('AUDIO_OUTPUT_DEVICE')
    forced_in = _force_index(env_in, 'input')
    forced_out = _force_index(env_out, 'output')

    in_idx, out_idx = choose_devices(
        samplerate=audio_config.SAMPLE_RATE,
        blocklist=[s.strip().lower() for s in os.getenv("WAKE_AUDIO_BLOCKLIST","").split(",") if s.strip()],
        allow_interactive=os.getenv("WAKE_AUDIO_SELECT","0").lower() in {"1","true","yes","on"},
        remember=True,
        strict_health=True,
        require_pair_for_bt=None,  # macOSではFalse, その他ではTrueに自動設定
    )
    # 強制指定があれば上書き
    if forced_in is not None:
        in_idx = forced_in
    if forced_out is not None:
        out_idx = forced_out
    # duplex 強制時は出力未指定でも in と合わせる
    if (os.getenv("WAKE_FORCE_DUPLEX", "0").lower() in {"1","true","yes","on"}) and (out_idx is None):
        out_idx = in_idx
    if in_idx is None:
        logger.error("利用可能な入力デバイスが見つかりませんでした。")
        sys.exit(1)

    # 出力未設定時は -1 ではなく None を指定
    sd.default.device = (in_idx, out_idx if out_idx is not None else None)
    sd.default.samplerate = audio_config.SAMPLE_RATE
    sd.default.channels = (audio_config.CHANNELS, 0 if (out_idx is None) else 1)
    
    try:
        dev_in_info = sd.query_devices(in_idx)
        dev_out_info = sd.query_devices(out_idx) if out_idx is not None else None
        in_name = dev_in_info['name'] if dev_in_info else 'N/A'
        out_name = dev_out_info['name'] if dev_out_info else 'N/A'
        # HFP（hands-free/headset）入力時は、出力がA2DP(2ch)などに分断されていれば同一デバイスに強制統一
        name_l = (in_name or '').lower()
        is_bt_hfp = any(k in name_l for k in ("hands-free", "headset", "hfp", "shokz", "opencomm", "airpods", "bluetooth"))
        out_ch = int(dev_out_info.get('max_output_channels', 0)) if dev_out_info else 0
        force_duplex = os.getenv("WAKE_FORCE_DUPLEX", "0").lower() in {"1", "true", "yes", "on"}
        if is_bt_hfp and (force_duplex or out_idx is None or out_ch != 1):
            out_idx = in_idx
            sd.default.device = (in_idx, out_idx)
            sd.default.channels = (audio_config.CHANNELS, 1)
            try:
                dev_out_info = sd.query_devices(out_idx)
                out_name = dev_out_info['name'] if dev_out_info else out_name
            except Exception:
                pass
        logger.info(f"🎤 Input='{in_name}'  🔊 Output='{out_name}'")
    except Exception as e:
        logger.warning(f"選択デバイス情報の取得に失敗: {e}")

    print("\n" + "="*50)
    print("Wake Saiteku クライアント")
    print("="*50)
    print(f"🎙  Wake Word待機中: 「もしもしサイテク」")
    print(f"📡 サーバー: {server_config.REMOTE_URL}")
    print(f"🔧 オフラインモード: {'有効' if server_config.LOCAL_STT_ENABLED else '無効'}")
    logger.info(
        f"設定: REMOTE_URL={server_config.REMOTE_URL}, LOCAL_STT_ENABLED={server_config.LOCAL_STT_ENABLED}, "
        f"LOCAL_LLM_URL={server_config.LOCAL_LLM_URL}, REQUEST_TIMEOUT={server_config.REQUEST_TIMEOUT}s, "
        f"PREFER_CHAT_API={server_config.PREFER_CHAT_API}, CHAT_API_BASE_URL={server_config.CHAT_API_BASE_URL or '-'}"
    )
    print("="*50 + "\n")

    # KokoroなどのローカルTTSを自動起動
    _ensure_local_tts_server()

    # Kokoro TTS疎通テスト（再生成功までリトライ）
    if server_config.ENABLE_TTS_PLAYBACK:
        test_text = os.getenv("WAKE_TTS_TEST_TEXT", "Wake Saitekuクライアント起動完了。TTSが正常に動作しています。")
        max_attempts = int(os.getenv("WAKE_TTS_TEST_RETRIES", "3"))
        for attempt in range(1, max_attempts + 1):
            ok = _run_tts_startup_check(test_text, attempt, max_attempts)
            if ok:
                break
        else:
            logger.warning("TTS疎通テストが全て失敗しました。WAKE_TTS_TEST_RETRIESで再試行回数を調整できます。")

    # オーディオプロセッサ初期化
    audio_processor = AudioProcessor()
    # Wake検出器（sherpa専用）
    try:
        wake_detector = SherpaWakeWordDetector()
        logger.info("Wakeバックエンド: SherpaWakeWordDetector")
    except Exception as e:
        logger.error(f"sherpaのWake初期化に失敗しました: {e}")
        logger.error(".envの WAKE_SHERPA_* 設定とモデルパスをご確認ください。")
        sys.exit(1)
    speech_recorder = SpeechRecorder(audio_processor)
    
    # オーディオストリーム開始
    audio_processor.start_stream()
    # ローカルSTTエンジン（sherpa専用）
    stt_backend = os.getenv("LOCAL_STT_BACKEND", "sherpa")
    try:
        local_stt = create_local_stt_engine(stt_backend, None)
        stt_name = local_stt.__class__.__name__
        logger.info(f"ローカルSTTバックエンド: {stt_name}")
    except Exception as e:
        logger.error(f"ローカルSTT初期化エラー: {e}")
        logger.error(".env の SHERPA_* 設定とモデルパスをご確認ください。")
        sys.exit(1)
    
    try:
        while True:
            # Wake Word検出フェーズ
            logger.info("Wake Word待機中...")
            wake_detector.reset()
            audio_processor.clear_queue()
            
            min_db = float(os.getenv("WAKE_MIN_DBFS", "-75"))
            while True:
                pcm = audio_processor.get_audio_frame(timeout=1.0)
                if pcm is None:
                    # しばらくフレームが来ない場合はストリームを復帰
                    stall_sec = float(os.getenv("WAKE_INPUT_STALL_SEC", "10"))
                    if time.time() - audio_processor.last_frame_time > stall_sec:
                        logger.warning(f"🎤 音声フレームが{int(stall_sec)}秒以上届いていません。ストリームを再起動します。")
                        try:
                            audio_processor.stop_stream()
                            time.sleep(0.2)
                            audio_processor.start_stream()
                        except Exception as e:
                            logger.error(f"オーディオストリーム再起動に失敗: {e}")
                            time.sleep(0.5)
                    continue
                
                # 1秒毎に入力レベルをINFOで出す（動作確認用）
                now = time.time()
                if now - audio_processor._last_level_log >= 1.0:
                    level = _dbfs(pcm.reshape(-1))
                    logger.info(f"🎚️ 入力レベル: {level:.1f} dBFS")
                    audio_processor._last_level_log = now

                # レベルが低すぎる場合はスキップ（誤検知抑制）
                if _dbfs(pcm.reshape(-1)) < min_db:
                    continue

                if wake_detector.process_audio(pcm.tobytes()):
                    # 発話処理ごとの相関IDを採番
                    interaction_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
                    print("\n✅ Wake Word検出!")
                    print(f"🆔 ID: {interaction_id}")
                    logger.info(f"[{interaction_id}] Wake Word検出")
                    # 通知音を鳴らし、回り込み対策でしばらく入力無視
                    beep_len = 0.18
                    audio_processor.squelch(beep_len + 0.15)
                    play_wake_sound()
                    print("📢 お話しください...")
                    break
            
            # 少し待ってキューをクリア（Wake Wordの残響を除去）
            time.sleep(0.2)
            audio_processor.clear_queue()
            
            # 音声録音フェーズ
            rec_start = time.perf_counter()
            audio_data = speech_recorder.record_speech()
            rec_dur = time.perf_counter() - rec_start
            
            if len(audio_data) == 0:
                print("⚠️  音声が検出されませんでした")
                continue
            
            print(f"📦 録音完了 ({len(audio_data) / audio_config.SAMPLE_RATE:.1f}秒, ⏱ {rec_dur:.2f}s)")
            logger.info(f"[{interaction_id}] 録音完了 duration={rec_dur:.2f}s samples={len(audio_data)}")
            
            # サーバー送信またはオフライン処理
            online_start = time.perf_counter()
            success, response = send_to_server(audio_data, interaction_id)
            online_dur = time.perf_counter() - online_start
            
            if success and response:
                # オンライン処理成功
                print("\n" + "-"*40)
                print(f"🆔 ID: {response.get('interaction_id', interaction_id)}")
                transcript_raw = response.get("transcript", "")
                transcript = dedupe_transcript(transcript_raw)
                print("📝 認識結果:", transcript)

                used_stream = False
                reply_text = response.get("reply", "")
                # 音声APIのSSEを使用した場合はその結果を優先
                if response.get("stream"):
                    used_stream = True
                else:
                    # Chat APIが使えるならストリーミング優先
                    if server_config.PREFER_CHAT_API and server_config.CHAT_API_BASE_URL and server_config.CHAT_API_KEY:
                        try:
                            reply_text = llm_streaming_chat_api(transcript, interaction_id)
                            used_stream = True
                        except Exception as e:
                            logger.warning(f"ストリーミングAPI利用に失敗: {e}. サーバー応答/ローカルへフォールバックします。")
                    # ストリーミング結果が空の場合はサーバー応答にフォールバック
                    if used_stream and (not reply_text) and response.get("reply"):
                        reply_text = response.get("reply")
                        used_stream = False
                # 最終出力
                print("🤖 応答:", reply_text)
                # 任意: サーバーTTSで応答を読み上げ
                if server_config.ENABLE_TTS_PLAYBACK and reply_text:
                    speak_via_server_tts(reply_text, audio_processor)
                # サーバー計測があれば表示
                timings = response.get("timings", {})
                if timings:
                    # Normalize to seconds whether server returns *_ms or seconds
                    if any(k.endswith('_ms') for k in timings.keys()):
                        asr = timings.get('asr_ms'); llm = timings.get('llm_ms'); total = timings.get('total_ms')
                        asr_s = (asr/1000.0) if isinstance(asr, (int, float)) else '-'
                        llm_s = (llm/1000.0) if isinstance(llm, (int, float)) else '-'
                        total_s = (total/1000.0) if isinstance(total, (int, float)) else '-'
                    else:
                        asr_s = timings.get('stt', '-')
                        llm_s = timings.get('llm', '-')
                        total_s = timings.get('total', '-')
                    if used_stream:
                        src = 'audio stream' if response.get('stream') else 'chat-api stream'
                        print(f"⏱ サーバー処理: STT {asr_s}s, LLM({src}) -, TOTAL {total_s}s")
                    else:
                        print(f"⏱ サーバー処理: STT {asr_s}s, LLM {llm_s}s, TOTAL {total_s}s")
                else:
                    print(f"⏱ オンライン往復: {online_dur:.2f}s")
                print("-"*40 + "\n")
                logger.info(f"[{interaction_id}] オンライン成功 roundtrip={online_dur:.2f}s transcript_len={len(transcript)} reply_len={len(reply_text)} stream_used={used_stream}")
            else:
                # オフラインフォールバック
                if server_config.LOCAL_STT_ENABLED:
                    print("🔄 オフラインモードで処理中...")
                    # オフラインSTT
                    t0 = time.perf_counter()
                    text = local_stt.transcribe(audio_data, audio_config.SAMPLE_RATE)
                    text = dedupe_transcript(text)
                    t1 = time.perf_counter()
                    
                    if text:
                        # 応答生成（Chat API優先。不可ならローカルLLM）
                        used_stream = False
                        t2 = t1
                        reply = ""
                        if server_config.PREFER_CHAT_API and server_config.CHAT_API_BASE_URL and server_config.CHAT_API_KEY:
                            try:
                                t2a = time.perf_counter()
                                reply = llm_streaming_chat_api(text, interaction_id)
                                t2 = time.perf_counter()
                                used_stream = True
                            except Exception as e:
                                logger.warning(f"[offline] ストリーミングAPI失敗: {e}. ローカルLLMにフォールバックします。")
                        if not used_stream:
                            reply = llm_local_reply(text, interaction_id)
                            t2 = time.perf_counter()

                        print("\n" + "-"*40)
                        print(f"🆔 ID: {interaction_id}")
                        print("📝 [オフライン] 認識結果:", text)
                        print("🤖 [オフライン] 応答:", reply)
                        # オフラインでもTTS再生（出力環境に合わせて自動整形）
                        if server_config.ENABLE_TTS_PLAYBACK and reply:
                            speak_via_server_tts(reply, audio_processor)
                        if used_stream and not reply:
                            # 空ならローカルにフォールバック
                            reply = llm_local_reply(text, interaction_id)
                            t2 = time.perf_counter()
                            used_stream = False
                        if used_stream:
                            print(f"⏱ [オフライン] STT {t1-t0:.2f}s, LLM(chat-api stream) ~{t2-t1:.2f}s, TOTAL {t2-t0:.2f}s")
                        else:
                            print(f"⏱ [オフライン] STT {t1-t0:.2f}s, LLM {t2-t1:.2f}s, TOTAL {t2-t0:.2f}s")
                        print("-"*40 + "\n")
                        logger.info(f"[{interaction_id}] オフライン成功 stt={t1-t0:.2f}s llm={(t2-t1):.2f}s total={(t2-t0):.2f}s text_len={len(text)} reply_len={len(reply)} stream={used_stream}")
                    else:
                        print("⚠️  音声を認識できませんでした")
                        logger.warning(f"[{interaction_id}] オフラインSTTでテキストなし")
                else:
                    print("❌ サーバーに接続できませんでした")
                    logger.error(f"[{interaction_id}] オンライン失敗・ローカル無効のため終了")
            
            print("🎙  再びWake Word待機中...")
            
    except KeyboardInterrupt:
        print("\n\n👋 終了します...")
    except Exception as e:
        logger.error(f"予期しないエラー: {e}", exc_info=True)
    finally:
        audio_processor.stop_stream()

if __name__ == "__main__":
    main()

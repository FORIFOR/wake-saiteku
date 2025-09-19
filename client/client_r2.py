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
import tempfile
import logging
import logging.handlers
import uuid
from pathlib import Path
from typing import Optional, Tuple, List, cast
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import sounddevice as sd
import webrtcvad

# デバイス安全選択（Bluetooth→USB→既定、対話選択可）
try:
    from audio_device_manager import choose_devices
except Exception:
    choose_devices = None

# --- プロジェクトルートを import path に追加（python client/client.py 実行対応）
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.wake_utils import recent_text_from_history, is_wake_in_text, squash_repeated_tokens, find_wake_match
from utils.text_utils import dedupe_transcript
from utils.stt_backends import create_local_stt_engine

# .env の自動読み込み（任意）
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(dotenv_path=Path(_PROJECT_ROOT) / ".env", override=False)
    logger = logging.getLogger(__name__)
    logger.debug(".env を読み込みました")
except Exception:
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
    END_SILENCE_MS: int = int(os.getenv("END_SILENCE_MS", "800"))
    MAX_RECORDING_MS: int = int(os.getenv("MAX_RECORDING_MS", "10000"))

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
    # LLM速度/品質調整
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "200"))
    LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", "0.9"))
    # 外部Chat API（OpenAI互換）設定（ストリーミング対応）
    PREFER_CHAT_API: bool = os.getenv("PREFER_CHAT_API", "true").lower() == "true"
    CHAT_API_BASE_URL: str = os.getenv("CHAT_API_BASE_URL", "")  # 例: http://localhost:8000/v1
    CHAT_API_KEY: str = os.getenv("CHAT_API_KEY", "")
    CHAT_API_MODEL: str = os.getenv("CHAT_API_MODEL", os.getenv("LLM_LOCAL_MODEL", "local-model"))
    # TTS（任意）
    ENABLE_TTS_PLAYBACK: bool = os.getenv("ENABLE_TTS_PLAYBACK", "false").lower() == "true"
    SERVER_TTS_URL: str = os.getenv("SERVER_TTS_URL", "")  # 未指定時は REMOTE_URL から導出
    TTS_ENGINE: str = os.getenv("TTS_ENGINE", "")  # kokoro_onnx|kokoro_pt|piper|openjtalk など
    TTS_VOICE: str = os.getenv("TTS_VOICE", "")   # 例: jf_alpha
    TTS_SAMPLE_RATE: int = int(os.getenv("TTS_SAMPLE_RATE", "24000"))

# 設定インスタンス
audio_config = AudioConfig()
wake_config = WakeConfig()
vad_config = VADConfig()
server_config = ServerConfig()

# ========== Voskモデルは使用しません（sherpa-ONNX専用） ==========

# ========== オーディオ処理 ==========
class AudioProcessor:
    def __init__(self):
        self.q = queue.Queue()
        self.vad = webrtcvad.Vad(vad_config.VAD_MODE)
        self.stream = None
        self._last_level_log = 0.0
        self.last_frame_time = time.time()
        self._squelch_until = 0.0  # 再生音の取り込み抑制用
        # watchdog / failure handling
        self._watchdog_th = None
        self._stop_flag = False
        self._noframe_threshold = float(os.getenv("AUDIO_NOFRAME_SEC", "2.5"))
        self._restart_backoff = 0.5
        self._fail_count = 0
        
    def audio_callback(self, indata, frames, time_info, status):
        """オーディオコールバック"""
        if status:
            logger.warning(f"オーディオステータス: {status}")
        now = time.time()
        # 再生音がマイクに回り込むのを抑制（半二重的に無視）
        if now < self._squelch_until:
            # 取り込まないが、最後のフレーム時刻は更新
            self.last_frame_time = now
            return
        self.q.put(indata.copy())
        self.last_frame_time = time.time()

    def squelch(self, duration_sec: float):
        """指定時間、入力フレームを無視（再生音の回り込み対策）"""
        self._squelch_until = max(self._squelch_until, time.time() + max(0.0, duration_sec))

    def _open_stream(self, in_idx: int) -> None:
        self.stream = sd.InputStream(
            channels=audio_config.CHANNELS,
            samplerate=audio_config.SAMPLE_RATE,
            dtype=audio_config.DTYPE,
            blocksize=audio_config.frame_length,
            device=in_idx,
            callback=self.audio_callback,
        )
        self.stream.start()
        di = sd.query_devices(in_idx)
        logger.info(
            f"🎤 オーディオストリーム開始 input='{di.get('name','unknown')}' "
            f"samplerate={audio_config.SAMPLE_RATE}Hz block={audio_config.frame_length}"
        )
        
    def start_stream(self):
        """オーディオストリーム開始（sd.default.device を前提）"""
        if self.stream is not None:
            return
        try:
            in_idx = cast(tuple, sd.default.device)[0]
        except Exception:
            in_idx = None
        if in_idx in (-1, None):
            raise RuntimeError("入力デバイスが未設定/無効です。")
        try:
            self._open_stream(in_idx)
        except Exception as e:
            # ダミーは明示許可時のみ
            if os.getenv("ALLOW_DUMMY_STREAM", "0").lower() in {"1","true","yes","on"}:
                logger.error(f"オーディオストリーム作成失敗: {e} → ダミーストリームで継続")
                self._start_dummy_stream()
            else:
                raise
        # watchdog 起動
        if self._watchdog_th is None:
            self._stop_flag = False
            self._watchdog_th = threading.Thread(target=self._watchdog_loop, daemon=True)
            self._watchdog_th.start()
    
    def stop_stream(self):
        """オーディオストリーム停止 (t-wada style robust implementation)"""
        self._stop_flag = True
        if self._watchdog_th:
            try:
                self._watchdog_th.join(timeout=0.5)
            except Exception:
                pass
            self._watchdog_th = None
        if self.stream:
            if self.stream == "dummy":
                # ダミーストリームの停止
                self.stream = None
                logger.info("🔇 ダミーストリーム停止")
            else:
                # 通常のストリーム停止
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception as e:
                    logger.warning(f"⚠️ ストリーム停止エラー: {e}")
                self.stream = None
                logger.info("🎤 オーディオストリーム停止")
    
    def get_audio_frame(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """オーディオフレームを取得"""
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def clear_queue(self):
        """キューをクリア"""
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
    
    # 旧スキャナは廃止（起動時に sd.default に安全デバイスを設定する方針）
    def _find_available_input_device(self) -> Optional[int]:
        try:
            return cast(tuple, sd.default.device)[0]
        except Exception:
            return None
    
    def _start_dummy_stream(self):
        """ダミーストリーム開始（明示許可時のみ）。本番では使わない。"""
        if os.getenv("ALLOW_DUMMY_STREAM", "0").lower() not in {"1","true","yes","on"}:
            raise RuntimeError("ダミーストリームは禁止されています（ALLOW_DUMMY_STREAM=1 で明示許可）。")
        logger.info("🔇 ダミーストリームモードで起動（マイク入力無し）")
        self.stream = "dummy"  # ダミーストリーム指標
        # 定期的にダミー音声データを生成するスレッドを開始
        import threading
        def dummy_audio_generator():
            while self.stream == "dummy":
                # 無音データを生成
                dummy_frame = np.zeros((audio_config.frame_length, audio_config.CHANNELS), dtype=np.int16)
                self.q.put(dummy_frame)
                time.sleep(audio_config.FRAME_DUR_MS / 1000.0)
        
        self.dummy_thread = threading.Thread(target=dummy_audio_generator, daemon=True)
        self.dummy_thread.start()
        logger.info("🤖 ダミー音声データ生成開始")

    # 監視: フレーム未到着（オーディオ検知失敗）を自動復旧
    def _watchdog_loop(self):
        backoff = self._restart_backoff
        while not self._stop_flag:
            time.sleep(0.5)
            if self.stream in (None, "dummy"):
                continue
            dt = time.time() - self.last_frame_time
            if dt > self._noframe_threshold:
                self._fail_count += 1
                logger.error(f"audio_detection_failure: no frames for {dt:.1f}s "
                             f"(fail#{self._fail_count}) → 再起動します")
                try:
                    # 再起動（デバイス index は sd.default.device を再利用）
                    try:
                        in_idx = cast(tuple, sd.default.device)[0]
                    except Exception:
                        in_idx = None
                    self.stop_stream()
                    time.sleep(0.2)
                    if in_idx in (-1, None):
                        raise RuntimeError("入力デバイスが未設定/喪失しました。")
                    self._open_stream(in_idx)
                    backoff = min(backoff * 2, 8.0)  # 上限 8s
                except Exception as e:
                    logger.error(f"再起動失敗: {e}")
                    time.sleep(backoff)

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
        sr = audio_config.SAMPLE_RATE
        # 880Hz -> 660Hz の二音（各90ms）
        dur1 = 0.09
        dur2 = 0.09
        t1 = np.linspace(0, dur1, int(sr * dur1), endpoint=False)
        t2 = np.linspace(0, dur2, int(sr * dur2), endpoint=False)
        tone1 = 0.2 * np.sin(2 * np.pi * 880 * t1).astype(np.float32)
        tone2 = 0.2 * np.sin(2 * np.pi * 660 * t2).astype(np.float32)
        signal = np.concatenate([_fade_in_out(tone1), _fade_in_out(tone2)])
        # 出力デバイス未設定時はスキップ（検出失敗の連鎖を防ぐ）
        try:
            out_idx = cast(tuple, sd.default.device)[1]
        except Exception:
            out_idx = None
        if out_idx in (-1, None):
            return
        sd.play(signal, samplerate=sr, blocking=True)
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


def send_to_server(audio_data: np.ndarray, interaction_id: str) -> Tuple[bool, Optional[dict]]:
    """
    音声データをサーバーに送信
    
    Returns:
        (成功フラグ, レスポンスデータ)
    """
    candidates = _iter_candidate_urls()
    logger.info(f"サーバーに送信中: {candidates[0]} (候補={len(candidates)})")
    
    # 一時WAVファイル作成
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        save_wav(audio_data, tmp_path)
        
        last_err: Optional[Exception] = None
        for url in candidates:
            try:
                with open(tmp_path, "rb") as f:
                    files = {"file": ("utterance.wav", f, "audio/wav")}
                    response = requests.post(
                        url,
                        files=files,
                        headers={
                            "X-Interaction-ID": interaction_id,
                            "User-Agent": "WakeSaitekuClient/1.0"
                        },
                        timeout=server_config.REQUEST_TIMEOUT
                    )
                response.raise_for_status()
                data = response.json()
                logger.info(f"サーバー応答受信成功 url={url}")
                return True, data
            except requests.exceptions.Timeout as e:
                logger.warning(f"サーバータイムアウト url={url}")
                last_err = e
                continue
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"サーバー接続エラー url={url}")
                last_err = e
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
    tts_url = server_config.SERVER_TTS_URL.strip() or _derive_tts_url_from_inference(server_config.REMOTE_URL)
    try:
        logger.info(f"サーバーTTS要求: {tts_url}")
        payload = {"text": text}
        if server_config.TTS_ENGINE:
            payload["engine"] = server_config.TTS_ENGINE
        if server_config.TTS_VOICE:
            payload["voice"] = server_config.TTS_VOICE
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
        dur = len(audio) / float(sr or audio_config.SAMPLE_RATE)
        # 再生音の回り込み抑制
        audio_processor.squelch(dur + 0.2)  # 半二重回り込み対策
        try:
            out_idx = cast(tuple, sd.default.device)[1]
        except Exception:
            out_idx = None
        if out_idx in (-1, None):
            logger.warning("出力デバイス未設定のためTTS再生をスキップ")
            return
        sd.play(audio, samplerate=sr, blocking=False)
        logger.info(f"サーバーTTS再生 len={len(audio)} sr={sr} dur={dur:.2f}s")
    except Exception as e:
        logger.warning(f"サーバーTTS再生エラー: {e}")

# ========== メイン処理 ==========
def main():
    """メインループ"""
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
    
    # === 起動時に安全な入出力デバイスを確定（Bluetooth→USB→既定、必要に応じて対話選択） ===
    try:
        allow_interactive = os.getenv("WAKE_AUDIO_SELECT", "0").lower() in {"1","true","yes","on"}
        extra_block = [s.strip().lower() for s in os.getenv("WAKE_AUDIO_BLOCKLIST","").split(",") if s.strip()]
        blocklist = ["opencomm2 by shokz", "shokz"] + extra_block
        if choose_devices:
            in_idx, out_idx = choose_devices(
                samplerate=audio_config.SAMPLE_RATE,
                blocklist=blocklist,
                allow_interactive=allow_interactive,
                remember=True,
            )
        else:
            in_idx = sd.default.device[0] if sd.default.device else None
            out_idx = sd.default.device[1] if sd.default.device else None
        if in_idx is None:
            if os.getenv("ALLOW_DUMMY_STREAM", "0").lower() in {"1","true","yes","on"}:
                logger.warning("入力デバイスが見つかりません。ダミーストリームモードで継続します。")
                # ダミー用に無効なデバイスIDを設定
                sd.default.device = (-1, out_idx if out_idx not in (-1, None) else -1)
                sd.default.samplerate = audio_config.SAMPLE_RATE
                sd.default.channels = (audio_config.CHANNELS, 1)
                do = sd.query_devices(out_idx) if out_idx not in (-1, None) else {"name": "なし"}
                logger.info(f"🎤 Input='ダミー（ALLOW_DUMMY_STREAM有効）'  🔊 Output='{do.get('name','?')}'")
                return  # ダミーモードで継続
            else:
                raise RuntimeError("利用可能な入力デバイスが見つかりません。")
        else:
            sd.default.device = (in_idx, out_idx if out_idx is not None else -1)
            sd.default.samplerate = audio_config.SAMPLE_RATE
            sd.default.channels = (audio_config.CHANNELS, 1)
            di = sd.query_devices(in_idx)
            do = sd.query_devices(out_idx) if out_idx not in (-1, None) else {"name": "なし"}
            logger.info(f"🎤 Input='{di.get('name','?')}'  🔊 Output='{do.get('name','?')}'")
    except Exception as e:
        logger.error(f"オーディオ入出力の初期化に失敗: {e}")
        raise

    # TTS疎通テスト（任意・出力がある時のみ）
    if server_config.ENABLE_TTS_PLAYBACK:
        print("🔊 TTS疎通テスト実行中...")
        try:
            test_text = "Wake Saitekuクライアント起動完了。TTSが正常に動作しています。"
            tts_url = server_config.SERVER_TTS_URL
            payload = {"text": test_text}
            if server_config.TTS_ENGINE:
                payload["engine"] = server_config.TTS_ENGINE
            if server_config.TTS_SAMPLE_RATE:
                payload["sample_rate"] = int(server_config.TTS_SAMPLE_RATE)
            
            r = requests.post(tts_url, json=payload, timeout=5)
            if r.status_code == 200 and (r.headers.get("Content-Type") or "").split(";")[0] == "audio/wav":
                print("✅ TTSサーバー接続成功")
                # 音声再生テスト
                import io as _io
                import wave as _wave
                with _wave.open(_io.BytesIO(r.content), "rb") as wf:
                    audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32767.0
                    try:
                        out_idx = cast(tuple, sd.default.device)[1]
                    except Exception:
                        out_idx = None
                    if out_idx not in (-1, None):
                        sd.play(audio, samplerate=wf.getframerate(), blocking=False)
                    logger.info("✅ TTS初期化テスト完了")
                    time.sleep(1)  # 音声再生待機
            else:
                logger.warning(f"⚠️ TTSテスト失敗: status={r.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ TTSテストエラー (続行可能): {e}")
    
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
    
    # オーディオストリーム開始（失敗時は例外で停止。ダミーは既定で使わない）
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
                    if time.time() - audio_processor.last_frame_time > 3.0:
                        logger.warning("🎤 音声フレームが3秒以上届いていません。ストリームを再起動します。")
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
                # Chat APIが使えるならストリーミング優先
                used_stream = False
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
                        print(f"⏱ サーバー処理: STT {asr_s}s, LLM(chat-api stream) -, TOTAL {total_s}s")
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

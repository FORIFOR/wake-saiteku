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
from typing import Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import sounddevice as sd
import webrtcvad
import vosk

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
    
    @property
    def frame_length(self) -> int:
        return int(self.SAMPLE_RATE * self.FRAME_DUR_MS / 1000)

@dataclass
class WakeConfig:
    WAKE_TIMEOUT_S: float = float(os.getenv("WAKE_TIMEOUT_S", "2.5"))
    WAKE_REQUIRE_BOTH: bool = os.getenv("WAKE_REQUIRE_BOTH", "true").lower() == "true"
    WAKE_WORDS: List[Tuple[str, str]] = None
    
    def __post_init__(self):
        if self.WAKE_WORDS is None:
            self.WAKE_WORDS = [
                ("もしもし", r"もしもし"),
                ("サイテク", r"(サイテク|さいてく|ｻｲﾃｸ|さいテク|サイテック|さいてっく|サイトク|さいとく)")
            ]

@dataclass
class VADConfig:
    VAD_MODE: int = 2  # 0-3, 3が最も厳しい
    MIN_UTTERANCE_MS: int = 400
    END_SILENCE_MS: int = 800
    MAX_RECORDING_MS: int = 10000

@dataclass
class ServerConfig:
    REMOTE_URL: str = os.getenv("SERVER_URL", "http://127.0.0.1:8000/inference")
    LOCAL_STT_ENABLED: bool = os.getenv("LOCAL_STT_ENABLED", "true").lower() == "true"
    LOCAL_LLM_URL: str = os.getenv("LLM_LOCAL_URL", "http://127.0.0.1:8081/v1/chat/completions")
    LOCAL_LLM_MODEL: str = os.getenv("LLM_LOCAL_MODEL", "local-model")
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))

# 設定インスタンス
audio_config = AudioConfig()
wake_config = WakeConfig()
vad_config = VADConfig()
server_config = ServerConfig()

# ========== Vosk初期化 ==========
def initialize_vosk(model_path: str = "wake-saiteku/models/ja") -> vosk.Model:
    """Voskモデルを初期化"""
    vosk.SetLogLevel(-1)  # ログレベルを抑制
    
    if not os.path.isdir(model_path):
        logger.error(f"Vosk日本語モデルが見つかりません: {model_path}")
        logger.info("セットアップスクリプトを実行してモデルをダウンロードしてください")
        sys.exit(1)
    
    logger.info(f"Voskモデルを読み込み中: {model_path}")
    try:
        model = vosk.Model(model_path)
        logger.info("Voskモデルの読み込み完了")
        return model
    except Exception as e:
        logger.error(f"Voskモデルの読み込みに失敗: {e}")
        sys.exit(1)

# モデル初期化
vosk_model = initialize_vosk()

# ========== オーディオ処理 ==========
class AudioProcessor:
    def __init__(self):
        self.q = queue.Queue()
        self.vad = webrtcvad.Vad(vad_config.VAD_MODE)
        self.stream = None
        self._last_level_log = 0.0
        self.last_frame_time = time.time()
        
    def audio_callback(self, indata, frames, time_info, status):
        """オーディオコールバック"""
        if status:
            logger.warning(f"オーディオステータス: {status}")
        self.q.put(indata.copy())
        self.last_frame_time = time.time()
        
    def start_stream(self):
        """オーディオストリーム開始"""
        if self.stream is None:
            self.stream = sd.InputStream(
                channels=audio_config.CHANNELS,
                samplerate=audio_config.SAMPLE_RATE,
                dtype=audio_config.DTYPE,
                blocksize=audio_config.frame_length,
                device=audio_config.INPUT_DEVICE,
                callback=self.audio_callback
            )
            self.stream.start()
            # デバイス情報を記録
            try:
                in_dev = None
                if audio_config.INPUT_DEVICE is not None:
                    in_dev = sd.query_devices(audio_config.INPUT_DEVICE)
                else:
                    default_in = sd.default.device[0]
                    in_dev = sd.query_devices(default_in)
                logger.info(f"オーディオストリーム開始 device='{in_dev.get('name','unknown')}' samplerate={audio_config.SAMPLE_RATE}Hz block={audio_config.frame_length}")
            except Exception:
                logger.info("オーディオストリーム開始")
    
    def stop_stream(self):
        """オーディオストリーム停止"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            logger.info("オーディオストリーム停止")
    
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

# ========== Wake Word検知 ==========
class WakeWordDetector:
    def __init__(self, model: vosk.Model):
        self.model = model
        self.recognizer = vosk.KaldiRecognizer(model, audio_config.SAMPLE_RATE)
        self._configure_grammar()
        self.text_history: List[Tuple[str, float]] = []
        
    def reset(self):
        """認識器をリセット"""
        self.recognizer = vosk.KaldiRecognizer(self.model, audio_config.SAMPLE_RATE)
        self._configure_grammar()
        self.text_history.clear()

    def _configure_grammar(self):
        """Wake Word専用の簡易文法を設定（誤検知抑制）"""
        use_grammar = os.getenv("WAKE_USE_GRAMMAR", "true").lower() == "true"
        if not use_grammar:
            return
        phrases = [
            "もしもし",
            "もしもし サイテク",
            "もしもし サイテック",
            "もしもし サイトク",
            "サイテク",
            "サイテック",
            "サイトク",
            "さいてく",
            "さいてっく",
            "さいとく",
        ]
        try:
            self.recognizer.SetGrammar(json.dumps(phrases, ensure_ascii=False))
            logger.info("Wake用簡易文法を適用しました")
        except Exception as e:
            logger.warning(f"Wake文法設定に失敗: {e}")
    
    def process_audio(self, pcm_data: bytes) -> bool:
        """オーディオを処理してWake Wordを検出"""
        current_time = time.time()
        
        # 音声認識
        if self.recognizer.AcceptWaveform(pcm_data):
            result = json.loads(self.recognizer.Result())
            text = result.get("text", "")
            if text:
                self.text_history.append((text, current_time))
                logger.debug(f"認識: {text}")
        else:
            # 部分結果も取得
            partial_result = json.loads(self.recognizer.PartialResult())
            partial_text = partial_result.get("partial", "")
            if partial_text:
                self.text_history.append((partial_text, current_time))
                logger.debug(f"部分認識: {partial_text}")
        
        # 古いエントリを削除
        cutoff_time = current_time - wake_config.WAKE_TIMEOUT_S - 0.5
        self.text_history = [(t, ts) for t, ts in self.text_history if ts >= cutoff_time]
        
        # Wake Word検出
        return self._check_wake_words()
    
    def _check_wake_words(self) -> bool:
        """Wake Wordが含まれているかチェック"""
        current_time = time.time()
        recent_text = "".join([
            text for text, timestamp in self.text_history
            if current_time - timestamp <= wake_config.WAKE_TIMEOUT_S
        ])
        
        # Wake Wordが含まれているかチェック（順序も考慮）
        if wake_config.WAKE_REQUIRE_BOTH:
            # 「もしもし」→「サイテク系」の順を優先
            ok = bool(re.search(r"もしもし.*(サイテク|さいてく|ｻｲﾃｸ|さいテク|サイテック|サイトク|さいとく)", recent_text))
        else:
            ok = any(re.search(pattern, recent_text) for _, pattern in wake_config.WAKE_WORDS)
        if ok:
            logger.info(f"Wake Word検出: {recent_text}")
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
        "temperature": 0.7,
        "max_tokens": 256
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

# ========== サーバー通信 ==========
def send_to_server(audio_data: np.ndarray, interaction_id: str) -> Tuple[bool, Optional[dict]]:
    """
    音声データをサーバーに送信
    
    Returns:
        (成功フラグ, レスポンスデータ)
    """
    logger.info(f"サーバーに送信中: {server_config.REMOTE_URL}")
    
    # 一時WAVファイル作成
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        save_wav(audio_data, tmp_path)
        
        with open(tmp_path, "rb") as f:
            files = {"file": ("utterance.wav", f, "audio/wav")}
            response = requests.post(
                server_config.REMOTE_URL,
                files=files,
                headers={
                    "X-Interaction-ID": interaction_id,
                    "User-Agent": "WakeSaitekuClient/1.0"
                },
                timeout=server_config.REQUEST_TIMEOUT
            )
        
        response.raise_for_status()
        data = response.json()
        
        logger.info("サーバー応答受信成功")
        return True, data
        
    except requests.exceptions.Timeout:
        logger.error("サーバータイムアウト")
        return False, None
        
    except requests.exceptions.ConnectionError:
        logger.error("サーバー接続エラー")
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

# ========== メイン処理 ==========
def main():
    """メインループ"""
    print("\n" + "="*50)
    print("Wake Saiteku クライアント")
    print("="*50)
    print(f"🎙  Wake Word待機中: 「もしもしサイテク」")
    print(f"📡 サーバー: {server_config.REMOTE_URL}")
    print(f"🔧 オフラインモード: {'有効' if server_config.LOCAL_STT_ENABLED else '無効'}")
    logger.info(f"設定: REMOTE_URL={server_config.REMOTE_URL}, LOCAL_STT_ENABLED={server_config.LOCAL_STT_ENABLED}, LOCAL_LLM_URL={server_config.LOCAL_LLM_URL}, REQUEST_TIMEOUT={server_config.REQUEST_TIMEOUT}s")
    print("="*50 + "\n")
    
    # オーディオプロセッサ初期化
    audio_processor = AudioProcessor()
    wake_detector = WakeWordDetector(vosk_model)
    speech_recorder = SpeechRecorder(audio_processor)
    
    # オーディオストリーム開始
    audio_processor.start_stream()
    
    try:
        while True:
            # Wake Word検出フェーズ
            logger.info("Wake Word待機中...")
            wake_detector.reset()
            audio_processor.clear_queue()
            
            min_db = float(os.getenv("WAKE_MIN_DBFS", "-55"))
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
                print("📝 認識結果:", response.get("transcript", ""))
                print("🤖 応答:", response.get("reply", ""))
                # サーバー計測があれば表示
                timings = response.get("timings", {})
                if timings:
                    print(f"⏱ サーバー処理: STT {timings.get('stt','-')}s, LLM {timings.get('llm','-')}s, TOTAL {timings.get('total','-')}s")
                else:
                    print(f"⏱ オンライン往復: {online_dur:.2f}s")
                print("-"*40 + "\n")
                logger.info(f"[{interaction_id}] オンライン成功 roundtrip={online_dur:.2f}s transcript_len={len(response.get('transcript',''))} reply_len={len(response.get('reply',''))}")
            else:
                # オフラインフォールバック
                if server_config.LOCAL_STT_ENABLED:
                    print("🔄 オフラインモードで処理中...")
                    # オフラインSTT
                    t0 = time.perf_counter()
                    text = stt_offline_vosk(audio_data)
                    t1 = time.perf_counter()
                    
                    if text:
                        # オフラインLLM
                        reply = llm_local_reply(text, interaction_id)
                        t2 = time.perf_counter()
                        
                        print("\n" + "-"*40)
                        print(f"🆔 ID: {interaction_id}")
                        print("📝 [オフライン] 認識結果:", text)
                        print("🤖 [オフライン] 応答:", reply)
                        print(f"⏱ [オフライン] STT {t1-t0:.2f}s, LLM {t2-t1:.2f}s, TOTAL {t2-t0:.2f}s")
                        print("-"*40 + "\n")
                        logger.info(f"[{interaction_id}] オフライン成功 stt={t1-t0:.2f}s llm={t2-t1:.2f}s total={t2-t0:.2f}s text_len={len(text)} reply_len={len(reply)}")
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

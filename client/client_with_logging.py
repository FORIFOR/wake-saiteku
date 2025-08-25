#!/usr/bin/env python3
"""
Wake Saiteku Client - 高度なログ機能付き
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import json
import queue
import threading
import wave
import re
import requests
import tempfile
import numpy as np
import sounddevice as sd
from typing import Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime

from utils.logger import (
    get_logger,
    AudioLogger,
    APILogger,
    PerformanceLogger,
    LogManager
)

# ロガー初期化
logger = get_logger("client")
audio_logger = get_logger("audio")
api_logger = get_logger("api")
perf_logger = get_logger("performance")
wake_logger = get_logger("wake_word")

# ========== 設定 ==========
@dataclass
class AudioConfig:
    SAMPLE_RATE: int = 16000
    FRAME_DUR_MS: int = 20
    CHANNELS: int = 1
    DTYPE: str = 'int16'
    
    @property
    def frame_length(self) -> int:
        return int(self.SAMPLE_RATE * self.FRAME_DUR_MS / 1000)

@dataclass
class WakeConfig:
    WAKE_TIMEOUT_S: float = 2.5
    WAKE_WORDS: List[Tuple[str, str]] = None
    
    def __post_init__(self):
        if self.WAKE_WORDS is None:
            self.WAKE_WORDS = [
                ("もしもし", r"もしもし"),
                ("サイテク", r"(サイテク|さいてく|ｻｲﾃｸ|さいテク)")
            ]

# 設定インスタンス
audio_config = AudioConfig()
wake_config = WakeConfig()

class AudioProcessor:
    """音声処理クラス（ログ機能強化版）"""
    
    def __init__(self):
        self.q = queue.Queue()
        self.stream = None
        self.total_frames = 0
        self.session_start = time.time()
        
        logger.info("AudioProcessor初期化")
        
    def audio_callback(self, indata, frames, time_info, status):
        """オーディオコールバック"""
        if status:
            logger.warning(f"オーディオステータス警告: {status}")
        
        self.q.put(indata.copy())
        self.total_frames += frames
        
        # 定期的に統計をログ
        if self.total_frames % (audio_config.SAMPLE_RATE * 10) == 0:  # 10秒ごと
            audio_logger.info(
                "オーディオ統計",
                total_frames=self.total_frames,
                queue_size=self.q.qsize(),
                session_duration=time.time() - self.session_start
            )
    
    def start_stream(self):
        """オーディオストリーム開始"""
        try:
            logger.info("オーディオストリーム開始中...")
            
            # デバイス情報をログ
            devices = sd.query_devices()
            default_device = sd.query_devices(kind='input')
            
            logger.info(
                "オーディオデバイス情報",
                default_device=default_device['name'],
                sample_rate=default_device['default_samplerate'],
                channels=default_device['max_input_channels']
            )
            
            self.stream = sd.InputStream(
                channels=audio_config.CHANNELS,
                samplerate=audio_config.SAMPLE_RATE,
                dtype=audio_config.DTYPE,
                blocksize=audio_config.frame_length,
                callback=self.audio_callback
            )
            self.stream.start()
            
            logger.info("オーディオストリーム開始成功")
            
        except Exception as e:
            logger.error(
                "オーディオストリーム開始失敗",
                error=str(e),
                exc_info=True
            )
            raise
    
    def stop_stream(self):
        """オーディオストリーム停止"""
        if self.stream:
            logger.info("オーディオストリーム停止中...")
            self.stream.stop()
            self.stream.close()
            self.stream = None
            
            # セッション統計
            session_duration = time.time() - self.session_start
            logger.info(
                "オーディオセッション終了",
                total_frames=self.total_frames,
                duration_seconds=round(session_duration, 2),
                average_fps=round(self.total_frames / session_duration, 2) if session_duration > 0 else 0
            )
    
    def get_audio_frame(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """オーディオフレームを取得"""
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            logger.debug("オーディオキュー空")
            return None
    
    def clear_queue(self):
        """キューをクリア"""
        cleared = 0
        while not self.q.empty():
            try:
                self.q.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        
        if cleared > 0:
            logger.debug(f"オーディオキューをクリア: {cleared}フレーム")


class WakeWordDetector:
    """Wake Word検知クラス（ログ機能強化版）"""
    
    def __init__(self):
        self.text_history: List[Tuple[str, float]] = []
        self.detection_count = 0
        self.false_positive_count = 0
        
        wake_logger.info(
            "Wake Word検知器初期化",
            wake_words=wake_config.WAKE_WORDS
        )
    
    def reset(self):
        """認識器をリセット"""
        self.text_history.clear()
        wake_logger.debug("Wake Word検知器リセット")
    
    def process_audio(self, text: str) -> bool:
        """テキストからWake Wordを検出（シミュレーション）"""
        current_time = time.time()
        
        if text:
            self.text_history.append((text, current_time))
            wake_logger.debug(f"テキスト追加: {text}")
        
        # 古いエントリを削除
        cutoff_time = current_time - wake_config.WAKE_TIMEOUT_S - 0.5
        old_count = len(self.text_history)
        self.text_history = [(t, ts) for t, ts in self.text_history if ts >= cutoff_time]
        
        if old_count > len(self.text_history):
            wake_logger.debug(f"古いエントリ削除: {old_count - len(self.text_history)}件")
        
        # Wake Word検出
        detected = self._check_wake_words()
        
        if detected:
            self.detection_count += 1
            wake_logger.info(
                "Wake Word検出！",
                detection_count=self.detection_count,
                confidence=0.95  # シミュレーション値
            )
        
        return detected
    
    def _check_wake_words(self) -> bool:
        """Wake Wordが含まれているかチェック"""
        current_time = time.time()
        recent_text = "".join([
            text for text, timestamp in self.text_history
            if current_time - timestamp <= wake_config.WAKE_TIMEOUT_S
        ])
        
        # すべてのWake Wordが含まれているかチェック
        for word_name, pattern in wake_config.WAKE_WORDS:
            if not re.search(pattern, recent_text):
                return False
        
        return True


class SpeechRecorder:
    """音声録音クラス（ログ機能強化版）"""
    
    def __init__(self, audio_processor: AudioProcessor):
        self.audio_processor = audio_processor
        self.recording_count = 0
        
        logger.info("SpeechRecorder初期化")
    
    def record_speech(self, max_duration: float = 10.0) -> np.ndarray:
        """音声を録音"""
        self.recording_count += 1
        
        logger.info(
            "録音開始",
            recording_id=self.recording_count,
            max_duration=max_duration
        )
        
        perf_logger.start_timer("recording")
        
        speech_frames = []
        start_time = time.time()
        
        # シミュレーション: 3秒録音
        duration = 3.0
        samples = int(audio_config.SAMPLE_RATE * duration)
        
        # テスト音声生成（サイン波）
        t = np.linspace(0, duration, samples, False)
        audio_data = np.sin(440 * 2 * np.pi * t) * 0.3
        audio_data = (audio_data * 32767).astype(np.int16)
        
        recording_time = perf_logger.end_timer("recording")
        
        # 録音統計
        audio_logger.log_recording(
            duration=duration,
            samples=len(audio_data),
            sample_rate=audio_config.SAMPLE_RATE
        )
        
        logger.info(
            "録音完了",
            recording_id=self.recording_count,
            duration=duration,
            samples=len(audio_data),
            size_kb=round(len(audio_data) * 2 / 1024, 2)
        )
        
        return audio_data


def send_to_server(audio_data: np.ndarray, server_url: str = "http://localhost:8002/inference") -> Tuple[bool, Optional[dict]]:
    """音声データをサーバーに送信（ログ機能強化版）"""
    
    request_id = f"req_{int(time.time() * 1000)}"
    LogManager.setup_request_id(request_id)
    
    logger.info(
        "サーバー送信準備",
        server_url=server_url,
        audio_size=len(audio_data) * 2
    )
    
    # WAVファイル作成
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        with wave.open(tmp_file.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(audio_config.SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())
        tmp_path = tmp_file.name
    
    try:
        # リクエスト開始
        api_logger.log_request(
            method="POST",
            url=server_url,
            body={"file_size": os.path.getsize(tmp_path)}
        )
        
        perf_logger.start_timer("api_request")
        
        with open(tmp_path, "rb") as f:
            files = {"file": ("utterance.wav", f, "audio/wav")}
            response = requests.post(
                server_url,
                files=files,
                timeout=30
            )
        
        request_time = perf_logger.end_timer("api_request")
        
        # レスポンスログ
        api_logger.log_response(
            status_code=response.status_code,
            response_time=request_time,
            body=response.text[:200]
        )
        
        if response.status_code == 200:
            data = response.json()
            
            logger.info(
                "サーバー応答成功",
                transcript=data.get("transcript", ""),
                reply_length=len(data.get("reply", "")),
                processing_time=data.get("processing_time", 0)
            )
            
            return True, data
        else:
            logger.warning(
                "サーバーエラー応答",
                status_code=response.status_code,
                response=response.text[:200]
            )
            return False, None
            
    except requests.exceptions.Timeout:
        api_logger.log_error("Timeout", "リクエストタイムアウト")
        return False, None
        
    except requests.exceptions.ConnectionError as e:
        api_logger.log_error("ConnectionError", str(e))
        return False, None
        
    except Exception as e:
        logger.error(
            "予期しないエラー",
            error=str(e),
            exc_info=True
        )
        return False, None
        
    finally:
        # クリーンアップ
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            logger.debug(f"一時ファイル削除: {tmp_path}")
        
        LogManager.clear_request_id()


def main():
    """メインループ（ログ機能強化版）"""
    
    # 起動ログ
    logger.info(
        "="*50 + "\n" +
        "Wake Saiteku クライアント起動\n" +
        "="*50
    )
    
    logger.info(
        "システム情報",
        platform=sys.platform,
        python_version=sys.version.split()[0],
        pid=os.getpid()
    )
    
    # メモリ使用量記録
    perf_logger.log_memory_usage()
    
    # コンポーネント初期化
    audio_processor = AudioProcessor()
    wake_detector = WakeWordDetector()
    speech_recorder = SpeechRecorder(audio_processor)
    
    session_start = time.time()
    cycle_count = 0
    
    try:
        # メインループ
        while cycle_count < 3:  # デモ用: 3サイクルで終了
            cycle_count += 1
            
            logger.info(f"\n{'='*40}\nサイクル {cycle_count} 開始\n{'='*40}")
            
            # Wake Wordシミュレーション
            time.sleep(1)
            wake_detected = wake_detector.process_audio("もしもしサイテク")
            
            if wake_detected:
                print("\n✅ Wake Word検出!")
                
                # 録音
                audio_data = speech_recorder.record_speech()
                
                # サーバー送信
                success, response = send_to_server(audio_data)
                
                if success and response:
                    print(f"📝 認識結果: {response.get('transcript', '')}")
                    print(f"🤖 応答: {response.get('reply', '')}")
                else:
                    print("❌ サーバー通信エラー")
            
            # パフォーマンス記録
            if cycle_count % 5 == 0:
                perf_logger.log_memory_usage()
                perf_logger.log_cpu_usage()
            
            time.sleep(2)  # 次のサイクルまで待機
    
    except KeyboardInterrupt:
        logger.info("ユーザーによる中断")
    except Exception as e:
        logger.error(
            "予期しないエラー",
            error=str(e),
            exc_info=True
        )
    finally:
        # セッション統計
        session_duration = time.time() - session_start
        
        logger.info(
            "セッション終了",
            duration_seconds=round(session_duration, 2),
            total_cycles=cycle_count,
            wake_detections=wake_detector.detection_count
        )
        
        audio_processor.stop_stream()
        
        logger.info("クライアント終了")


if __name__ == "__main__":
    # ログディレクトリ作成
    os.makedirs("logs", exist_ok=True)
    
    main()
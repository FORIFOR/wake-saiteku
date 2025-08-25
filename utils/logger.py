#!/usr/bin/env python3
"""
統合ログシステム
構造化ログ、パフォーマンス計測、エラー追跡を提供
"""

import logging
import logging.config
import json
import time
import traceback
import os
import yaml
from datetime import datetime
from contextlib import contextmanager
from functools import wraps
from typing import Any, Dict, Optional, Callable
import sys

# ログディレクトリ作成
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# ログ設定ファイルのパス
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "logging.yaml")

class StructuredLogger:
    """構造化ログを提供するカスタムロガー"""
    
    def __init__(self, name: str, config_path: Optional[str] = None):
        """
        Args:
            name: ロガー名
            config_path: logging.yamlのパス
        """
        self.name = name
        self.config_path = config_path or CONFIG_PATH
        self._setup_logging()
        self.logger = logging.getLogger(name)
        self.context = {}  # コンテキスト情報
        
    def _setup_logging(self):
        """ログ設定を読み込み"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                    logging.config.dictConfig(config)
            else:
                # デフォルト設定
                self._setup_default_logging()
        except Exception as e:
            print(f"ログ設定エラー: {e}", file=sys.stderr)
            self._setup_default_logging()
    
    def _setup_default_logging(self):
        """デフォルトのログ設定"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(LOG_DIR, 'app.log'))
            ]
        )
    
    def set_context(self, **kwargs):
        """ログコンテキストを設定"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """ログコンテキストをクリア"""
        self.context = {}
    
    def _format_message(self, message: str, extra: Dict[str, Any]) -> str:
        """構造化メッセージを作成"""
        data = {
            'timestamp': datetime.utcnow().isoformat(),
            'logger': self.name,
            'message': message,
            **self.context,
            **extra
        }
        return json.dumps(data, ensure_ascii=False)
    
    def debug(self, message: str, **kwargs):
        """DEBUGレベルログ"""
        if kwargs:
            message = self._format_message(message, kwargs)
        self.logger.debug(message)
    
    def info(self, message: str, **kwargs):
        """INFOレベルログ"""
        if kwargs:
            message = self._format_message(message, kwargs)
        self.logger.info(message)
    
    def warning(self, message: str, **kwargs):
        """WARNINGレベルログ"""
        if kwargs:
            message = self._format_message(message, kwargs)
        self.logger.warning(message)
    
    def error(self, message: str, exc_info: bool = False, **kwargs):
        """ERRORレベルログ"""
        if exc_info:
            kwargs['traceback'] = traceback.format_exc()
        if kwargs:
            message = self._format_message(message, kwargs)
        self.logger.error(message, exc_info=exc_info)
    
    def critical(self, message: str, **kwargs):
        """CRITICALレベルログ"""
        if kwargs:
            message = self._format_message(message, kwargs)
        self.logger.critical(message)
    
    @contextmanager
    def timer(self, operation: str, **kwargs):
        """処理時間を計測するコンテキストマネージャー"""
        start_time = time.perf_counter()
        self.info(f"{operation} 開始", **kwargs)
        
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_time
            self.info(
                f"{operation} 完了",
                duration_ms=round(elapsed * 1000, 2),
                **kwargs
            )
    
    def log_performance(self, func: Callable) -> Callable:
        """関数の実行時間を記録するデコレーター"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.timer(f"{func.__name__}"):
                return func(*args, **kwargs)
        return wrapper


class AudioLogger(StructuredLogger):
    """音声処理専用ロガー"""
    
    def __init__(self):
        super().__init__("audio")
    
    def log_recording(self, duration: float, samples: int, sample_rate: int):
        """録音情報をログ"""
        self.info(
            "録音完了",
            duration_sec=duration,
            samples=samples,
            sample_rate=sample_rate,
            size_kb=round(samples * 2 / 1024, 2)  # int16 = 2 bytes
        )
    
    def log_vad(self, speech_segments: int, total_duration: float):
        """VAD(音声区間検出)結果をログ"""
        self.info(
            "VAD処理完了",
            speech_segments=speech_segments,
            total_duration_sec=total_duration
        )
    
    def log_wake_word(self, detected: bool, confidence: float = 0.0):
        """Wake Word検出結果をログ"""
        level = "info" if detected else "debug"
        getattr(self, level)(
            "Wake Word検出",
            detected=detected,
            confidence=confidence
        )


class APILogger(StructuredLogger):
    """API通信専用ロガー"""
    
    def __init__(self):
        super().__init__("api")
    
    def log_request(self, method: str, url: str, headers: Dict = None, body: Any = None):
        """APIリクエストをログ"""
        self.info(
            "APIリクエスト",
            method=method,
            url=url,
            headers=headers or {},
            body_size=len(str(body)) if body else 0
        )
    
    def log_response(self, status_code: int, response_time: float, body: Any = None):
        """APIレスポンスをログ"""
        level = "info" if 200 <= status_code < 400 else "warning"
        getattr(self, level)(
            "APIレスポンス",
            status_code=status_code,
            response_time_ms=round(response_time * 1000, 2),
            body_size=len(str(body)) if body else 0
        )
    
    def log_error(self, error_type: str, error_message: str, **kwargs):
        """APIエラーをログ"""
        self.error(
            f"APIエラー: {error_type}",
            error_message=error_message,
            **kwargs
        )


class PerformanceLogger(StructuredLogger):
    """パフォーマンス監視専用ロガー"""
    
    def __init__(self):
        super().__init__("performance")
        self.metrics = {}
    
    def start_timer(self, metric_name: str):
        """タイマー開始"""
        self.metrics[metric_name] = time.perf_counter()
    
    def end_timer(self, metric_name: str) -> float:
        """タイマー終了して経過時間を返す"""
        if metric_name in self.metrics:
            elapsed = time.perf_counter() - self.metrics[metric_name]
            del self.metrics[metric_name]
            self.log_metric(metric_name, elapsed * 1000, "ms")
            return elapsed
        return 0.0
    
    def log_metric(self, name: str, value: float, unit: str = ""):
        """メトリクスを記録"""
        self.info(
            "パフォーマンスメトリクス",
            metric_name=name,
            value=value,
            unit=unit
        )
    
    def log_memory_usage(self):
        """メモリ使用量を記録"""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            self.info(
                "メモリ使用量",
                rss_mb=round(memory_info.rss / 1024 / 1024, 2),
                vms_mb=round(memory_info.vms / 1024 / 1024, 2) if hasattr(memory_info, 'vms') else 0,
                percent=round(process.memory_percent(), 2)
            )
        except ImportError:
            pass  # psutilがない場合はスキップ
    
    def log_cpu_usage(self):
        """CPU使用率を記録"""
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.info(
                "CPU使用率",
                cpu_percent=cpu_percent,
                cpu_count=psutil.cpu_count()
            )
        except ImportError:
            pass


class LogManager:
    """ログシステム全体を管理"""
    
    _instances = {}
    
    @classmethod
    def get_logger(cls, name: str) -> StructuredLogger:
        """ロガーインスタンスを取得"""
        if name not in cls._instances:
            if name == "audio":
                cls._instances[name] = AudioLogger()
            elif name == "api":
                cls._instances[name] = APILogger()
            elif name == "performance":
                cls._instances[name] = PerformanceLogger()
            else:
                cls._instances[name] = StructuredLogger(name)
        return cls._instances[name]
    
    @classmethod
    def setup_request_id(cls, request_id: str):
        """リクエストIDを全ロガーに設定"""
        for logger in cls._instances.values():
            logger.set_context(request_id=request_id)
    
    @classmethod
    def clear_request_id(cls):
        """リクエストIDをクリア"""
        for logger in cls._instances.values():
            logger.clear_context()


# 便利な関数
def get_logger(name: str) -> StructuredLogger:
    """ロガーを取得"""
    return LogManager.get_logger(name)


def log_function_call(logger_name: str = "app"):
    """関数呼び出しをログするデコレーター"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name)
            func_name = f"{func.__module__}.{func.__name__}"
            
            logger.debug(
                f"関数呼び出し: {func_name}",
                args=str(args)[:100],
                kwargs=str(kwargs)[:100]
            )
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"関数成功: {func_name}")
                return result
            except Exception as e:
                logger.error(
                    f"関数エラー: {func_name}",
                    error=str(e),
                    exc_info=True
                )
                raise
        return wrapper
    return decorator


# エクスポート
__all__ = [
    'StructuredLogger',
    'AudioLogger',
    'APILogger',
    'PerformanceLogger',
    'LogManager',
    'get_logger',
    'log_function_call'
]
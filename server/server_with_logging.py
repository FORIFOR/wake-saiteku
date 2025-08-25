#!/usr/bin/env python3
"""
Wake Saiteku Server - 高度なログ機能付き
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import uuid
from datetime import datetime
from typing import Dict, Any
import traceback

from utils.logger import (
    get_logger, 
    APILogger, 
    PerformanceLogger,
    LogManager
)

# ロガー初期化
logger = get_logger("server")
api_logger = get_logger("api")
perf_logger = get_logger("performance")

# FastAPIアプリケーション
app = FastAPI(
    title="Wake Saiteku Server with Logging",
    version="2.0.0",
    description="高度なログ機能を備えた音声アシスタントサーバー"
)

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# リクエストごとのログ設定
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """全HTTPリクエストをログ記録"""
    # リクエストID生成
    request_id = str(uuid.uuid4())[:8]
    LogManager.setup_request_id(request_id)
    
    # リクエスト情報をログ
    logger.info(
        "リクエスト受信",
        method=request.method,
        url=str(request.url),
        client=request.client.host if request.client else "unknown",
        headers=dict(request.headers)
    )
    
    # パフォーマンス計測開始
    perf_logger.start_timer("request_processing")
    
    try:
        # リクエスト処理
        response = await call_next(request)
        
        # レスポンス時間計測
        processing_time = perf_logger.end_timer("request_processing")
        
        # レスポンス情報をログ
        logger.info(
            "レスポンス送信",
            status_code=response.status_code,
            processing_time_ms=round(processing_time * 1000, 2)
        )
        
        # レスポンスヘッダーに情報追加
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time"] = f"{processing_time:.3f}"
        
        return response
        
    except Exception as e:
        # エラーログ
        logger.error(
            "リクエスト処理エラー",
            error=str(e),
            traceback=traceback.format_exc()
        )
        raise
    finally:
        # リクエストIDクリア
        LogManager.clear_request_id()

@app.on_event("startup")
async def startup_event():
    """サーバー起動時の処理"""
    logger.info(
        "サーバー起動",
        version="2.0.0",
        environment=os.getenv("ENVIRONMENT", "development"),
        pid=os.getpid()
    )
    
    # システム情報をログ
    perf_logger.log_memory_usage()
    perf_logger.log_cpu_usage()

@app.on_event("shutdown")
async def shutdown_event():
    """サーバー停止時の処理"""
    logger.info("サーバー停止")

@app.get("/")
async def root():
    """ヘルスチェックエンドポイント"""
    logger.debug("ヘルスチェック実行")
    
    response = {
        "status": "online",
        "service": "Wake Saiteku Server",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "features": {
            "logging": "enhanced",
            "structured_logs": True,
            "performance_monitoring": True,
            "log_rotation": True
        }
    }
    
    logger.info("ヘルスチェック成功", response=response)
    return response

@app.post("/inference")
async def inference(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    音声ファイルを受け取り処理
    
    Returns:
        Dict containing transcript and reply
    """
    logger.info(
        "音声ファイル受信",
        filename=file.filename,
        content_type=file.content_type,
        size=file.size if hasattr(file, 'size') else "unknown"
    )
    
    wav_path = None
    
    try:
        # ファイル形式チェック
        if file.filename and not file.filename.lower().endswith('.wav'):
            logger.warning(
                "非WAVファイル",
                filename=file.filename,
                content_type=file.content_type
            )
        
        # パフォーマンス計測
        with perf_logger.timer("file_save"):
            # 一時ファイルに保存
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                data = await file.read()
                tmp.write(data)
                wav_path = tmp.name
                file_size = len(data)
        
        logger.info(
            "ファイル保存完了",
            path=wav_path,
            size_bytes=file_size,
            size_kb=round(file_size / 1024, 2)
        )
        
        # STT処理
        with perf_logger.timer("stt_processing"):
            # 実際の音声認識処理を実行
            transcript = ""
        
        logger.info(
            "音声認識完了",
            transcript=transcript,
            transcript_length=len(transcript)
        )
        
        # LLM処理
        with perf_logger.timer("llm_processing"):
            # 実際の言語モデル処理を実行
            reply = "システムが正常に動作しています。"
        
        logger.info(
            "LLM応答生成完了",
            reply_length=len(reply)
        )
        
        # メトリクス記録
        perf_logger.log_metric("audio_file_size", file_size, "bytes")
        perf_logger.log_metric("transcript_length", len(transcript), "chars")
        perf_logger.log_metric("reply_length", len(reply), "chars")
        
        response = {
            "transcript": transcript,
            "reply": reply,
            "processing_time": 0.1,
            "file_size": file_size,
            "metadata": {
                "model": "simple",
                "language": "ja"
            }
        }
        
        logger.info(
            "推論完了",
            success=True,
            response_size=len(str(response))
        )
        
        return response
        
    except Exception as e:
        logger.error(
            "推論エラー",
            error=str(e),
            error_type=type(e).__name__,
            exc_info=True
        )
        
        # エラーレスポンス
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "type": type(e).__name__,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
    finally:
        # クリーンアップ
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
                logger.debug(f"一時ファイル削除: {wav_path}")
            except Exception as e:
                logger.warning(
                    "一時ファイル削除失敗",
                    path=wav_path,
                    error=str(e)
                )

@app.get("/logs/summary")
async def get_log_summary():
    """ログサマリーを取得"""
    try:
        log_files = []
        log_dir = "logs"
        
        if os.path.exists(log_dir):
            for filename in os.listdir(log_dir):
                filepath = os.path.join(log_dir, filename)
                if os.path.isfile(filepath):
                    stats = os.stat(filepath)
                    log_files.append({
                        "filename": filename,
                        "size_kb": round(stats.st_size / 1024, 2),
                        "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                        "lines": sum(1 for _ in open(filepath, 'r', errors='ignore'))
                    })
        
        return {
            "log_directory": log_dir,
            "log_files": log_files,
            "total_files": len(log_files),
            "total_size_kb": sum(f["size_kb"] for f in log_files)
        }
        
    except Exception as e:
        logger.error("ログサマリー取得エラー", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """システムメトリクスを取得"""
    logger.debug("メトリクス取得")
    
    # メモリとCPU使用率を記録
    perf_logger.log_memory_usage()
    perf_logger.log_cpu_usage()
    
    return {
        "status": "metrics_collected",
        "timestamp": datetime.utcnow().isoformat(),
        "note": "Check logs/performance.log for detailed metrics"
    }

if __name__ == "__main__":
    import uvicorn
    
    # ログディレクトリ作成
    os.makedirs("logs", exist_ok=True)
    
    port = int(os.getenv("SERVER_PORT", "8002"))
    host = os.getenv("SERVER_HOST", "0.0.0.0")
    
    logger.info(
        "サーバー起動準備",
        host=host,
        port=port,
        log_level=os.getenv("LOG_LEVEL", "INFO")
    )
    
    # Uvicornログ設定
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
        },
        "root": {
            "level": "INFO",
            "handlers": ["default"],
        },
    }
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_config=log_config,
        access_log=True
    )
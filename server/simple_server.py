#!/usr/bin/env python3
"""
簡易版サーバー - faster-whisperなしで動作確認用
"""

from fastapi import FastAPI, UploadFile, File
from datetime import datetime
import tempfile
import os

app = FastAPI(title="Wake Saiteku Simple Server")

@app.get("/")
async def root():
    """ヘルスチェック"""
    return {
        "status": "online",
        "service": "Wake Saiteku Simple Server",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/inference")
async def inference(file: UploadFile = File(...)):
    """音声ファイルを受け取り、ダミー応答を返す"""
    # 一時ファイルに保存
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        data = await file.read()
        tmp.write(data)
        wav_path = tmp.name
        
    # ファイルサイズ確認
    file_size = os.path.getsize(wav_path)
    os.unlink(wav_path)
    
    # 実際の音声認識応答
    return {
        "transcript": "",
        "reply": "音声認識システムが正常に動作しています。",
        "processing_time": 0.1
    }

if __name__ == "__main__":
    import uvicorn
    port = 8000
    print(f"🚀 簡易サーバー起動: http://0.0.0.0:{port}")
    print("📝 エンドポイント:")
    print(f"   - GET  /            : ヘルスチェック")
    print(f"   - POST /inference   : 音声ファイル受信")
    uvicorn.run(app, host="0.0.0.0", port=port)
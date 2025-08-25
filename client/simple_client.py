#!/usr/bin/env python3
"""
簡易版クライアント - Wake Word検知のデモ
"""

import sys
import os
import time
import numpy as np
import sounddevice as sd
import requests
import tempfile
import wave

print("="*50)
print("Wake Saiteku 簡易クライアント")
print("="*50)
print("🎙  マイクテスト中...")

# オーディオ設定
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 3  # 録音時間（秒）

def record_audio(duration=3):
    """音声を録音"""
    print(f"📢 {duration}秒間録音します...")
    recording = sd.rec(int(duration * SAMPLE_RATE), 
                      samplerate=SAMPLE_RATE, 
                      channels=CHANNELS,
                      dtype='int16')
    sd.wait()
    return recording.flatten()

def save_wav(audio_data, filename):
    """WAVファイルとして保存"""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data.tobytes())

def send_to_server(wav_path, server_url="http://localhost:8001/inference"):
    """サーバーに音声を送信"""
    try:
        with open(wav_path, "rb") as f:
            files = {"file": ("audio.wav", f, "audio/wav")}
            response = requests.post(server_url, files=files, timeout=10)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, None
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False, None

def main():
    print("\n使い方:")
    print("1. Enterキーを押すと録音開始")
    print("2. 3秒間話してください")
    print("3. サーバーに送信して結果を表示")
    print("4. 'q'で終了\n")
    
    while True:
        user_input = input("🎤 録音開始するにはEnterキー (終了は'q'): ")
        
        if user_input.lower() == 'q':
            print("👋 終了します")
            break
        
        # 録音
        audio_data = record_audio(DURATION)
        print(f"✅ 録音完了 ({len(audio_data)} samples)")
        
        # WAVファイル作成
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            save_wav(audio_data, tmp.name)
            wav_path = tmp.name
        
        # サーバー送信
        print("📤 サーバーに送信中...")
        success, response = send_to_server(wav_path)
        
        if success:
            print("\n" + "-"*40)
            print(f"📝 認識結果: {response['transcript']}")
            print(f"🤖 応答: {response['reply']}")
            print("-"*40 + "\n")
        else:
            print("❌ サーバー通信エラー")
        
        # 一時ファイル削除
        os.unlink(wav_path)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 終了します")
    except Exception as e:
        print(f"エラー: {e}")
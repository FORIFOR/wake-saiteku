#! Wake Saiteku Makefile (simplified)

.PHONY: help install setup run-server run-client clean

help:
	@echo "Wake Saiteku - Commands"
	@echo "  make setup       - 初期セットアップ（対話）"
	@echo "  make install     - 依存パッケージインストール（サーバー/クライアント）"
	@echo "  make run-server  - サーバー起動"
	@echo "  make run-client  - クライアント起動"
	@echo "  make clean       - 一時ファイル削除"

setup:
	@echo "セットアップを実行中..."
	./scripts/setup.sh

install:
	@echo "依存パッケージをインストール中..."
	pip install -r requirements-server.txt
	pip install -r requirements-client.txt

run-server:
	@echo "=== サーバー起動 ==="
	python server/server.py

run-client:
	@echo "=== クライアント起動 ==="
	python client/client.py

clean:
	@echo "=== 一時ファイル削除 ==="
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

# Wake Saiteku Makefile
# TDD開発用のコマンド集

.PHONY: help test test-unit test-integration test-performance coverage lint format clean install setup run-server run-client

# デフォルトターゲット
help:
	@echo "Wake Saiteku - TDD開発コマンド"
	@echo ""
	@echo "使用可能なコマンド:"
	@echo "  make setup          - 初期セットアップ"
	@echo "  make install        - 依存パッケージインストール"
	@echo "  make test           - すべてのテスト実行"
	@echo "  make test-unit      - ユニットテストのみ"
	@echo "  make test-integration - 統合テストのみ"
	@echo "  make test-performance - パフォーマンステスト"
	@echo "  make coverage       - カバレッジレポート生成"
	@echo "  make lint           - コードのlint実行"
	@echo "  make format         - コードフォーマット"
	@echo "  make run-server     - サーバー起動"
	@echo "  make run-client     - クライアント起動"
	@echo "  make clean          - 一時ファイル削除"

# セットアップ
setup:
	@echo "セットアップを実行中..."
	./scripts/setup.sh

# 依存パッケージインストール
install:
	@echo "依存パッケージをインストール中..."
	pip install -r requirements-server.txt
	pip install -r requirements-client.txt
	pip install -r requirements-test.txt

# テスト実行（TDD: Red → Green → Refactor）
test:
	@echo "=== TDD: テスト実行 ==="
	pytest tests/ -v

test-unit:
	@echo "=== ユニットテスト実行 ==="
	pytest tests/ -v -m "not integration and not performance"

test-integration:
	@echo "=== 統合テスト実行 ==="
	pytest tests/ -v -m integration

test-performance:
	@echo "=== パフォーマンステスト実行 ==="
	pytest tests/ -v -m performance

# カバレッジ
coverage:
	@echo "=== カバレッジレポート生成 ==="
	pytest tests/ --cov=server --cov=client --cov-report=html --cov-report=term
	@echo "レポート: htmlcov/index.html"

# コード品質
lint:
	@echo "=== Lintチェック ==="
	flake8 server/ client/ tests/ --max-line-length=100 --ignore=E203,W503 || true
	mypy server/ client/ --ignore-missing-imports || true

format:
	@echo "=== コードフォーマット ==="
	black server/ client/ tests/
	isort server/ client/ tests/

# サーバー/クライアント起動
run-server:
	@echo "=== サーバー起動 ==="
	@if [ -f config/server.env ]; then \
		set -a && source config/server.env && set +a; \
	fi
	python server/server.py

run-client:
	@echo "=== クライアント起動 ==="
	@if [ -f config/client.env ]; then \
		set -a && source config/client.env && set +a; \
	fi
	python client/client.py

# クリーンアップ
clean:
	@echo "=== 一時ファイル削除 ==="
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/
	rm -f .coverage
	rm -f coverage.xml
	rm -rf *.egg-info
	rm -rf build/
	rm -rf dist/

# TDDサイクル（連続実行）
tdd:
	@echo "=== TDDサイクル開始 ==="
	@echo "テストを監視モードで実行します..."
	@echo "コードを変更すると自動的にテストが再実行されます"
	pytest tests/ --watch

# Docker関連（将来の拡張用）
docker-build:
	docker build -t wake-saiteku:latest .

docker-run-server:
	docker run -p 8000:8000 --env-file config/server.env wake-saiteku:latest python server/server.py

docker-run-client:
	docker run --device /dev/snd --env-file config/client.env wake-saiteku:latest python client/client.py
#!/bin/bash
#
# Wake Saiteku セットアップスクリプト
#

set -e

echo "========================================="
echo "Wake Saiteku セットアップ"
echo "========================================="

# 色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# OS判定
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    echo -e "${GREEN}OS: Linux${NC}"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="mac"
    echo -e "${GREEN}OS: macOS${NC}"
else
    echo -e "${RED}サポートされていないOS: $OSTYPE${NC}"
    exit 1
fi

# Python確認
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python3が見つかりません${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}Python バージョン: $PYTHON_VERSION${NC}"

# 作業ディレクトリ
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo -e "\n${YELLOW}プロジェクトディレクトリ: $PROJECT_ROOT${NC}"

# ========================================
# 1. Python仮想環境セットアップ
# ========================================
echo -e "\n${YELLOW}[1/5] Python仮想環境をセットアップ中...${NC}"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ 仮想環境を作成しました${NC}"
else
    echo -e "${GREEN}✓ 仮想環境は既に存在します${NC}"
fi

# 仮想環境をアクティベート
source venv/bin/activate

# pipをアップグレード
pip install --upgrade pip > /dev/null 2>&1

# ========================================
# 2. システム依存パッケージ（Linux/Raspberry Pi）
# ========================================
if [ "$OS" == "linux" ]; then
    echo -e "\n${YELLOW}[2/5] システムパッケージをインストール中...${NC}"
    
    # PortAudio（sounddevice用）
    if ! dpkg -l | grep -q libportaudio2; then
        echo "PortAudioをインストール中..."
        sudo apt-get update > /dev/null 2>&1
        sudo apt-get install -y libportaudio2 > /dev/null 2>&1
        echo -e "${GREEN}✓ PortAudioをインストールしました${NC}"
    else
        echo -e "${GREEN}✓ PortAudioは既にインストール済み${NC}"
    fi
    
    # llama.cpp用依存（オプション）
    if [ ! -d "$HOME/llama.cpp" ]; then
        echo -e "${YELLOW}llama.cpp用の依存パッケージをインストールしますか? (y/n)${NC}"
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            sudo apt-get install -y git build-essential cmake libopenblas-dev > /dev/null 2>&1
            echo -e "${GREEN}✓ llama.cpp依存をインストールしました${NC}"
        fi
    fi
else
    echo -e "\n${YELLOW}[2/5] macOSではシステムパッケージのインストールをスキップ${NC}"
fi

# ========================================
# 3. Pythonパッケージインストール
# ========================================
echo -e "\n${YELLOW}[3/5] Pythonパッケージをインストール中...${NC}"

# サーバー/クライアント判定
echo -e "${YELLOW}インストールタイプを選択してください:${NC}"
echo "  1) サーバー（リモートPC）"
echo "  2) クライアント（端末/Raspberry Pi）"
echo "  3) 両方"
read -p "選択 (1-3): " install_type

case $install_type in
    1)
        echo "サーバーパッケージをインストール中..."
        pip install -r requirements-server.txt
        echo -e "${GREEN}✓ サーバーパッケージをインストールしました${NC}"
        ;;
    2)
        echo "クライアントパッケージをインストール中..."
        pip install -r requirements-client.txt
        echo -e "${GREEN}✓ クライアントパッケージをインストールしました${NC}"
        ;;
    3)
        echo "全パッケージをインストール中..."
        pip install -r requirements-server.txt
        pip install -r requirements-client.txt
        echo -e "${GREEN}✓ 全パッケージをインストールしました${NC}"
        ;;
    *)
        echo -e "${RED}無効な選択${NC}"
        exit 1
        ;;
esac

# ========================================
# 4. Voskモデルダウンロード（クライアント用）
# ========================================
if [ "$install_type" == "2" ] || [ "$install_type" == "3" ]; then
    echo -e "\n${YELLOW}[4/5] Vosk日本語モデルをダウンロード中...${NC}"
    
    MODELS_DIR="$PROJECT_ROOT/models"
    mkdir -p "$MODELS_DIR"
    
    if [ ! -d "$MODELS_DIR/ja" ]; then
        cd "$MODELS_DIR"
        
        # 軽量モデル（約50MB）
        echo "軽量日本語モデルをダウンロード中..."
        curl -L -o ja-small.zip https://alphacephei.com/vosk/models/vosk-model-small-ja-0.22.zip
        
        echo "展開中..."
        unzip -q ja-small.zip
        mv vosk-model-small-ja-0.22 ja
        rm ja-small.zip
        
        cd "$PROJECT_ROOT"
        echo -e "${GREEN}✓ Vosk日本語モデルをダウンロードしました${NC}"
    else
        echo -e "${GREEN}✓ Vosk日本語モデルは既に存在します${NC}"
    fi
else
    echo -e "\n${YELLOW}[4/5] サーバーのみのインストールのためVoskモデルをスキップ${NC}"
fi

# ========================================
# 5. 設定ファイル作成
# ========================================
echo -e "\n${YELLOW}[5/5] 設定ファイルを作成中...${NC}"

CONFIG_DIR="$PROJECT_ROOT/config"
mkdir -p "$CONFIG_DIR"

# サーバー設定テンプレート
if [ "$install_type" == "1" ] || [ "$install_type" == "3" ]; then
    if [ ! -f "$CONFIG_DIR/server.env" ]; then
        cat > "$CONFIG_DIR/server.env" << EOF
# Wake Saiteku Server Configuration

# Whisper設定
FW_MODEL=small
FW_COMPUTE=int8
FW_DEVICE=cpu

# LLM設定（LM Studio / OpenAI互換API）
LM_URL=http://localhost:1234/v1/chat/completions
LM_API_KEY=lm-studio
LM_MODEL=local-model
LM_TIMEOUT=60

# サーバー設定
SERVER_PORT=8000
SERVER_HOST=0.0.0.0
EOF
        echo -e "${GREEN}✓ サーバー設定ファイルを作成しました${NC}"
    fi
fi

# クライアント設定テンプレート
if [ "$install_type" == "2" ] || [ "$install_type" == "3" ]; then
    if [ ! -f "$CONFIG_DIR/client.env" ]; then
        # IPアドレスを取得
        if [ "$OS" == "mac" ]; then
            LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}')
        else
            LOCAL_IP=$(hostname -I | awk '{print $1}')
        fi
        
        cat > "$CONFIG_DIR/client.env" << EOF
# Wake Saiteku Client Configuration

# リモートサーバー設定
SERVER_URL=http://${LOCAL_IP}:8000/inference

# ローカルフォールバック設定
LOCAL_STT_ENABLED=true
LLM_LOCAL_URL=http://127.0.0.1:8081/v1/chat/completions
LLM_LOCAL_MODEL=qwen2.5-1.5b-instruct-q4_k_m

# タイムアウト設定
REQUEST_TIMEOUT=30
EOF
        echo -e "${GREEN}✓ クライアント設定ファイルを作成しました${NC}"
        echo -e "${YELLOW}注: SERVER_URLを実際のサーバーIPに変更してください${NC}"
    fi
fi

# ========================================
# 完了
# ========================================
echo -e "\n${GREEN}=========================================${NC}"
echo -e "${GREEN}セットアップが完了しました！${NC}"
echo -e "${GREEN}=========================================${NC}"

echo -e "\n${YELLOW}次のステップ:${NC}"

if [ "$install_type" == "1" ] || [ "$install_type" == "3" ]; then
    echo -e "\n${YELLOW}サーバー起動:${NC}"
    echo "  1. 設定を編集: config/server.env"
    echo "  2. 起動:"
    echo "     source venv/bin/activate"
    echo "     source config/server.env"
    echo "     python server/server.py"
fi

if [ "$install_type" == "2" ] || [ "$install_type" == "3" ]; then
    echo -e "\n${YELLOW}クライアント起動:${NC}"
    echo "  1. 設定を編集: config/client.env"
    echo "  2. 起動:"
    echo "     source venv/bin/activate"
    echo "     source config/client.env"
    echo "     python client/client.py"
fi

echo -e "\n${YELLOW}詳細はREADME.mdを参照してください${NC}"
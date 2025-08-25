#!/usr/bin/env bash
set -euo pipefail

# macOS LaunchAgent installer for Wake Saiteku Server

LABEL="com.saiteku.server"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLIST="$HOME/Library/LaunchAgents/${LABEL}.plist"
PORT="${SERVER_PORT:-9050}"
PYTHON_BIN="$PROJECT_ROOT/venv311/bin/python"
ZSH_BIN="/bin/zsh"

echo "Project: $PROJECT_ROOT"
echo "Plist:   $PLIST"
echo "Port:    $PORT"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "[!] $PYTHON_BIN not found. Create venv first:"
  echo "    /opt/homebrew/bin/python3.11 -m venv $PROJECT_ROOT/venv311 && source $PROJECT_ROOT/venv311/bin/activate && pip install -r $PROJECT_ROOT/requirements-server.txt"
  exit 1
fi

mkdir -p "$HOME/Library/LaunchAgents" "$PROJECT_ROOT/logs"

cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>${ZSH_BIN}</string>
    <string>-lc</string>
    <string>cd ${PROJECT_ROOT} && source venv311/bin/activate && source config/server.env 2>/dev/null || true; SERVER_HOST=127.0.0.1 SERVER_PORT=${PORT} LOG_TO_FILE=true LOG_DIR=logs ${PYTHON_BIN} server/server.py</string>
  </array>
  <key>WorkingDirectory</key>
  <string>${PROJECT_ROOT}</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>${PROJECT_ROOT}/logs/launchd-server.out</string>
  <key>StandardErrorPath</key>
  <string>${PROJECT_ROOT}/logs/launchd-server.err</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHONUNBUFFERED</key>
    <string>1</string>
    <key>LOG_LEVEL</key>
    <string>INFO</string>
  </dict>
</dict>
</plist>
PLIST

echo "[+] Wrote $PLIST"

# Reload
launchctl unload "$PLIST" >/dev/null 2>&1 || true
launchctl load "$PLIST"
launchctl start "$LABEL" || true

echo "[✓] LaunchAgent loaded. Check status: launchctl list | grep ${LABEL}"
echo "    Tail logs: tail -f ${PROJECT_ROOT}/logs/server.log ${PROJECT_ROOT}/logs/launchd-server.err"
echo "    Health:   curl -sS http://127.0.0.1:${PORT}/"


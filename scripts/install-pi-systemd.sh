#!/usr/bin/env bash
set -euo pipefail

# Raspberry Pi systemd installer for Wake Saiteku Client

SERVICE_SRC="$(cd "$(dirname "$0")/.." && pwd)/config/saiteku-client.service"
SERVICE_DST="/etc/systemd/system/saiteku-client.service"

if [[ $EUID -ne 0 ]]; then
  echo "[!] Please run as root: sudo $0"
  exit 1
fi

if [[ ! -f "$SERVICE_SRC" ]]; then
  echo "[!] Service file not found: $SERVICE_SRC"
  exit 1
fi

cp "$SERVICE_SRC" "$SERVICE_DST"
systemctl daemon-reload
systemctl enable --now saiteku-client

echo "[✓] Installed and started saiteku-client"
echo "    Logs: journalctl -u saiteku-client -f"


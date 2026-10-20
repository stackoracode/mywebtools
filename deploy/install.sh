#!/usr/bin/env bash
# Install mywebtools on Ubuntu 20.04+ (aarch64 boards such as LubanCat, or x86_64).
# Usage: deploy/install.sh [--with-whisper] [--no-service]
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
USER_NAME="${SUDO_USER:-$USER}"
WITH_WHISPER=0
SERVICE=1
for arg in "$@"; do
  case "$arg" in
    --with-whisper) WITH_WHISPER=1 ;;
    --no-service) SERVICE=0 ;;
    *) echo "unknown option $arg" >&2; exit 1 ;;
  esac
done

echo "==> System packages (ffmpeg, tesseract + language packs, SMB client, CJK subtitle fonts)"
sudo apt-get update
sudo apt-get install -y ffmpeg tesseract-ocr tesseract-ocr-eng tesseract-ocr-chi-sim tesseract-ocr-chi-tra \
  tesseract-ocr-jpn tesseract-ocr-kor cifs-utils curl ca-certificates fonts-noto-cjk

echo "==> uv + Python 3.11 (Ubuntu 20.04 ships 3.8, too old for current wheels)"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
cd "$DIR"
"$UV" venv --python 3.11 .venv
"$UV" pip install --python .venv/bin/python -r requirements.txt
if [ "$WITH_WHISPER" = 1 ]; then
  "$UV" pip install --python .venv/bin/python -r requirements-optional.txt
fi

if [ ! -f .env ]; then
  cp .env.example .env
  chmod 600 .env
  echo "==> Created .env (edit it or use the Admin page)"
fi

if [ "$SERVICE" = 1 ]; then
  echo "==> systemd service"
  sed -e "s#__DIR__#$DIR#g" -e "s#__USER__#$USER_NAME#g" deploy/mywebtools.service | sudo tee /etc/systemd/system/mywebtools.service >/dev/null
  sudo systemctl daemon-reload
  sudo systemctl enable --now mywebtools
  sleep 2
  systemctl --no-pager --lines=5 status mywebtools || true
fi

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo
echo "Done. Open http://${IP:-<board-ip>}:$(grep -E '^MWT_PORT=' .env | cut -d= -f2 || echo 8080) on your phone."
echo "Next: share the Duix folder on Windows and run deploy/mount-duix-share.sh (see README)."

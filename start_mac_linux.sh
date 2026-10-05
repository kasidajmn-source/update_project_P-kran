#!/usr/bin/env bash
# เริ่มเซิร์ฟเวอร์ Crack-AI (macOS / Linux) ครั้งแรกจะติดตั้งไลบรารีให้อัตโนมัติ
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -x .venv/bin/python ]; then
  echo "[1/3] สร้าง virtual environment ..."
  $PY -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  echo "[2/3] ติดตั้ง PyTorch ..."
  if [ "$(uname)" = "Darwin" ]; then
    .venv/bin/python -m pip install torch torchvision            # Apple Silicon ใช้ MPS
  else
    .venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  fi
  echo "[3/3] ติดตั้งไลบรารีอื่น ..."
  .venv/bin/python -m pip install -r server/requirements.txt
fi
( sleep 3; (command -v open >/dev/null && open http://127.0.0.1:8765) || (command -v xdg-open >/dev/null && xdg-open http://127.0.0.1:8765) || true ) &
exec .venv/bin/python server/server.py "$@"

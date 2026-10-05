@echo off
REM Crack-AI server for Windows. First run installs Python packages (about 3-4 GB).
title Crack-AI server
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating Python virtual environment...
  py -3.11 -m venv .venv 2>nul || python -m venv .venv
  if not exist ".venv\Scripts\python.exe" (
    echo Python 3.10-3.12 not found. Install from https://www.python.org and tick "Add python.exe to PATH".
    pause
    exit /b 1
  )
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  echo [2/3] Installing PyTorch with CUDA 12.8 - supports RTX 50 series...
  ".venv\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  echo [3/3] Installing other packages...
  ".venv\Scripts\python.exe" -m pip install -r server\requirements.txt
)
echo.
echo Open http://127.0.0.1:8765 in your browser. Close this window to stop the server.
start "" http://127.0.0.1:8765
".venv\Scripts\python.exe" server\server.py
pause

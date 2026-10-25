#!/usr/bin/env bash
# Signora — one-command local setup & run (macOS / Linux)
set -e
cd "$(dirname "$0")"

# Find Python 3.11
PY=""
for c in python3.11 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; exit(0 if sys.version_info[:2]==(3,11) else 1)'; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  echo "❌ Python 3.11 is required (TensorFlow 2.13 does not support 3.12+)."
  echo "   macOS:  brew install python@3.11"
  echo "   Ubuntu: sudo apt install python3.11 python3.11-venv"
  exit 1
fi

if [ ! -d venv ]; then
  echo "📦 Creating virtual environment…"
  "$PY" -m venv venv
  ./venv/bin/python -m pip install --upgrade pip
  echo "📥 Installing dependencies (first run takes a few minutes)…"
  ./venv/bin/pip install -r requirements-local.txt
fi

exec ./venv/bin/python run_local.py

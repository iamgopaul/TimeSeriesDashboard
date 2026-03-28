#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PORT="${PORT:-8502}"
HOST="${HOST:-127.0.0.1}"

cd "$ROOT_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment in $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

echo "Installing requirements"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/requirements.txt"

echo "Starting WRDS Forecast Dashboard at http://$HOST:$PORT"
exec "$VENV_DIR/bin/python" -m streamlit run "$ROOT_DIR/app.py" \
  --server.headless true \
  --server.address "$HOST" \
  --server.port "$PORT"

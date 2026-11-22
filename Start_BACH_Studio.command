#!/bin/bash
set -e
cd "$(dirname "$0")"

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "Python 3.11 is required. Install it from https://www.python.org/downloads/macos/ and run this file again."
  read -r -p "Press Enter to close..."
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Creating isolated Python environment..."
  python3.11 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Starting BACH Research Studio..."
python -m streamlit run app.py

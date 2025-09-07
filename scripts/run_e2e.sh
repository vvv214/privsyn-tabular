#!/usr/bin/env bash
set -euo pipefail

echo "[run_e2e] Using Python: $(command -v python3 || true)"

if [ ! -d .venv ]; then
  echo "[run_e2e] Creating virtualenv .venv"
  python3 -m venv .venv
fi

source .venv/bin/activate

echo "[run_e2e] Installing Python deps"
pip install -r requirements.txt >/dev/null

echo "[run_e2e] Installing Playwright browsers"
python -m playwright install --with-deps >/dev/null

echo "[run_e2e] Installing frontend deps"
pushd frontend >/dev/null
npm install >/dev/null
popd >/dev/null

echo "[run_e2e] Running E2E: backend+frontend"
E2E=1 pytest -q -k e2e


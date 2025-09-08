#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root and activate venv
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT_DIR/.venv/bin/activate"

# Run the FastAPI application
cd "$ROOT_DIR"
uvicorn web_app.main:app --reload --port 8001

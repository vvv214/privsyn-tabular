#!/usr/bin/env bash
set -euo pipefail

# Script to start the FastAPI backend in production using Gunicorn

# Resolve repo root and activate venv
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT_DIR/.venv/bin/activate"

# Run the FastAPI application with Gunicorn
cd "$ROOT_DIR"
gunicorn web_app.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --timeout 300

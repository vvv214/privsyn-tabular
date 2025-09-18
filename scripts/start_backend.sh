#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root and activate venv
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT_DIR/.venv/bin/activate"

# Decide between dev (uvicorn reload) and prod (gunicorn) modes based on flag
MODE=${1:-dev}
shift || true

cd "$ROOT_DIR"

if [[ "$MODE" == "prod" ]]; then
  gunicorn web_app.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker --timeout 300 "$@"
else
  uvicorn web_app.main:app --reload --port 8001 "$@"
fi

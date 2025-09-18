#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODE=${1:-dev}
shift || true

pushd "$ROOT_DIR/frontend" >/dev/null
if [ ! -d node_modules ]; then
  npm install
fi

if [[ "$MODE" == "prod" ]]; then
  npm run build "$@"
  popd >/dev/null
  rm -rf "$ROOT_DIR/web_app/static"
  mkdir -p "$ROOT_DIR/web_app/static"
  cp -r "$ROOT_DIR/frontend/dist/"* "$ROOT_DIR/web_app/static/"
else
  npm run dev "$@"
  popd >/dev/null
fi

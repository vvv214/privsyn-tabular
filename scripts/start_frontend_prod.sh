#!/usr/bin/env bash
set -euo pipefail

# Script to build the React frontend for production and copy to backend static directory

# Resolve repo root
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Install dependencies (if not already installed) and build
pushd "$ROOT_DIR/frontend" >/dev/null
npm install
npm run build
popd >/dev/null

# Replace web_app/static with built assets
rm -rf "$ROOT_DIR/web_app/static"
mkdir -p "$ROOT_DIR/web_app/static"
cp -r "$ROOT_DIR/frontend/dist/"* "$ROOT_DIR/web_app/static/"

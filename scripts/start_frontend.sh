#!/usr/bin/env bash
set -euo pipefail

# Script to start the React frontend

# Resolve repo root (venv activation optional for npm)
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Start the development server
cd "$ROOT_DIR/frontend"
npm run dev

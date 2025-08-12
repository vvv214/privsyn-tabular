#!/bin/bash
# Script to start the FastAPI backend

# Activate the virtual environment
source .venv/bin/activate

# Run the FastAPI application
uvicorn web_app.main:app --reload --port 8001

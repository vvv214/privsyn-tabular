#!/bin/bash

# Activate the virtual environment
source .venv/bin/activate

# Run the FastAPI application
uvicorn web_app.main:app --reload --port 8001

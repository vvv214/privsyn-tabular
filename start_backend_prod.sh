#!/bin/bash
# Script to start the FastAPI backend in production using Gunicorn

# Activate the virtual environment
source .venv/bin/activate

# Run the FastAPI application with Gunicorn
gunicorn web_app.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker

#!/bin/bash
# Script to start the FastAPI backend in production using Gunicorn

# Activate the virtual environment
source .venv/bin/activate

# Run the FastAPI application with Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker web_app.main:app --bind 0.0.0.0:8001

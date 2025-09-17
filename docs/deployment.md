# Deployment Guide

This document outlines the most common deployment paths for the PrivSyn web application—local Docker usage, Google Cloud Run, and Vercel (frontend). It highlights the environment variables you need to set and the expected directory layout.

## 1. Local Docker Image

1. Build the image:
   ```bash
   docker build -t privsyn-tabular .
   ```
2. Run the container, exposing the FastAPI port:
   ```bash
   docker run --rm -p 8080:8080 \
     -e VITE_API_BASE_URL="http://localhost:8080" \
     privsyn-tabular
   ```
3. The backend listens on `$PORT` (defaults to `8080` for Cloud Run compatibility). The Dockerfile bundles the built frontend under `web_app/static/`, so no additional proxy is required.

## 2. Google Cloud Run

1. Build and push the container:
   ```bash
   gcloud builds submit --tag gcr.io/<PROJECT_ID>/privsyn
   ```
2. Deploy to Cloud Run:
   ```bash
   gcloud run deploy privsyn \
     --image gcr.io/<PROJECT_ID>/privsyn \
     --platform managed \
     --allow-unauthenticated \
     --set-env-vars="VITE_API_BASE_URL=https://<service-url>"
   ```
3. Configure Cloud Run service variables:
   - `VITE_API_BASE_URL`: the public URL of the service if you intend to serve the frontend from the same container.
   - `CORS_ALLOW_ORIGINS`: optional comma-separated list of additional origins to append to the defaults in `web_app/main.py`.

## 3. Vercel Frontend + Hosted Backend

1. Deploy the backend (Docker, Cloud Run, or elsewhere) and note the public base URL.
2. In the Vercel project (or other static hosting provider):
   - Set `VITE_API_BASE_URL` to the backend URL.
   - Run `npm run build` to produce `frontend/dist`.
   - Serve the built assets or configure Vercel to use the static output directory.
3. Update `allow_origins` in `web_app/main.py` to include the Vercel domain.

## 4. Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `VITE_API_BASE_URL` | Frontend → backend URL. | `http://127.0.0.1:8001` |
| `CORS_ALLOW_ORIGINS` | Optional extra origins; comma separated. | — |
| `PORT` | FastAPI listening port (Cloud Run). | `8080` |
| `LOG_LEVEL` | Optional override for FastAPI logging level. | `INFO` |

## 5. Storage & Sessions

- Temporary artifacts (uploaded parquet files, synthesized CSVs) are stored under `temp_synthesis_output/runs/{session_id}`.
- Sessions expire automatically after six hours. For production, consider pointing the session store to Redis or another durable cache.
- Use a cron or Cloud Run job to prune `temp_synthesis_output` if you retain disk between deployments.

## 6. Health Checks

- Use `GET /` for a lightweight ping.
- To simulate the metadata flow without a real upload, POST to `/synthesize` with `dataset_name=debug_dataset` (returns stub metadata).

Refer to `docs/testing.md` for CI-friendly commands to verify the deployment image before shipping.

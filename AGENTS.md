# Repository Guidelines

## Project Structure & Module Organization
- backend core: `privsyn/` (DP synthesis algorithm), `preprocess_common/` (preprocessing), `util/` (helpers).
- web app: `web_app/` (FastAPI backend), `frontend/` (Vite + React UI).
- tests & assets: `test/` (pytest tests), `sample_data/` (example inputs).
- tooling: `requirements.txt`, `Dockerfile`, `start_backend*.sh`, `start_frontend*.sh`.

## Build, Test, and Development Commands
- setup (one‑time): `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- backend (dev): `uvicorn web_app.main:app --reload --port 8001` (or `./start_backend.sh`).
- frontend (dev): `cd frontend && npm install && npm run dev -- --port 5174` (or `./start_frontend.sh`).
- tests (python): `pytest test/` (use `.venv/bin/pytest` if needed).
- frontend build: `cd frontend && npm run build` (or `./start_frontend_prod.sh`).
- backend (prod): `gunicorn web_app.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker` (or `./start_backend_prod.sh`).

## Coding Style & Naming Conventions
- python: PEP 8, 4‑space indent, `snake_case` for modules/functions, `CamelCase` for classes, add docstrings and type hints where helpful. Prefer `logging` over prints.
- frontend: React function components and hooks; component files `PascalCase.jsx/tsx`. Run `npm run lint` before PRs.
- naming: tests as `test_*.py`; keep directories lowercase with underscores.

## Testing Guidelines
- framework: `pytest` under `test/` (e.g., `test_preprocessing.py`, `test_synthesis.py`).
- run locally: `pytest -q` or `.venv/bin/pytest test/`.
- guidance: add unit tests for preprocessing and synthesis paths; use small synthetic DataFrames. Name tests descriptively and keep them deterministic.

## Commit & Pull Request Guidelines
- commits: concise, imperative; prefer Conventional Commits (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`). Example: `feat: support CSV upload and metadata inference`.
- pull requests: include purpose, linked issues, how to test (commands and sample inputs), and screenshots for UI changes. Ensure `pytest` passes and the frontend lints/builds.

## Security & Configuration Tips
- config: set `VITE_API_BASE_URL` in the frontend for the deployed backend URL. Update CORS `allow_origins` in `web_app/main.py` for new domains.
- data handling: avoid committing datasets or secrets; uploads are processed to temporary files—clean up on errors too.
- resources: synthesis can be memory‑intensive; reduce `n_sample` and watch backend logs if constrained.


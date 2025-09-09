# Repository Guidelines

## Project Structure & Module Organization
- `privsyn/`: Core PrivSyn algorithm and submodules.
- `preprocess_common/`: Data loading and preprocessing utilities.
- `web_app/`: FastAPI backend (`web_app/main.py`, services, metrics).
- `frontend/`: React + Vite UI (`frontend/src/*`).
- `util/`: Shared helpers (e.g., privacy accounting).
- `test/`: Pytest-based tests; see `test/TESTING_GUIDE.md`.
- `sample_data/`: Small datasets for local trials.
- Scripts: `start_backend.sh`, `start_frontend.sh`, `start_*_prod.sh`; container spec in `Dockerfile`.

## Build, Test, and Development Commands
- Backend (dev): `uvicorn web_app.main:app --reload --port 8001` or `./start_backend.sh`.
- Frontend (dev): `cd frontend && npm install && npm run dev -- --port 5174` or `./start_frontend.sh`.
- Tests: `pytest test/` (run from repo root). See `test/TESTING_GUIDE.md` for examples.
- Lint (frontend): `cd frontend && npm run lint`.
- Prod build (frontend → static): `./start_frontend_prod.sh` copies `frontend/dist` into `web_app/static`.
- Container: `docker build -t privsyn-tabular .` then run with `-p 8080:8080` (Cloud Run expects `$PORT`).

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indents, snake_case for functions/vars, PascalCase for classes, UPPER_CASE for constants. Prefer type hints and concise docstrings.
- React: Components PascalCase (e.g., `MetadataConfirmation.jsx`), hooks `useX`. Keep stateful logic in components; utilities in `frontend/src` helpers.
- Keep modules small and cohesive; avoid cyclic imports across `privsyn/*`.

## Testing Guidelines
- Framework: `pytest` (unit/integration under `test/`, files like `test_*.py`).
- Run: `pytest -q` or target a file (e.g., `pytest test/test_synthesis.py`).
- Add deterministic seeds where applicable; avoid relying on network or large external data. Prefer `sample_data/` for fixtures.

## Commit & Pull Request Guidelines
- Use Conventional Commits: `feat:`, `fix:`, `chore(scope):`, `docs(frontend): ...` to match existing history.
- PRs: include a clear summary, motivation, linked issues, test plan (commands + expected output), and screenshots for UI changes.
- Keep diffs focused. Update README/config docs when changing env vars or endpoints.

## Security & Configuration Tips
- Frontend API URL: set `VITE_API_BASE_URL` (Vercel/locally) to your backend.
- CORS: update `allow_origins_list` in `web_app/main.py` for new domains.
- Do not commit secrets or large private datasets. Use `.env.local` and `.venv/`.
- Performance: watch backend logs for memory usage; lower `n_sample` when testing locally.

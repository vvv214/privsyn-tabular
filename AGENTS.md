# Repository Guidelines

## Project Structure & Module Organization
- Core synthesis methods live under `method/` (`privsyn/`, `AIM/`) with adapters exposing consistent `prepare`/`run` hooks.
- Backend FastAPI service sits in `web_app/` (entrypoint `web_app/main.py`, dispatcher `methods_dispatcher.py`).
- React UI is in `frontend/` with Vite tooling; static builds copy into `web_app/static/`.
- Shared preprocessing utilities are under `method/preprocess_common/`; reconstruction helpers in `method/reconstruct_algo/`.
- Tests reside in `test/` (unit/integration) and `test/e2e/`; sample fixtures under `sample_data/`.

## Build, Test, and Development Commands
- `uvicorn web_app.main:app --reload --port 8001` – run backend in dev mode (`./start_backend.sh` wraps this).
- `cd frontend && npm install && npm run dev -- --port 5174` – start the frontend; `./start_frontend.sh` is equivalent.
- `pytest` or `pytest -q` – execute Python unit/integration tests; see `test/TESTING_GUIDE.md` for focus options.
- `cd frontend && npm run lint` – lint React codebase.
- `pytest --cov=. --cov-report=xml` – coverage run expected in CI; keep ~77%+ overall.

## Coding Style & Naming Conventions
- Python follows PEP 8, 4-space indents, snake_case for functions/variables, PascalCase for classes, `UPPER_CASE` constants; add type hints and concise docstrings.
- React components use PascalCase; hooks named `useX`. Keep stateful logic in components and share utilities via `frontend/src` helpers.
- Prefer ASCII in source files; avoid cyclic imports across `web_app/` and `method/` modules.

## Testing Guidelines
- Use `pytest` with deterministic seeds where possible; E2E flows live in `test/e2e/` and are marked `e2e`.
- New features should extend coverage-aligned suites (e.g., `test/test_metadata_overrides.py`, `test/test_privsyn_domain_utils.py`).
- Run `pytest -q -W error::DeprecationWarning -W error::FutureWarning -k "web_app or method/preprocess_common or method/privsyn"` to enforce strict warning handling on core modules.

## Commit & Pull Request Guidelines
- Follow Conventional Commits (`feat:`, `fix:`, `chore(scope):`, etc.).
- PRs should explain motivation, link issues, list test commands, and attach screenshots for UI changes.
- Update docs/config when adjusting env vars or endpoints; keep diffs scoped and avoid committing secrets.

## Security & Configuration Tips
- Configure API base URL via `VITE_API_BASE_URL`; adjust backend CORS origins in `web_app/main.py` for new domains.
- Keep temporary synthesis outputs in git-ignored `temp_synthesis_output/` and avoid adding large private datasets.

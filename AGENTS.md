# Repository Guidelines

## Project Structure & Module Organization
- `method/privsyn/`: Authoritative PrivSyn implementation and submodules.
- `method/AIM/`: AIM (Adaptive and Iterative Mechanism) implementation.
- `preprocess_common/`: Data loading and preprocessing utilities.
- `reconstruct_algo/`: Root-level helpers to reconstruct decoded DataFrames from encoded parts.
- `web_app/`: FastAPI backend (`web_app/main.py`, services, dispatcher).
- `frontend/`: React + Vite UI (`frontend/src/*`).
- `util/`: Shared helpers (e.g., privacy accounting `rho_cdp`).
- `test/`: Pytest tests; end-to-end tests live in `test/e2e/`.
- `sample_data/`: Small datasets for local trials.
- Scripts: `start_backend.sh`, `start_frontend.sh`, `start_*_prod.sh`; container spec in `Dockerfile`.

Notes:
- A legacy `privsyn/temp_data/` may exist only for temporary artifacts; algorithm code lives under `method/privsyn/`.

## Build, Test, and Development Commands
- Backend (dev): `uvicorn web_app.main:app --reload --port 8001` or `./start_backend.sh`.
- Frontend (dev): `cd frontend && npm install && npm run dev -- --port 5174` or `./start_frontend.sh`.
- Tests: `pytest` (run from repo root). See `test/TESTING_GUIDE.md` for examples and e2e setup.
- Lint (frontend): `cd frontend && npm run lint`.
- Prod build (frontend → static): `./start_frontend_prod.sh` copies `frontend/dist` into `web_app/static`.
- Container: `docker build -t privsyn-tabular .` then run with `-p 8080:8080` (Cloud Run expects `$PORT`).

## Methods Integration
- Methods are exposed behind lightweight adapters to present a consistent interface to the backend and tests.
  - Adapters live under `method/<MethodName>/adapter.py` and implement `prepare(...)` and `run(...)` returning a decoded `pd.DataFrame` with original column order/types.
  - Backend dispatch is centralized in `web_app/methods_dispatcher.py`.
- New methods should integrate via an adapter first to minimize core changes; once stable, consider refactoring the method to natively implement the unified API.

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indents, snake_case for functions/vars, PascalCase for classes, UPPER_CASE for constants. Prefer type hints and concise docstrings.
- React: Components PascalCase (e.g., `MetadataConfirmation.jsx`), hooks `useX`. Keep stateful logic in components; utilities in `frontend/src` helpers.
- Keep modules small and cohesive; avoid cyclic imports across `method/*` and `web_app/*`.

## Testing Guidelines
- Framework: `pytest` (unit/integration under `test/`, files like `test_*.py`). E2E tests live under `test/e2e/` and are marked `e2e`.
- Run: `pytest -q` or target a file (e.g., `pytest test/test_synthesis.py`).
- Determinism: Seed RNGs where feasible; avoid network or large external data. Prefer `sample_data/` for fixtures.
- Warnings policy: We suppress noisy warnings from heavy third‑party method folders in `pytest.ini` to keep core warnings visible. See below for strict mode.

## Commit & Pull Request Guidelines
- Use Conventional Commits: `feat:`, `fix:`, `chore(scope):`, `docs(frontend): ...`.
- PRs: include a clear summary, motivation, linked issues, test plan (commands + expected output), and screenshots for UI changes.
- Keep diffs focused. Update README/config docs when changing env vars or endpoints.

## Security & Configuration Tips
- Frontend API URL: set `VITE_API_BASE_URL` (Vercel/locally) to your backend.
- CORS: update allowed origins in `web_app/main.py` for new domains.
- Do not commit secrets or large private datasets. Use `.env.local` and `.venv/`.
- Performance: watch backend logs for memory usage; lower `n_sample` when testing locally.

## Warnings & Strict Mode
- Pytest filters in `pytest.ini` suppress Deprecation/Future/Runtime warnings from `method/AIM`, `GEM`, `TabDDPM`, `DP_MERF`, `private_gsd`, and `reconstruct_algo`.
- Core project warnings (e.g., under `web_app/`, `preprocess_common/`, `method/privsyn/`) remain visible.
- To enforce strictness for core code locally or in CI, you can run:
  - `pytest -q -W error::DeprecationWarning -W error::FutureWarning -k 'web_app or preprocess_common or method/privsyn'`

## Scratch Artifacts
- Do not add scratch or exploratory files to the repo. Convert valuable experiments into tests under `test/` or notebooks ignored by git. Place E2E flows under `test/e2e/`.

## Current Context (2024-xx)
- Metadata confirmation now supports per-column overrides (categorical value selection, numerical bounds/binning). Backend defers encoding until overrides are applied.
- Evaluation (`/evaluate`) uses metadata-aware TVD: numerical columns leverage histogram bins (prefer domain-provided edges) while categorical columns retain exact TVD.
- When a column without numeric values is forced to numeric, backend generates deterministic pseudo-random values within the provided bounds to avoid degenerate outputs; UI warns the user about the coercion.
- Tests: `test/test_metadata_overrides.py`, `test/test_data_inference.py`, and `test/test_data_comparison.py` cover the new flows. Focused coverage command is documented in README.
- Temporary run artifacts (`temp_synthesis_output/`, `temp_bench_runs/`) stay git-ignored; the helper scripts under `scripts/` run dev/prod servers, benches, and E2E.

### 2025-09-16 Session Notes
- CI now runs `pytest --cov=. --cov-report=xml` and uploads results via `codecov/codecov-action@v4`; coverage badge embedded in README.
- Added deterministic unit suites for PrivSyn internals and discretizers:
  - `test/test_privsyn_domain_utils.py`, `test/test_privsyn_update_config.py`, `test/test_privsyn_records_update.py`.
  - `test/test_privtree.py`, `test/test_dawa.py`.
  - Expanded backend override checks in `test/test_metadata_overrides.py` and path handling in `test/test_datastore.py`.
- With these tests, coverage is ~77% overall; PrivSyn modules and preprocessors (PrivTree/DAWA) now exceed 80% coverage.

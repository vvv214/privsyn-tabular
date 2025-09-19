# Repository Guidelines

## Project Structure & Module Organization
- Core synthesis methods now live under `method/synthesis/` (e.g., `privsyn/`, `AIM/`) with adapters exposing consistent `prepare`/`run` hooks.
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
- Run `pytest -q -W error::DeprecationWarning -W error::FutureWarning -k "web_app or method/preprocess_common or method/synthesis/privsyn"` to enforce strict warning handling on core modules.

## Commit & Pull Request Guidelines
- Follow Conventional Commits (`feat:`, `fix:`, `chore(scope):`, etc.).
- PRs should explain motivation, link issues, list test commands, and attach screenshots for UI changes.
- Update docs/config when adjusting env vars or endpoints; keep diffs scoped and avoid committing secrets.

## Security & Configuration Tips
- Configure API base URL via `VITE_API_BASE_URL`; adjust backend CORS origins in `web_app/main.py` for new domains.
- Keep temporary synthesis outputs in git-ignored `temp_synthesis_output/` and avoid adding large private datasets.
- PrivSyn temp artifacts default to the system temp dir; override via `PRIVSYN_DATA_ROOT` / `PRIVSYN_EXP_ROOT` when automation needs deterministic paths.

## Latest Fixes & Notes
- Fixed MinMaxScaler casting so `num_preprocess='none'` preserves numeric variation; see `test/test_preprocessing.py::test_numeric_preprocessing_none_preserves_variation`.
- Respected user-specified numeric bin edges when preprocessing so post-transform discretisation matches confirmed domains (`method/preprocess_common/load_data_common.py`).
- PrivSyn now seeds via context-managed RNG to avoid mutating global NumPy state (`method/synthesis/privsyn/native.py`); regression covered in `test/test_privsyn_rng.py`.
- Frontend sample gating only bypasses uploads when the dataset name is exactly `adult`; unit tested in `frontend/src/components/SynthesisForm.test.jsx`.
- Session store supports eviction callbacks, enabling automatic temp-dir cleanup for expired inference sessions (`web_app/session_store.py`).
- CSV/ZIP uploads stream directly from the FastAPI `UploadFile` object, avoiding eager `read()` into memory (`web_app/data_inference.py`).
- API no longer requires or strips a `target_column`; the entire uploaded table participates in metadata inference and synthesis.
- Centralised DP noise helpers live in `method/util/dp_noise.py`; PrivSyn, preprocessing discretisers, PrivTree, PrivMRF, and GEM now rely on these wrappers for all DP noise draws.
- AIM fit/sample now use the same temporary NumPy seed context as PrivSyn to avoid mutating global RNG state (`method/synthesis/AIM/native.py`).
- Server-side validation rejects mismatched numeric bin edges/bin counts so UI overrides cannot desynchronise run configs (`web_app/main.py`).

### Suggested Test Commands
- `pytest -q test/test_preprocessing.py test/test_privsyn_rng.py test/test_session_store.py test/test_metadata_overrides.py`
- `cd frontend && npm test -- SynthesisForm.test.jsx`

## Working Mental Model
- **User flow**: React form (`SynthesisForm` → `MetadataConfirmation` → `ResultsDisplay`) collects inputs, posts to `/synthesize`, confirms metadata, then downloads CSV and triggers `/evaluate` for TVD metrics.
- **Inference phase**: `/synthesize` loads CSV/ZIP, infers column domain info via `web_app.data_inference`, writes parquet + metadata to a temp dir stored in `SessionStore` until the user confirms or TTL expires.
- **Confirmation & synthesis**: `/confirm_synthesis` merges user overrides, rebuilds numeric/categorical matrices, persists run artifacts, and calls `run_synthesis` which dispatches through `method.api.SynthRegistry` (default PrivSyn/AIM adapters).
- **PrivSyn adapter**: `method/synthesis/privsyn/native.py` preprocesses with `data_preporcesser_common`, constructs the PrivSyn core, and samples via `FittedPrivSyn` while keeping RNG state scoped.
- **Evaluation**: `/evaluate` reloads the synthesized CSV alongside cached original data, computing TVD by column (numeric bins honored when provided).
- **Session lifecycle**: `SessionStore` imposes TTL, runs eviction callbacks to clean temp directories, and keeps synthesized outputs available for download/evaluation for six hours.

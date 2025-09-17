# Testing Guide

## Test Layers

| Layer | Path | Description |
|-------|------|-------------|
| Unit / Integration | `test/test_*.py` | Covers API contracts, PrivSyn helpers, preprocessors, etc. |
| Frontend unit | `frontend/src/**/*.test.jsx` | Vitest + React Testing Library exercises UI validation (e.g., `MetadataConfirmation`). |
| End-to-End | `test/e2e/` | Playwright flows that boot both frontend and backend. |
| Frontend lint | `frontend/eslint` | Run via `npm run lint`. |

Markers & conventions:
- `slow`: long-running algorithmic tests. Skip with `-m "not slow"` for fast feedback.
- `e2e`: Playwright browser automation. Enable with `E2E=1` to avoid booting browsers by default.

## Quick Commands

```bash
# Fast unit/integration loop
pytest -q -m "not slow"

# Full suite with coverage
pytest --cov=. --cov-report=term

# Frontend component tests
cd frontend
npm test -- --run

# Run only API contract tests
pytest -q test/test_api_contract.py

# Playwright end-to-end (requires Playwright browsers & npm deps)
E2E=1 pytest -q -k end_to_end
```

## What’s Covered?

- **Metadata overrides** (`test/test_metadata_overrides.py`): round-trip of `/synthesize` → `/confirm_synthesis`, categorical resample strategy, numeric coercion, and exponential binning scenarios.
- **PrivSyn internals** (`test/test_privsyn_*`): domain helpers, update config, records update, marginal selection.
- **Discretizers** (`test/test_privtree.py`, `test/test_dawa.py`): deterministic unit tests for PrivTree and DAWA helpers.
- **API contracts** (`test/test_api*.py`): unified synthesizer interface, metrics hook, deterministic sampling semantics.
- **End-to-end** (`test/e2e/test_frontend_backend_e2e.py`): upload sample → confirm metadata (category fallback + validation) → synthesize → download.
- **Frontend components** (`frontend/src/components/*.test.jsx`): Vitest suites covering metadata confirmation, categorical editing, and numeric bound validation warnings.

## Local Tips

- Install Playwright browsers once with `python -m playwright install --with-deps` if you plan to run E2E.
- Ensure ports 8001 (backend) and 5174 (frontend) are free before running `E2E=1` tests; the suite spawns both servers automatically.
- Browser tests default to Chromium; you can change `p.chromium.launch()` to other browsers if needed.

## CI Notes

- GitHub Actions workflow `ci.yml` runs the backend tests with coverage and uploads to Codecov when `CODECOV_TOKEN` is configured.
- E2E job (`jobs.e2e`) is opt-in—triggered by setting `E2E=1 pytest -q -k e2e` after installing Playwright dependencies.

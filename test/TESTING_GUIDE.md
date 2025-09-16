# Testing Guide

This document describes how the PrivSyn tabular project is tested, the scope of
each suite, and practical tips for adding new coverage. The goal is to make the
entire synthesis loop – from metadata inference to downloadable CSVs – easy to
validate at different depths without running heavy experiments.

## Test Matrix Overview

| Layer | Files | Focus |
|-------|-------|-------|
| **Unit / Fast integration** | `test/test_*.py` | Algorithms and services in isolation with minimal IO.|
| **Adapter coverage** | `test/test_aim_adapter*.py`, `test/test_methods_dispatcher.py` | Ensures each synthesis method conforms to the backend contract. |
| **API contract** | `test/test_api_*.py` | FastAPI endpoints, request/response validation, temporary storage. |
| **Algorithm behaviour** | `test/test_privsyn_*.py`, `test/test_synthesis_service.py` | PrivSyn marginal selection, parameter parsing, synthesis plumbing. |
| **UI + backend E2E** | `test/e2e/*.py` | Launches Uvicorn + Vite and drives the React UI via Playwright. Skipped unless `E2E=1`. |

Markers:

- `slow`: guard long-running algorithm suites. Skip with `-m "not slow"`.
- `e2e`: Playwright/browser tests. Enable with `E2E=1`.

## Running the Suites

All commands assume the project’s virtual environment is active (`source .venv/bin/activate`). Replace `pytest` with `.venv/bin/pytest` if you want to bypass `$PATH`.

### Core smoke

```bash
pytest -q
```

### Targeted areas

```bash
# Metadata overrides, inference, and preprocessing
pytest -q test/test_metadata_overrides.py test/test_data_inference.py test/test_preprocessing.py

# REST API contract including upload and download flow
pytest -q -k api

# PrivSyn algorithm internals
pytest -q test/test_privsyn_*.py
```

### End-to-end (Playwright)

Run when you need to exercise the real UI:

```bash
pip install -r requirements.txt
python -m playwright install
cd frontend && npm install && cd ..
E2E=1 pytest -q -k e2e
```

This spins up Uvicorn (port 8001) and the Vite dev server (port 5174). Make sure those ports are available before running. The E2E suite currently contains:

- `test/e2e/test_frontend_backend_e2e.py::test_end_to_end_frontend_backend`
  – uploads `sample_data/adult.csv.zip`, walks through metadata confirmation, waits for synthesis, and checks the download link and preview table.
- `test/e2e/test_ui_smoke.py::test_ui_smoke`
  – ensures the landing page renders headline copy and captures a screenshot under `test/e2e/artifacts/`.

## What Each Test Family Covers

### Metadata and preprocessing

- `test/test_metadata_overrides.py`, `test/test_data_inference.py`, `test/test_preprocessing.py`
  validate the inferred domain, numeric/categorical coercion, custom bounds, binning strategies, and error handling (e.g., inconsistent CSV shapes). Recent additions such as
  `test_categorical_resample_strategy_persists_domain`, `test_force_categorical_to_numeric_generates_range`, and
  `test_numerical_exponential_binning_edges_are_monotonic` exercise variations of the PrivSyn confirmation flow by driving `/synthesize` → `/confirm_synthesis` end-to-end with different overrides.
- `test/test_privtree.py`, `test/test_dawa.py` cover the numerical discretizers leveraged by the preprocessing pipeline (PrivTree and DAWA). They patch randomness for determinism and assert that binning/inversion round-trips remain within bounds.

### API layer

- `test/test_api_contract.py`, `test/test_api_integration.py`, `test/test_api_sample.py`, `test/test_api_smoke.py`, and `test/test_api_utils.py` exercise the FastAPI endpoints directly. They verify multipart parsing, temporary storage lifetimes, sample dataset download, and evaluation payloads. The tests stub uploaded files using in-memory streams to keep them fast.

### Data store & reconstruction helpers

- `test/test_datastore.py`, `test/test_datastore_paths_created.py`, and
  `test/test_reverse_data_paths.py` ensure intermediate parquet/npz paths are created and cleaned correctly.
- `test/test_domain_ops.py` and `test/test_data_comparison.py` assert that marginal computations and domain manipulation utilities stay consistent with expectations.

### PrivSyn pipeline

- `test/test_privsyn_dataset.py`, `test/test_privsyn_domain_alignment.py`,
  `test/test_privsyn_marginal_selection.py`, `test/test_privsyn_parameter_parser.py`, and `test/test_privsyn_static_noise.py` target the core algorithm. They cover marginal selection heuristics, consistency enforcement, parameter parsing defaults, and noise calibration. Together they guard regressions in the iterative update loop described in the PrivSyn paper.
- `test/test_synthesis.py` and `test/test_synthesis_service.py` focus on the orchestration layer: preparing numpy arrays, instantiating adapters, and persisting synthesized CSVs.

### AIM adapter

- `test/test_aim_adapter.py` and `test/test_aim_adapter_dtypes.py` ensure the AIM implementation respects the unified adapter API: categorical encoding, numpy arrays, and dtype preservation. These tests use tiny toy data to keep runtime short.

### End-to-end smoke

See the previous section – they confirm that the UI and API wiring still work when combined.

## Writing New Tests

1. **Pick the narrowest layer** that exercises the behaviour. Most regressions can be caught in unit/integration tests without launching Playwright.
2. **Seed randomness** (`numpy.random.default_rng`, `random.seed`) so results stay deterministic.
3. **Use sample fixtures** in `sample_data/` (e.g., `adult.csv.zip`) or build tiny DataFrames inline rather than large datasets.
4. **Avoid heavy iterations** in algorithm tests. For PrivSyn or AIM coverage, lower `update_iterations`, `n_sample`, or similar knobs; these are already parameterised in helper fixtures such as `MINIMAL_PRIVSYN_ARGS` used across the unit suite.
5. **Keep outputs small and in-memory**. Write to `tmp_path` provided by pytest instead of repo paths, and clean up temp folders created during the test.
6. **For new Playwright flows**, ensure you gate them behind `pytest.mark.e2e` and guard selectors with generous waits; rely on `page.route` to intercept backend requests when you only need to validate payloads instead of waiting for long syntheses.

## Pipeline Cheat Sheet

### PrivSyn

1. **Metadata inference (`/synthesize`)** – detects numeric vs categorical columns using heuristics (integer uniqueness thresholds, float variance) and produces `domain.json` + `info.json` previews.
2. **User confirmation** – the frontend allows per-column overrides (clip bounds, binning strategy, unexpected categorical handling). Confirms via `/confirm_synthesis`.
3. **Preprocessing** – numerical columns are clipped/bucketed, categorical columns remapped, then encoded into numpy arrays.
4. **Marginal selection & updates** – PrivSyn iteratively selects marginals, injects calibrated noise (rho-CDP accounting), and refines a synthetic distribution.
5. **Sampling & evaluation** – generates a synthetic CSV, stores metadata for evaluation, and provides download + fidelity metrics (histogram-aware TVD).

Fast tests mostly cover steps 1–4 with deterministic fixtures; the e2e test confirms the wiring for steps 1–5 end to end.

### AIM (Adaptive and Iterative Mechanism)

1. **Preprocessing / encoding** – reuses the same metadata pipeline as PrivSyn through the adapter layer.
2. **Adaptive query selection** – AIM iteratively chooses workload-aware measurements, balancing accuracy and privacy cost (as detailed in the AIM PVLDB paper).
3. **Measurement & reconstruction** – queries are answered with DP noise, then a synthetic dataset is generated that matches those measurements.

Current unit tests (`test/test_aim_adapter*.py`) verify that our adapter passes the correctly encoded numpy arrays and that dtype handling matches expectations. When adding additional AIM tests, prefer stubbing the workload or lowering iteration counts to keep runtime reasonable.

## Suggested Workflow for Contributors

1. Run `pytest -q` before submitting changes.
2. If modifying API contracts or metadata flows, run targeted tests listed above.
3. Only run `E2E=1 pytest -q -k e2e` when touching UI-to-backend flows or deployment wiring.
4. Update this guide whenever you add a new suite or change test prerequisites.

Following this structure keeps day-to-day development fast while still giving confidence that both the PrivSyn and AIM pipelines behave as described in their respective papers.

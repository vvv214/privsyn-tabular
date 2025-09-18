# Backend Guide

## Architecture Overview

- **FastAPI application**: `web_app/main.py`
  - `/synthesize` infers metadata and caches the original DataFrame plus draft domain info.
  - `/confirm_synthesis` reconstructs the DataFrame with user overrides and invokes the synthesizer.
  - `/download_synthesized_data/{session_id}` streams the generated CSV for a confirmed synthesis session.
  - `/evaluate` calculates metadata-aware metrics using `web_app/data_comparison.py` (keyed by the same session ID).
- **Synthesis service**: `web_app/synthesis_service.py`
  - Bridges the cached inference bundle to the selected synthesizer.
  - Handles preprocessing (clipping, binning, categorical remap) before handing off to PrivSyn or AIM.

## Key Modules

| Module | Role |
|--------|------|
| `web_app/data_inference.py` | Detect column types, normalise metadata, and prepare draft domain/info payloads. |
| `web_app/synthesis_service.py` | Applies overrides, constructs the preprocesser, runs the synthesizer, and persists outputs. |
| `web_app/data_comparison.py` | Implements histogram-aware TVD and other metrics for evaluation. |
| `method/synthesis/privsyn/privsyn.py` | PrivSyn implementation (marginal selection + GUM). |
| `method/api/base.py` | Core synthesizer API (`SynthRegistry`, `PrivacySpec`, `RunConfig`, `Synthesizer` protocol). |
| `method/api/utils.py` | Helper utilities used by adapters (e.g., `split_df_by_type`, schema enforcement). |
| `method/synthesis/AIM/adapter.py` | Adapter wiring AIM into the unified interface provided by `method/api`. |
| `method/preprocess_common/` | Shared discretizers (PrivTree, DAWA) and helper utilities. |

### Unified Synthesis Interface

`method/api/base.py` defines the shared contract every synthesis method must follow:

- `SynthRegistry` exposes `register`, `get`, and `list` helpers so adapters (e.g., `method/synthesis/privsyn/__init__.py`, `method/synthesis/AIM/__init__.py`) can self-register at import time.
- `PrivacySpec` and `RunConfig` capture the caller’s DP/compute requirements and are passed through to each adapter.
- `_AdapterSynth` and `_AdapterFitted` wrap legacy prepare/run functions so existing method code needs minimal changes.

The backend dispatcher (`web_app/methods_dispatcher.py`) and tests such as `test/test_methods_dispatcher.py` rely on this registry to treat every method uniformly. Method-specific modules (`method/synthesis/<name>/native.py`, `config.py`, `parameter_parser.py`, etc.) stay alongside each algorithm because they encode behaviour that other methods do not share (e.g., PrivSyn’s marginal-selection parameters or AIM’s workload configuration). Keep the registry small and general, and let each method own its internal configuration files.

## Endpoint Notes

### POST `/synthesize`
- Expects multipart form (fields documented in `test/test_api_contract.py`).
- For sample runs, omit the file and set `dataset_name=adult`.
- Stores the uploaded DataFrame and inferred metadata under a temporary UUID in memory.

### POST `/confirm_synthesis`
- Requires the `unique_id` returned by `/synthesize`.
- Accepts JSON strings for `confirmed_domain_data` and `confirmed_info_data`.
- Runs the chosen synthesizer (`privsyn` or `aim`) and writes synthesized CSV + evaluation bundle to the temp directory.

### GET `/download_synthesized_data/{session_id}`
- Streams the generated CSV for a previously confirmed synthesis session.
- Backed by an in-memory `SessionStore` keyed by the `unique_id` returned from `/synthesize`.

### POST `/evaluate`
- Accepts `session_id` (form field) and reuses cached original/synth data to compute metrics (e.g., histogram TVD for numeric columns).

## Local Development

```bash
uvicorn web_app.main:app --reload --port 8001

# Optionally set VITE_API_BASE_URL when running the frontend separately
export VITE_API_BASE_URL=http://127.0.0.1:8001
```

## Configuration Tips

- CORS origins are defined in `web_app/main.py`. Update the `allow_origins` list to include any new frontend domains.
- Set the `ADDITIONAL_CORS_ORIGINS` environment variable (comma-separated list) in production to append extra origins—useful for Vercel preview/prod URLs.
- Temporary artifacts (original data, synthesized CSVs) land under `temp_synthesis_output/`. Keep an eye on disk usage during iterative testing.
- Use environmental overrides or `.env` files for production secrets (database URLs, etc.)—the current setup only handles the stateless demo flow.

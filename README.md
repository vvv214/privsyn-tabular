# PrivSyn - Differentially Private Data Synthesis 

![Coverage](https://codecov.io/gh/vvv214/privsyn-tabular/branch/main/graph/badge.svg) ![CI](https://github.com/vvv214/privsyn-tabular/actions/workflows/ci.yml/badge.svg) ![License](https://img.shields.io/github/license/vvv214/privsyn-tabular) ![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)

> End-to-end tooling for creating differentially private tabular datasets with PrivSyn and AIM.

## Highlights

- **Unified web app.** Upload a dataset, review inferred metadata, tweak categorical/numerical encodings, and download synthesized data in minutes.
- **Two synthesis engines.** PrivSyn (rho-CDP, iterative marginal updates) and AIM (adaptive measurement selection) exposed behind the same API.
- **Notebook-friendly modules.** Reusable preprocessing (PrivTree, DAWA), marginal selection, and synthesis utilities under `method/synthesis/privsyn` and `method/preprocess_common`.
- **Coverage-first test suite.** 100+ pytest cases plus Playwright E2E flows keep the UI/back-end contract in check.
- **MkDocs documentation.** Browse the Markdown guides with `mkdocs serve` (install via `pip install mkdocs`) and open <http://127.0.0.1:8000/> for a structured site.

## Quick Start

### Clone & setup

```bash
git clone https://github.com/vvv214/privsyn-tabular.git
cd privsyn-tabular
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Backend (FastAPI)

```bash
uvicorn web_app.main:app --reload --port 8001
```

The API is now live at `http://127.0.0.1:8001`. Docs: `http://127.0.0.1:8001/docs`.

### Frontend (Vite + React)

```bash
cd frontend
npm install
npm run dev -- --port 5174
```

Visit `http://127.0.0.1:5174`. The frontend defaults to `http://127.0.0.1:8001` unless `VITE_API_BASE_URL` is set.

### Try the sample dataset

1. Launch both servers.
2. Open the app and click **Load Sample** (loads `adult.csv.zip`).
3. Confirm metadata (tweak domains if desired) and click **Confirm & Synthesize**.
4. Download the resulting CSV or explore the preview table.
5. Want a quick preview without running anything? Check `docs/sample_output/` for a tiny synthetic CSV and matching metrics JSON.

## Architecture Diagrams

<p><em>All diagrams live in <code>docs/media/</code>; replace or expand them to match your deployment.</em></p>

### End-to-end sequence

![Sequence](docs/media/sequence.svg)

Shows how the user-facing frontend interacts with the FastAPI backend: upload → metadata inference → confirmation → synthesis → download.

### PrivSyn pipeline

<img src="docs/media/privsyn.svg" alt="PrivSyn pipeline" width="600" />

Highlights the preprocessing stage (metadata normalisation), the PrivSyn core (marginal selection + GUM), and post-processing steps (storage/evaluation).

### AIM workflow

![AIM workflow](docs/media/aim.svg)

Summarises AIM’s adaptive measurement loop: initialise workload → iteratively measure queries with DP noise → update the model → generate synthetic data.

### Request flow

![Request flow](docs/media/flow.svg)

Illustrates the main request/response boundaries between frontend, backend, synthesis engines, and temporary storage.

## Testing

```bash
# Fast feedback (skips slow markers)
pytest -q -m "not slow"

# Full suite with coverage
pytest --cov=. --cov-report=term

# Frontend component tests
cd frontend
npm test -- --run

# Browser E2E flows (requires Playwright browsers & frontend deps)
E2E=1 pytest -q -k e2e
```

Useful snippets:

```bash
# Metadata / preprocessing focus
pytest -q test/test_metadata_overrides.py test/test_preprocessing.py test/test_data_inference.py

# Strict warnings for core modules
pytest -q -W error::DeprecationWarning -W error::FutureWarning -k "web_app or method/preprocess_common or method/synthesis/privsyn"
```

## Documentation Site

- GitHub Actions now build MkDocs output on every push to `main` (see `.github/workflows/deploy-docs.yml`) and publish the result to the `gh-pages` branch.
- On Vercel, create a second project that points to this repository’s `gh-pages` branch with output directory `.` (no build command required). The project URL will serve the docs directly.
- Update `vercel.json`’s rewrite target (`https://<replace-with-docs-project>.vercel.app/`) once you know the docs project URL so your primary site proxies `/doc/*` to the generated documentation.
- Local preview remains available with `mkdocs serve`.

## Repository Layout

| Path | Description |
|------|-------------|
| `frontend/` | React + Vite SPA (metadata review UI, synthesis form, results view). |
| `web_app/` | FastAPI backend: metadata inference, synthesis orchestration, evaluation endpoints. |
| `method/synthesis/privsyn/` | Authoritative PrivSyn implementation (marginal selection, GUM synthesis, dataset helpers). |
| `method/api/` | Unified synthesizer interface (`SynthRegistry`, `PrivacySpec`, `RunConfig`) that lets the backend/tests treat every synthesis method the same way. |
| `method/synthesis/AIM/` | AIM (Adaptive & Iterative Mechanism) adapter + reference implementation. |
| `method/preprocess_common/` | Shared discretizers (PrivTree, DAWA) and preprocessing pipelines. |
| `test/` | Pytest suite; `test/e2e/` hosts Playwright end-to-end flows. |
| `sample_data/` | Small datasets for local trials (`adult.csv.zip`, etc.). |
| `scripts/` | Convenience scripts for booting servers, benchmarks, and E2E automation. |

## Key Components

### PrivSyn pipeline

- `web_app/data_inference.py` infers column metadata, builds draft `domain.json` / `info.json`, and returns it to the UI.
- The frontend collects overrides and posts them back via `/confirm_synthesis`, shaping the inputs for synthesis.
- `web_app/synthesis_service.py` rebuilds the dataframe, drives `method/synthesis/privsyn/privsyn.py` (marginal selection + GUM), and stores run artifacts.
- `web_app/data_comparison.py` evaluates outputs with metadata-aware TVD metrics and feeds results to the UI and APIs.

### AIM adapter

Method adapters live in `method/synthesis/AIM/adapter.py`; they map the unified interface (`prepare` / `run`) onto the original AIM workflow so the backend and tests treat AIM exactly like PrivSyn via the shared registry in `method/api`.

### Preprocessing discretizers

- **PrivTree (`method/preprocess_common/privtree.py`)** – recursive binary splits with Laplace noise and inverse transforms.
- **DAWA (`method/preprocess_common/dawa.py`)** – L1 partitioning utilities used by AIM.
- Corresponding tests under `test/test_privtree.py` and `test/test_dawa.py` keep these helpers deterministic.

## Deployment Notes

- **Local:** `./scripts/start_backend.sh` / `./scripts/start_frontend.sh` mirror the commands above; pass `prod` to emit gunicorn builds or copy frontend assets.
- **Cloud Run:** Build & deploy via Docker (see `gcloud builds submit …` example in the docs section). Remember to set `VITE_API_BASE_URL` in the frontend environment and extend `allow_origins` in `web_app/main.py` for any new domains.
- **Temp artifacts:** PrivSyn stores run products under the system temp dir by default. Set `PRIVSYN_DATA_ROOT` / `PRIVSYN_EXP_ROOT` if you need to redirect them (e.g., a workspace cache in CI).

## Contributing

1. Fork & create a branch (`git checkout -b feat/awesome-improvement`).
2. Keep PRs focused, add/update tests, and run `pytest -q` locally.
3. Follow Conventional Commits (`feat:`, `fix:`, `chore(scope):`, etc.).
4. Include screenshots/GIFs for UI changes (drop them under `docs/media/`).

## License

PrivSyn is released under the [MIT License](LICENSE).

---

> Need help? Open an issue or ping us on GitHub Discussions – we love hearing about new differential privacy use-cases!

# PrivSyn - Differentially Private Data Synthesis ![Coverage](https://codecov.io/gh/vvv214/privsyn-tabular/branch/main/graph/badge.svg) ![CI](https://github.com/vvv214/privsyn-tabular/actions/workflows/ci.yml/badge.svg) ![License](https://img.shields.io/github/license/vvv214/privsyn-tabular) ![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)

> End-to-end tooling for creating differentially private tabular datasets with PrivSyn and AIM.

<p align="center">
  <a href="docs/media/workflow.gif">
    <img src="docs/media/workflow.gif" alt="PrivSyn workflow" width="720" />
  </a>
  <br />
  <em>⬆️ Replace <code>docs/media/workflow.gif</code> with a screen recording of your favourite synthesis flow.</em>
</p>

## Highlights

- **Unified web app.** Upload a dataset, review inferred metadata, tweak categorical/numerical encodings, and download synthesized data in minutes.
- **Two synthesis engines.** PrivSyn (rho-CDP, iterative marginal updates) and AIM (adaptive measurement selection) exposed behind the same API.
- **Notebook-friendly modules.** Reusable preprocessing (PrivTree, DAWA), marginal selection, and synthesis utilities under `method/privsyn` and `preprocess_common`.
- **Coverage-first test suite.** 100+ pytest cases plus Playwright E2E flows keep the UI/back-end contract in check.

## Demo Walkthrough

1. **Upload** a CSV/ZIP or click “Load Sample” to stream `sample_data/adult.csv.zip`.
2. **Confirm metadata** – review inferred types, adjust categorical domains, clip numerical ranges, choose binning strategies.
3. **Run PrivSyn/AIM** – the backend preprocesses the data, generates a private synthetic dataset, and computes evaluation metrics.
4. **Download & compare** – preview the synthesized sample in the UI or fetch the CSV via `/download_synthesized_data/<name>`.

Record a GIF by following those four steps locally (e.g., `ffmpeg` or macOS screen recording) and save it as `docs/media/workflow.gif` to bring the above hero image to life.

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

## Testing

```bash
# Fast feedback (skips slow markers)
pytest -q -m "not slow"

# Full suite with coverage
pytest --cov=. --cov-report=term

# Browser E2E flows (requires Playwright browsers & frontend deps)
E2E=1 pytest -q -k e2e
```

Useful snippets:

```bash
# Metadata / preprocessing focus
pytest -q test/test_metadata_overrides.py test/test_preprocessing.py test/test_data_inference.py

# Strict warnings for core modules
pytest -q -W error::DeprecationWarning -W error::FutureWarning -k "web_app or preprocess_common or method/privsyn"
```

## Repository Layout

| Path | Description |
|------|-------------|
| `frontend/` | React + Vite SPA (metadata review UI, synthesis form, results view). |
| `web_app/` | FastAPI backend: metadata inference, synthesis orchestration, evaluation endpoints. |
| `method/privsyn/` | Authoritative PrivSyn implementation (marginal selection, GUM synthesis, dataset helpers). |
| `method/AIM/` | AIM (Adaptive & Iterative Mechanism) adapter + reference implementation. |
| `preprocess_common/` | Shared discretizers (PrivTree, DAWA) and preprocessing pipelines. |
| `test/` | Pytest suite; `test/e2e/` hosts Playwright end-to-end flows. |
| `sample_data/` | Small datasets for local trials (`adult.csv.zip`, etc.). |
| `scripts/` | Convenience scripts for booting servers, benchmarks, and E2E automation. |

## Key Components

### PrivSyn pipeline

1. **Inference** – `web_app/data_inference.py` classifies columns (numeric vs categorical), surfaces heuristics to the UI, and emits draft `domain.json` / `info.json`.
2. **Confirmation** – user overrides (categories, special tokens, clipping, binning) are serialized and posted to `/confirm_synthesis`.
3. **Synthesis** – `web_app/synthesis_service.py` reconstructs the dataframe, runs `method/privsyn/privsyn.py` (marginal selection + GUM), and stores artifacts for evaluation/download.
4. **Evaluation** – `web_app/data_comparison.py` computes metadata-aware TVD metrics (histograms for numeric columns, exact values for categorical columns).

### AIM adapter

Method adapters live in `method/AIM/adapter.py`; they map the unified interface (`prepare` / `run`) onto the original AIM workflow so the backend and tests treat AIM exactly like PrivSyn.

### Preprocessing discretizers

- **PrivTree (`preprocess_common/privtree.py`)** – recursive binary splits with Laplace noise and inverse transforms.
- **DAWA (`preprocess_common/dawa.py`)** – L1 partitioning utilities used by AIM.
- Corresponding tests under `test/test_privtree.py` and `test/test_dawa.py` keep these helpers deterministic.

## Deployment Notes

- **Local:** `start_backend.sh` / `start_frontend.sh` mirror the commands above.
- **Cloud Run:** Build & deploy via Docker (see `gcloud builds submit …` example in the docs section). Remember to set `VITE_API_BASE_URL` in the frontend environment and extend `allow_origins` in `web_app/main.py` for any new domains.

## Contributing

1. Fork & create a branch (`git checkout -b feat/awesome-improvement`).
2. Keep PRs focused, add/update tests, and run `pytest -q` locally.
3. Follow Conventional Commits (`feat:`, `fix:`, `chore(scope):`, etc.).
4. Include screenshots/GIFs for UI changes (drop them under `docs/media/`).

## License

PrivSyn is released under the [MIT License](LICENSE).

---

> Need help? Open an issue or ping us on GitHub Discussions – we love hearing about new differential privacy use-cases!

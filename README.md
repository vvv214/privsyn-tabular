# PrivSyn - Differentially Private Data Synthesis

This repository implements the PrivSyn algorithm for Differentially Private Data Synthesis, as described in the paper: [PrivSyn: Differentially Private Data Synthesis](https://www.usenix.org/system/files/sec21fall-zhang-zhikun.pdf).

## Introduction
The PrivSyn pipeline comprises three functional modules: data preprocessing, the PrivSyn main process, and synthesis evaluation.

### Project Structure
*   `data/`: Stores raw datasets (for local experiments).
*   `preprocess_common/`: Data preprocessing and encoders.
*   `method/privsyn/`: Core PrivSyn algorithm (authoritative).
*   `method/AIM/`: AIM (Adaptive and Iterative Mechanism) implementation.
*   `reconstruct_algo/`: Helpers to reconstruct decoded DataFrames from encoded parts.
*   `util/`: Shared helpers (e.g., privacy accounting `rho_cdp`).
*   `web_app/`: FastAPI backend for the web application.
*   `frontend/`: React.js frontend for the web application.
*   `test/`: Pytest tests; end-to-end tests live in `test/e2e/`.
*   `exp/`: Experimental outputs (created at runtime).

## Setup

To get started with PrivSyn, follow these steps:

### 1. Local Development Setup

#### Prerequisites
*   Python 3.8+
*   Node.js and npm (or Yarn)

#### Backend Setup (Python FastAPI Application)

1.  **Navigate to the project root:**
    ```bash
    cd [path]
    ```
2.  **Create a virtual environment:**
    ```bash
    python3 -m venv .venv
    ```
3.  **Activate the virtual environment:**
    *   On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```
    *   On Windows:
        ```bash
        .venv\Scripts\activate
        ```
4.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Run the FastAPI backend server:**
    Ensure your virtual environment is activated.
    ```bash
    uvicorn web_app.main:app --reload --port 8001
    ```
    The backend will typically run on `http://localhost:8001`.

#### Frontend Setup (React.js Application)

1.  **Open a new terminal window.**
2.  **Navigate to the frontend directory:**
    ```bash
    cd frontend
    ```
3.  **Install Node.js dependencies:**
    ```bash
    npm install
    ```
4.  **Run the frontend development server:**
    ```bash
    npm run dev -- --port 5174
    ```
    The frontend will typically run on `http://localhost:5174` (or another port as indicated by Vite).
    If `VITE_API_BASE_URL` is not set, the app will default requests to `http://127.0.0.1:8001`.

### Unit Tests (Python)

Run the Python unit test suite:

```
pytest -q test/
```

Useful options:
- Exclude slow tests: `pytest -q -m "not slow"`
- Run a subset by keyword (e.g., API smoke): `pytest -q -k api_smoke`

Tests use markers:
- `slow`: long-running tests; exclude with `-m "not slow"`
- `e2e`: frontend+backend end-to-end tests (see below)

Run targeted coverage for the metadata/data-inference pipeline:

```
.venv/bin/pytest \
  --cov=web_app.main \
  --cov=web_app.data_inference \
  --cov=preprocess_common.load_data_common \
  --cov-report=term \
  test/test_metadata_overrides.py test/test_preprocessing.py test/test_data_inference.py
```

This focuses the coverage on the components updated to support editable metadata while avoiding the heavier algorithm suites.

We keep UI automation under `test_e2e/` (Playwright) and the rest of the suite under `test/`; keeping them separate ensures fast inner-loop runs while still supporting browser-based checks when needed.

### Helper Scripts

The `scripts/` directory contains a few convenience wrappers:

- `start_backend.sh` / `start_backend_prod.sh`: run the FastAPI app in dev or Gunicorn mode.
- `start_frontend.sh` / `start_frontend_prod.sh`: launch Vite dev server or serve the production build.
- `bench_backend.py`, `bench_synthesis_speed.py`: quick performance probes for inference and synthesis stages.
- `run_e2e.sh`: boot both servers and execute the Playwright E2E suite.

Temporary runtime artifacts live under `temp_synthesis_output/` and `temp_bench_runs/`; both are ignored via `.gitignore` and can be safely cleared if disk space is needed.

### Metadata Confirmation UI Highlights

- **Inference explanation.** The screen shows the heuristics coming from `web_app/data_inference.py`: integer columns with fewer than 20 distinct values (or very sparse integers) default to categorical, floats need ≥10 distinct values to remain numerical, and non-numeric dtypes are always categorical. This clarifies cases like `age` being inferred as categorical when the uploaded sample contains only a handful of ages.
- **Categorical controls.** Each categorical attribute lists detected values with frequency counts. You can deselect values (treated as unexpected), add custom categories, and decide whether unexpected values are mapped to a special token (default `__OTHER__`) or remain part of the domain. Missing values are represented by a `__NULL__` token.
- **Numerical controls.** Numerical attributes expose clip bounds, target bin width/count, and a strategy selector (uniform spacing, exponential growth with a user-supplied rate, or a placeholder PrivTree option that currently falls back to uniform edges while reserving privacy budget for future DP use).
- **Backend alignment.** On confirmation the backend reconstructs the dataframe, applies the overrides (including categorical special-token mapping and numeric clipping/binning), and only then invokes preprocessing and synthesis, ensuring the confirmed metadata is authoritative.

### 1.1 End-to-End Test (Frontend + Backend, optional)

We provide a Playwright-based E2E test that drives the frontend against a live backend.

One-time setup:

```
pip install -r requirements.txt
python -m playwright install
cd frontend && npm install && cd ..
```

Run the E2E test (requires free ports 8001 and 5174):

```
E2E=1 pytest -q -k e2e
```

Notes:
- The test sets `VITE_API_BASE_URL` to `http://localhost:8001` and starts both servers.
- It uploads `sample_data/adult.csv.zip`, confirms inferred metadata, waits for synthesis to complete, and verifies the download link and preview table.
- E2E tests are marked `e2e`; you can exclude them from normal runs with `-m "not e2e"` or run only them with `-k e2e`.

Convenience script:

```
./scripts/run_e2e.sh
```

This will create a `.venv`, install Python deps, install Playwright browsers, install frontend deps, and run the E2E. E2E tests are located under `test/e2e/`.

### 1.2 Quick Benchmarks

Run a small end-to-end synthesis benchmark on the sample data:

```
python scripts/bench_synthesis_speed.py --n_sample 1000
```

It prints timings for load/infer/preprocess/marginal selection/synthesis stages.

#### Memory Usage Monitoring (Local)

When running the backend locally, you can observe memory peak usage in the terminal where the `uvicorn` server is running. We've added logging statements to `web_app/main.py` to output memory usage at key stages.

Example log output:
```
Memory usage at synthesize_data_start: RSS=XX.XX MB, VMS=YY.YY MB
```
This helps in identifying memory-intensive operations.

### 2. Programmatic Usage (Native Synthesizers)

Both PrivSyn and AIM now expose a native `Synthesizer` interface, which is the recommended way to use them programmatically. The synthesizers can be accessed through the `SynthRegistry`.

Example:
```python
from method.api import SynthRegistry, PrivacySpec, RunConfig

# df is a pandas DataFrame
# domain and info are dictionaries describing the data
# (These are typically inferred by the UI, but can be constructed manually)
domain = {'age': {'type': 'num', 'size': 256}, 'gender': {'type': 'cat', 'size': 3, 'categories': ['M', 'F', 'U']}}
info = {'num_columns': ['age'], 'cat_columns': ['gender']}

# 1. Get the synthesizer from the registry
synth = SynthRegistry.get('privsyn') # or 'aim'

# 2. Fit the synthesizer on the data
fitted_synth = synth.fit(
    df,
    domain,
    info,
    privacy=PrivacySpec(epsilon=1.0, delta=1e-5),
    config=RunConfig(device='cpu')
)

# 3. Sample synthetic data
synthetic_df = fitted_synth.sample(n=len(df))
```

## Deployment

This section outlines the deployment process for the PrivSyn web application to cloud environments like Render (for backend) and Vercel (for frontend).

### Backend Deployment (Render)

The backend is a FastAPI application deployed on Render.

1.  **Service Type:** Web Service
2.  **Build Command:** (Usually auto-detected, or `pip install -r requirements.txt`)
3.  **Start Command:**
    ```bash
    gunicorn web_app.main:app --workers 1 --worker-class uvicorn.workers.UvicornWorker
    ```
    *Explanation:* FastAPI is an ASGI application. Gunicorn is a WSGI server but can manage ASGI applications using `uvicorn.workers.UvicornWorker`. This setup provides robust process management and load balancing for production.
4.  **Environment Variables:**
    *   No specific environment variables are required for the backend itself, but ensure any database credentials or other sensitive information are configured as environment variables on Render.
5.  **Memory Considerations:**
    *   The application can be memory-intensive, especially when processing large datasets. If you encounter "Ran out of memory" errors, consider:
        *   **Optimizing data handling:** The application has been modified to use temporary files for intermediate data storage between `/synthesize` and `/confirm_synthesis` calls to reduce memory footprint.
        *   **Reducing `n_sample`:** In the frontend, try reducing the "Number of Samples" (`n_sample`) parameter.
        *   **Upgrading Render Plan:** If memory issues persist, you may need to upgrade your Render plan to one with higher memory limits.

### Frontend Deployment (Vercel)

The frontend is a React.js application built with Vite, deployed on Vercel.

1.  **Framework Preset:** Vite
2.  **Build Command:** `npm run build`
3.  **Output Directory:** `dist` (usually auto-detected)
4.  **Environment Variables:**
    *   **`VITE_API_BASE_URL`**: This crucial environment variable tells the frontend where your backend API is located.
        *   **Value:** Set this to the URL of your deployed backend (e.g., `https://privsyn-tabular.onrender.com`).
        *   **Important:** Ensure this variable is set in Vercel's project settings under "Environment Variables". Vite requires environment variables to be prefixed with `VITE_` to be exposed to the client-side code.
5.  **CORS Configuration:**
    *   The backend (FastAPI) is configured with `CORSMiddleware` to allow cross-origin requests from your frontend.
    *   In `web_app/main.py`, the `allow_origins` list should include your frontend's domain (e.g., `https://www.privsyn.com`).
    *   If you encounter CORS errors, ensure:
        *   The backend is running the latest code with the correct `allow_origins` configuration.
        *   There are no intermediary proxies or CDNs stripping the `Access-Control-Allow-Origin` header. (Setting `allow_origins=["*"]` temporarily can help diagnose if an external factor is interfering, but should not be used in production).

### Backend Deployment (Google Cloud Run)

The backend can also be deployed to Google Cloud Run using Docker.

1.  **Build the Docker image using Google Cloud Build:**

    Replace `<your-gcp-project-id>` with your actual Google Cloud Project ID.

    ```bash
    gcloud builds submit --tag gcr.io/<your-gcp-project-id>/privsyn-backend .
    ```

2.  **Deploy the image to Cloud Run:**

    This command deploys the container image to Cloud Run, making it publicly accessible.

    ```bash
    gcloud run deploy privsyn-tabular --image gcr.io/<your-gcp-project-id>/privsyn-backend --platform managed --region us-east4 --memory 1Gi --allow-unauthenticated
    ```

## Methods

The repo exposes multiple methods via a unified `Synthesizer` interface. Both PrivSyn and AIM return synthesized data in the same (decoded) tabular format (original column names and value types). If you prefer to reconstruct from encoded representations, use `reconstruct_algo/reconstruct.py`.

The legacy adapter interface is now deprecated and will be removed in a future version.

### Frontend/Backend Integration
- Frontend (Advanced Settings) includes a Method selector (`privsyn` or `aim`).
- Backend dispatch is centralized in `web_app/methods_dispatcher.py`.

## Warnings Policy

Tests silence noisy Deprecation/Future/Runtime warnings coming from heavy third‑party method folders (AIM, GEM, TabDDPM, DP_MERF, private_gsd, reconstruct_algo) via `pytest.ini` so core project warnings remain visible. To opt into strict mode for core code during local development or CI, run:

```
pytest -q -W error::DeprecationWarning -W error::FutureWarning -k "web_app or preprocess_common or method/privsyn"
```

## PrivSyn Modules (Modularized API)

Beyond an overall implementation of PrivSyn, this repository also offers the modularized API of PrivSyn, which are `InDif selection` and `GUM`.

### InDif selection
This is a method for marginal selection by measuring InDif. We implement it as a static method in `PrivSyn` class, called `two_way_marginal_selection` (see `method/privsyn/privsyn.py`). This method will return a list of 2-way marginal tuple, as the final selection. The hypermeters of this method can be summarized as
* `df`: a dataframe of dataset
* `domain`: a dictionary of attributes domain
* `rho_indif`: privacy budget for measuring InDif
* `rho_measure`: privacy budget for measuring selected marginals (actually this budget will not be used in this phase, but works as an optimization term during selection)

You can use this static method to select marginals for other synthesis modules
```python
selected_marginals = PrivSyn.two_way_marginal_selection(df, domain, rho_indif, rho_measure)
```

### GUM
We construct a class of GUM synthesis method called `GUM_Mechanisms` (see `method/privsyn/lib_synthesize/GUM.py`). Here is a instruction of using this closed-form synthesis module.
* `Initialization`. The initialization of GUM requires an input of hyperparameters dictionary (same as PrivSyn), dataset (Dataset class), a dictionary of 1-way marginals (used for data initialization, can be an empty dictionary), a dictionary of 2-way marginals. The dictionary of marginals should be in the form of:

    ```
    {'(attr1, attr2)': Marginal1, '(attr3, attr4)': Marginal2, ...}
    ```

    Here `Marginal1` and `Marginal2` should be in Marginal class (see `method/privsyn/lib_marginal/marg.py`), and measured by method `count_records`. You can initialize a GUM class like

    ```python
    model = GUM_Mechanism(args, df, dict1, dict2)
    ```

* `Main procedure`. The main procedure of GUM is finished by method `run`, which only requires the sampling number. This process includes three main steps: graph seperation, marginal consistency, and records updation.
    ```python
    model.run(n_sample = 10000)
    synthesized_df = model.synthesized_df
    ```

* `Adaptive mechanism`. We also support measurement for synthesized dataset, which can be used for adaptive marginal selection.
    ```python
    syn_vector = model.project(('attr1', 'attr2')).datavector()
    real_vector = dataset.project(('attr1', 'attr2')).datavector()
    gap = np.linalg.norm(syn_vector - real_vector, ord=1)
    ```

# PrivSyn - Differentially Private Data Synthesis

This repository implements the PrivSyn algorithm for Differentially Private Data Synthesis, as described in the paper: [PrivSyn: Differentially Private Data Synthesis](https://www.usenix.org/system/files/sec21fall-zhang-zhikun.pdf).

## Introduction
The PrivSyn pipeline comprises three functional modules: data preprocessing, the PrivSyn main process, and synthesis evaluation.

### Project Structure
*   `data/`: Stores raw datasets.
*   `preprocess_common/`: Contains code for data preprocessing.
*   `privsyn/`: Implements the core PrivSyn algorithm.
*   `evaluator/`: Provides code for evaluating synthesis results.
*   `eval_models/`: Stores settings for evaluation models.
*   `util/`: Contains various helper functions.
*   `web_app/`: FastAPI backend for the web application.
*   `frontend/`: React.js frontend for the web application.
*   `exp/`: Collects and saves experimental results.

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
    pip install psutil  # For memory monitoring
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

#### Memory Usage Monitoring (Local)

When running the backend locally, you can observe memory peak usage in the terminal where the `uvicorn` server is running. We've added logging statements to `web_app/main.py` to output memory usage at key stages.

Example log output:
```
Memory usage at synthesize_data_start: RSS=XX.XX MB, VMS=YY.YY MB
```
This helps in identifying memory-intensive operations.

### 2. Running PrivSyn via CLI (for Research/Evaluation)

The `main.py` script provides a command-line interface for running PrivSyn experiments.

#### Hyper-parameters
*   `method`: Synthesis method to run (e.g., `privsyn`).
*   `dataset`: Name of the dataset (e.g., `bank`).
*   `device`: Device for running algorithms (e.g., `cuda:0` or `cpu`).
*   `epsilon`: Differential privacy parameter (required).
*   `--delta`: Differential privacy parameter (default: `1e-5`).
*   `--num_preprocess`: Preprocessing method for numerical attributes (default: `uniform_kbins`).
*   `--rare_threshold`: Threshold for categorical attribute preprocessing (default: `0.002`).
*   `--sample_device`: Device for data sampling (defaults to `device`).

#### Example Usage
First, ensure your datasets are placed in the `data/` folder (e.g., `data/bank`). Necessary datasets are usually provided.

If evaluation models need tuning (e.g., for a new dataset), you can finetune them:
```bash
python evaluator/tune_eval_model.py bank mlp cv cuda:0
```

To run an overall evaluation with PrivSyn:
```bash
python main.py privsyn bank cuda:0 1.0
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

## PrivSyn Modules (Modularized API)

Beyond an overall implementation of PrivSyn, this repository also offers the modularized API of PrivSyn, which are `InDif selection` and `GUM`.

### InDif selection
This is a method for marginal selection by measuring InDif. We implement it as a static method in `PrivSyn` class, called `two_way_marginal_selection` (see `privsyn/privsyn.py`). This method will return a list of 2-way marginal tuple, as the final selection. The hypermeters of this method can be summarized as
* `df`: a dataframe of dataset
* `domain`: a dictionary of attributes domain
* `rho_indif`: privacy budget for measuring InDif
* `rho_measure`: privacy budget for measuring selected marginals (actually this budget will not be used in this phase, but works as an optimization term during selection)

You can use this static method to select marginals for other synthesis modules
```python
selected_marginals = PrivSyn.two_way_marginal_selection(df, domain, rho_indif, rho_measure)
```

### GUM
We construct a class of GUM synthesis method called `GUM_Mechanisms` (see `privsyn/lib_synthesize/GUM.py`). Here is a instruction of using this closed-form synthesis module.
* `Initialization`. The initialization of GUM requires an input of hyperparameters dictionary (same as PrivSyn), dataset (Dataset class), a dictionary of 1-way marginals (used for data initialization, can be an empty dictionary), a dictionary of 2-way marginals. The dictionary of marginals should be in the form of:

    ```
    {'(attr1, attr2)': Marginal1, '(attr3, attr4)': Marginal2, ...}
    ```

    Here `Marginal1` and `Marginal2` should be in Marginal class (see `privsyn/lib_marginal/marg.py`), and measured by method `count_records`. You can initialize a GUM class like

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
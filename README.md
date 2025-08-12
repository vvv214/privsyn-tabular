# PrivSyn

This repository is an implementation of paper [PrivSyn: Differentially Private Data Synthesis](https://www.usenix.org/system/files/sec21fall-zhang-zhikun.pdf). 

## Introducion
The pipeline of the PrivSyn consists of three functional modules: data preprocessing, PrivSyn main process, and synthesis evaluation. The file structure can be summarized as follows.
* `data/`: used for save raw data.
* `preprocess_common/`: code for data preprocessing.
* `privsyn/`: code for the main procedure of PrivSyn.
* `evaluator/`: code for evaluation.
* `eval_models/`: this file stores the settings of evaluation models.
* `util/`: code for some helper functions.
* `exp/`: the results of experiments will be collected and save in this file. 


## Quick Start
### Hyper-paprameters
The code for running experiments is in `main.py`. The detailed description of some common hyper-parameters are give as follows.
* `method`: which synthesis method you will run.
* `dataset`: name of dataset.
* `device`: the device used for running algorithms. 
* `epsilon`: DP parameter, which must be delivered when running code. 
* `--delta`: DP parameter, which is set to $1e-5$ by default.
* `--num_preprocess`: preprocessing method for numerical attributes, which is set to uniform binning by default. 
* `--rare_threshold`: threshold of preprocessing method for categorical attributes, which is set to $0.2\%$ by default.
* `--sample_device`: device used for sample data, by default is set to the same as running device.
There are some other hyper-paramters specifically for PrivSyn main procedure in file `privsyn/privsyn.py`. We recommend using default values for these hyper-parameters.

### Run PrivSyn
Firstly, make sure the datasets are put in the correct fold (in the following examples, the fold is `data/bank`, and the necessary dataset has already been provided). In this repository, the evaluation model is already tuned so users do not need any operation. Otherwise, you should tune the evaluation model (using the following code) before any further operation. For instance, you can finetune a mlp model for evaluation like
```
python evaluator/tune_eval_model.py bank mlp cv cuda:0
```

After preparation, we can try the following code to make an overall evaluation. Usually, we by default set `num_preprocess` to be "uniform_kbins" except for DP-MERF and TabDDPM, and set `rare_threshold` to 0.002 for overall evaluation. Therefore, if you do not want to change these settings, you do not need to include these hyper-parameters in your command line.
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
*   `exp/`: Collects and saves experimental results.
*   `web_app/`: FastAPI backend for the web application.
*   `frontend/`: Vue.js frontend for the web application.

## Setup

To get started with PrivSyn, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-repo/privsyn-tabular.git
    cd privsyn-tabular
    ```

2.  **Set up Python Environment:**
    It's highly recommended to use a virtual environment to manage dependencies.

    ```bash
    # Create a virtual environment
    python3 -m venv .venv

    # Activate the virtual environment
    source .venv/bin/activate

    # Install Python dependencies
    pip install -r requirements.txt
    ```

3.  **Set up Frontend Environment:**
    Navigate to the `frontend` directory and install Node.js dependencies.

    ```bash
    cd frontend
    npm install # or yarn install
    cd ..
    ```

## Running the Application

This project includes both a command-line interface (CLI) for research/evaluation and a web-based application.

### 1. Running the Backend (Web Application API)

The backend is a FastAPI application.

```bash
# Ensure your Python virtual environment is activated
source .venv/bin/activate

# Start the backend server
./start_backend.sh
```
The backend will typically run on `http://127.0.0.1:8001`.

### 2. Running the Frontend (Web User Interface)

The frontend is a Vue.js application.

```bash
# Navigate to the frontend directory
cd frontend

# Start the frontend development server
./start_frontend.sh
```
The frontend will typically run on `http://localhost:5173` (or another port as indicated by Vite).

### 3. Running PrivSyn via CLI (for Research/Evaluation)

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

## PrivSyn Modules (Modularized API)

Beyond the overall implementation, this repository offers modularized APIs for `InDif selection` and `GUM`.

### InDif Selection
This module performs marginal selection by measuring InDif. It's implemented as a static method `two_way_marginal_selection` within the `PrivSyn` class (`privsyn/privsyn.py`). It returns a list of 2-way marginal tuples.

#### Parameters:
*   `df`: A pandas DataFrame of the dataset.
*   `domain`: A dictionary of attribute domains.
*   `rho_indif`: Privacy budget for measuring InDif.
*   `rho_measure`: Privacy budget for measuring selected marginals (used as an optimization term).

#### Example Usage:
```python
from privsyn.privsyn import PrivSyn
# Assume df, domain, rho_indif, rho_measure are defined
selected_marginals = PrivSyn.two_way_marginal_selection(df, domain, rho_indif, rho_measure)
```

### GUM (General Update Mechanism)
The `GUM_Mechanism` class (`privsyn/lib_synthesize/GUM.py`) provides a closed-form synthesis module.

#### Initialization:
GUM initialization requires a hyper-parameters dictionary (similar to PrivSyn), a `Dataset` class instance, a dictionary of 1-way marginals (can be empty), and a dictionary of 2-way marginals. Marginals should be `Marginal` class instances (`privsyn/lib_marginal/marg.py`), measured using `count_records`.

```python
from privsyn.lib_synthesize.GUM import GUM_Mechanism
# Assume args, dataset, one_way_marg_dict, combined_marg_dict are defined
model = GUM_Mechanism(args, dataset, one_way_marg_dict, combined_marg_dict)
```

#### Main Procedure:
The `run` method executes the GUM process, including graph separation, marginal consistency, and record updates. It requires the sampling number.

```python
# Assume model is initialized
model.run(n_sample = 10000)
synthesized_df = model.synthesized_df
```

#### Adaptive Mechanism:
The module also supports measurement for synthesized datasets, useful for adaptive marginal selection.

```python
# Assume model and dataset are defined
syn_vector = model.project(('attr1', 'attr2')).datavector()
real_vector = dataset.project(('attr1', 'attr2')).datavector()
gap = np.linalg.norm(syn_vector - real_vector, ord=1)
```

## Deployment Considerations

For deploying the PrivSyn web application to a cloud environment, consider the following:

*   **Environment Variables:** Configure environment variables for sensitive information or configurable parameters (e.g., API keys, database connections, port numbers).
*   **Production Build:** For the frontend, create a production build (`npm run build` in the `frontend` directory) which generates optimized static assets. These assets can then be served by a web server (e.g., Nginx, Apache) or integrated directly into the backend.
*   **Process Management:** Use a production-ready WSGI server for the Python backend (e.g., Gunicorn, uWSGI) instead of `uvicorn --reload`. A process manager like PM2 (for Node.js) or systemd (for Linux) can manage the backend and frontend processes.
*   **Containerization:** Consider using Docker to containerize both the backend and frontend applications for easier deployment and scalability across different environments.
*   **Data Storage:** Ensure your cloud environment provides suitable persistent storage for your `data/` and `exp/` directories if they need to persist across deployments or instances.

## PrivSyn Modules
Except for an overall implementation of PrivSyn, this repository also offers the modularized API of PrivSyn, which are `InDif selection` and `GUM`. 

### InDif selection
This is a method for marginal selection by measuring InDif. We implement it as a static method in `PrivSyn` class, called `two_way_marginal_selection` (see `privsyn/privsyn.py`). This method will return a list of 2-way marginal tuple, as the final selection. The hypermeters of this method can be summarized as 
* `df`: a dataframe of dataset
* `domain`: a dictionary of attributes domain
* `rho_indif`: privacy budget for measuring InDif
* `rho_measure`: privacy budget for measuring selected marginals (actually this budget will not be used in this phase, but works as an optimization term during selection)

You can use this static method to select marginals for other synthesis modules
```
selected_marginals = PrivSyn.two_way_marginal_selection(df, domain, rho_indif, rho_measure)
```

### GUM
We construct a class of GUM synthesis method called `GUM_Mechanisms` (see `privsyn/lib_synthesize/GUM.py`). Here is a instruction of using this closed-form synthesis module.
* `Initialization`. The initialization of GUM requires an input of hyperparameters dictionary (same as PrivSyn), dataset (Dataset class), a dictionary of 1-way marginals (used for data initialization, can be an empty dictionary), a dictionary of 2-way marginals. The dictionary of marginals should be in the form of:

    ```
    {'(attr1, attr2)': Marginal1, '(attr3, attr4)': Marginal2, ...}
    ```

    Here `Marginal1` and `Marginal2` should be in Marginal class (see `privsyn/lib_marginal/marg.py`), and measured by method `count_records`. You can initialize a GUM class like 

    ```
    model = GUM_Mechanism(args, df, dict1, dict2)
    ```

* `Main procedure`. The main procedure of GUM is finished by method `run`, which only requires the sampling number. This process includes three main steps: graph seperation, marginal consistency, and records updation. 
    ```
    model.run(n_sample = 10000)
    synthesized_df = model.synthesized_df
    ```

* `Adaptive mechanism`. We also support measurement for synthesized dataset, which can be used for adaptive marginal selection. 
    ```
    syn_vector = model.project(('attr1', 'attr2')).datavector()
    real_vector = dataset.project(('attr1', 'attr2')).datavector()
    gap = np.linalg.norm(syn_vector - real_vector, ord=1)
    ```
     
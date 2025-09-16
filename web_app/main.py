from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import logging
import shutil
import tempfile
import os
import io
import pandas as pd
import numpy as np
import json
import importlib # For dynamic import of evaluation scripts
import math
from .data_comparison import calculate_tvd_metrics
from fastapi.staticfiles import StaticFiles
import zipfile
import sys

import psutil # For memory monitoring

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def log_memory_usage(stage: str):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    # RSS (Resident Set Size) is the non-swapped physical memory a process has used.
    # VMS (Virtual Memory Size) is the total virtual memory used by the process.
    logger.info(f"Memory usage at {stage}: RSS={mem_info.rss / (1024 * 1024):.2f} MB, VMS={mem_info.vms / (1024 * 1024):.2f} MB")



project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from .synthesis_service import run_synthesis, Args # Import Args from synthesis_service
import uuid # For generating unique IDs for temporary storage
from .data_inference import (
    infer_data_metadata,
    load_dataframe_from_uploaded_file,
    canonicalize_category,
    CATEGORY_NULL_TOKEN,
)
from fastapi import Response

app = FastAPI()

# Define allowed origins for CORS
# Add the production domain, and local development domains.
allow_origins_list = [
    "https://www.privsyn.com",
    "http://localhost:5174",
    "http://127.0.0.1:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info(f"CORS allow_origins configured for: {repr(allow_origins_list)}")

# Mount static files for the frontend
# app.mount("/static", StaticFiles(directory="web_app/static"), name="static")

# Store inferred data temporarily for confirmation flow
# Key: unique_id, Value: {"df": pd.DataFrame, "domain_data": dict, "info_data": dict, "target_column": str}
inferred_data_temp_storage = {}

# Store paths to synthesized and original data for evaluation
# In a real application, this would be a more robust storage solution (e.g., database, persistent storage)
# For this example, we'll use a dictionary in memory.
# Key: dataset_name, Value: {"synthesized_csv_path": "...", "original_data_dir": "..."}
data_storage = {}

@app.get("/")
async def read_root():
    return {"message": "Welcome to the PrivSyn Web App!"}

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello, {name}"}

@app.get("/sample_dataset/{name}")
async def get_sample_dataset(name: str):
    """
    Serves a small sample dataset ZIP for quick try-out in the UI.
    Currently supports: 'adult'.
    """
    if name != "adult":
        raise HTTPException(status_code=404, detail="Sample dataset not found")
    sample_path = os.path.join(project_root, "sample_data", "adult.csv.zip")
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample dataset file missing on server")
    return StreamingResponse(
        io.FileIO(sample_path, "rb"),
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=adult.csv.zip"}
    )

@app.post("/synthesize")
async def synthesize_data(
    method: str = Form(...),
    dataset_name: str = Form(...),
    epsilon: float = Form(...),
    delta: float = Form(...),
    num_preprocess: str = Form(...),
    rare_threshold: float = Form(...),
    n_sample: int = Form(...),
    consist_iterations: int = Form(501),
    non_negativity: str = Form('N3'),
    append: bool = Form(True),
    sep_syn: bool = Form(False),
    initialize_method: str = Form('singleton'),
    update_method: str = Form('S5'),
    update_rate_method: str = Form('U4'),
    update_rate_initial: float = Form(1.0),
    update_iterations: int = Form(50),
    data_file: UploadFile | None = File(None, description="Upload your dataset as a CSV or ZIP file."),
    target_column: str = Form('y_attr', description="Name of the target column in your CSV. Defaults to 'y_attr'."),):
    logger.info(f"synthesize_data received: method={method}, dataset_name={dataset_name}, epsilon={epsilon}, delta={delta}, num_preprocess={num_preprocess}, rare_threshold={rare_threshold}, n_sample={n_sample}, consist_iterations={consist_iterations}, update_iterations={update_iterations}")
    """
    Receives an uploaded CSV or ZIP file, infers metadata,
    and returns the inferred metadata for user confirmation.
    """
    logger.info(f"Entering synthesize_data endpoint. Dataset: {dataset_name}, Method: {method}")
    log_memory_usage("synthesize_data_start")
    if dataset_name == "debug_dataset":
        logger.info("Debug mode: Bypassing data inference and returning dummy metadata.")
        unique_id = str(uuid.uuid4())
        inferred_data_temp_storage[unique_id] = {
            "df": pd.DataFrame(np.random.rand(10, 5)), # Dummy DataFrame
            "X_cat": np.array([["A", "B"], ["C", "D"]]), # Dummy X_cat
            "X_num": np.array([[1.0, 2.0], [3.0, 4.0]]), # Dummy X_num
            "domain_data": {
                "num_col_1": {"type": "numerical", "size": 10},
                "cat_col_1": {"type": "categorical", "size": 5}
            }, # Dummy domain_data
            "info_data": {"name": "debug_dataset"}, # Dummy info_data
            "target_column": target_column,
            "synthesis_params": {
                "method": method,
                "dataset_name": dataset_name,
                "epsilon": epsilon,
                "delta": delta,
                "num_preprocess": num_preprocess,
                "rare_threshold": rare_threshold,
                "n_sample": n_sample,
                "consist_iterations": consist_iterations,
                "non_negativity": non_negativity,
                "append": append,
                "sep_syn": sep_syn,
                "initialize_method": initialize_method,
                "update_method": update_method,
                "update_rate_method": update_rate_method,
                "update_rate_initial": update_rate_initial,
                "update_iterations": update_iterations,
            }
        }
        logger.info(f"synthesize_data populated inferred_data_temp_storage with unique_id: {unique_id}, dataset_name: {dataset_name}")
        logger.info(f"Domain data sent to frontend: {repr(inferred_data_temp_storage[unique_id]['domain_data'])}")
        logger.info(f"Info data sent to frontend: {repr(inferred_data_temp_storage[unique_id]['info_data'])}")
        return JSONResponse(content={
            "message": "Metadata inferred. Please confirm.",
            "unique_id": unique_id,
            "domain_data": inferred_data_temp_storage[unique_id]["domain_data"],
            "info_data": inferred_data_temp_storage[unique_id]["info_data"]
        })

    try:
        if data_file is None:
            # Allow server-side sample loading when dataset_name matches a built-in sample
            if dataset_name == "adult":
                sample_path = os.path.join(project_root, "sample_data", "adult.csv.zip")
                if not os.path.exists(sample_path):
                    raise HTTPException(status_code=404, detail="Sample dataset file missing on server")
                logger.info("Loading built-in sample dataset: adult")
                with zipfile.ZipFile(sample_path, 'r') as zf:
                    csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
                    if not csv_files:
                        raise HTTPException(status_code=500, detail="No CSV found in sample ZIP")
                    with zf.open(csv_files[0]) as csv_file:
                        df = pd.read_csv(csv_file)
            else:
                raise HTTPException(status_code=400, detail="Data file is required for synthesis unless using a built-in sample or debug mode.")
        else:
            # 1. Load the uploaded file into a DataFrame
            logger.info("Attempting to load dataframe from uploaded file.")
            df = load_dataframe_from_uploaded_file(data_file)
        log_memory_usage("synthesize_data_after_df_load")
        logger.info(f"DataFrame loaded. Shape: {df.shape}")

        # 2. Infer data metadata
        logger.info("Attempting to infer data metadata.")
        inferred_data = infer_data_metadata(df, target_column=target_column)
        log_memory_usage("synthesize_data_after_metadata_inference")
        logger.info("Data metadata inferred successfully.")

        # Store original df and y temporarily with a unique ID
        unique_id = str(uuid.uuid4())
        
        # Create a temporary directory for this unique_id
        temp_dir = os.path.join(tempfile.gettempdir(), unique_id)
        os.makedirs(temp_dir, exist_ok=True)

        # Save df to a temporary parquet file
        df_path = os.path.join(temp_dir, "df.parquet")
        df.to_parquet(df_path, index=False)
        logger.info(f"Saved df to {df_path}")

        inferred_data_temp_storage[unique_id] = {
            "df_path": df_path,
            "domain_data": inferred_data['domain_data'],
            "info_data": inferred_data['info_data'],
            "target_column": target_column,
            "synthesis_params": {
                "method": method,
                "dataset_name": dataset_name,
                "epsilon": epsilon,
                "delta": delta,
                "num_preprocess": num_preprocess,
                "rare_threshold": rare_threshold,
                "n_sample": n_sample,
                "consist_iterations": consist_iterations,
                "non_negativity": non_negativity,
                "append": append,
                "sep_syn": sep_syn,
                "initialize_method": initialize_method,
                "update_method": update_method,
                "update_rate_method": update_rate_method,
                "update_rate_initial": update_rate_initial,
                "update_iterations": update_iterations,
            },
            "temp_dir": temp_dir # Store the temporary directory path for cleanup
        }
        logger.info(f"synthesize_data populated inferred_data_temp_storage with unique_id: {unique_id}, dataset_name: {dataset_name}")
        log_memory_usage("synthesize_data_before_return")
        return JSONResponse(content={
            "message": "Metadata inferred. Please confirm.",
            "unique_id": unique_id,
            "domain_data": inferred_data['domain_data'],
            "info_data": inferred_data['info_data']
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during data inference or file processing.")
        raise HTTPException(status_code=500, detail=f"File processing or inference failed: {str(e)}")

@app.post("/confirm_synthesis")
async def confirm_synthesis(
    unique_id: str = Form(...),
    method: str = Form(...),
    dataset_name: str = Form(...),
    epsilon: float = Form(...),
    delta: float = Form(...),
    num_preprocess: str = Form(...),
    rare_threshold: float = Form(...),
    n_sample: int = Form(...),
    consist_iterations: int = Form(501),
    non_negativity: str = Form('N3'),
    append: bool = Form(True),
    sep_syn: bool = Form(False),
    initialize_method: str = Form('singleton'),
    update_method: str = Form('S5'),
    update_rate_method: str = Form('U4'),
    update_rate_initial: float = Form(1.0),
    update_iterations: int = Form(50),
    confirmed_domain_data: str = Form(...), # JSON string of domain.json content
    confirmed_info_data: str = Form(...),   # JSON string of info.json content
):
    logger.info(f"confirm_synthesis received dataset_name: {dataset_name}")
    log_memory_usage("confirm_synthesis_start")
    """
    Receives confirmation of inferred metadata and proceeds with data synthesis.
    """
    if unique_id not in inferred_data_temp_storage:
        raise HTTPException(status_code=404, detail="Inferred data not found or session expired. Please re-upload.")

    temp_data = inferred_data_temp_storage.pop(unique_id) # Retrieve and remove from temp storage
    df_path = temp_data["df_path"]
    temp_dir = temp_data["temp_dir"]

    # Load df from temporary parquet file
    df = pd.read_parquet(df_path)
    logger.info(f"Loaded df from {df_path}")
    log_memory_usage("confirm_synthesis_after_df_load")

    target_column = temp_data["target_column"]
    synthesis_params = temp_data["synthesis_params"] # Retrieve synthesis parameters

    # Parse the JSON strings back into dictionaries
    domain_data = json.loads(confirmed_domain_data)
    info_data = json.loads(confirmed_info_data)

    # Apply user overrides to the dataframe before encoding
    processed_df = df.copy()
    num_col_names = []
    cat_col_names = []

    def compute_bin_edges(series: pd.Series, min_bound: float, max_bound: float, binning_config: dict):
        method = (binning_config or {}).get("method", "uniform")
        bin_width = binning_config.get("bin_width") if binning_config else None
        bin_count = binning_config.get("bin_count") if binning_config else None
        growth_rate = binning_config.get("growth_rate") if binning_config else None

        if min_bound is None or max_bound is None:
            observed_min = series.min() if not series.empty else None
            observed_max = series.max() if not series.empty else None
            if min_bound is None:
                min_bound = observed_min
            if max_bound is None:
                max_bound = observed_max

        if min_bound is None or max_bound is None or max_bound <= min_bound:
            return None, None, None

        span = max_bound - min_bound
        if bin_width and (bin_width > 0):
            bin_count = int(math.ceil(span / bin_width))
        if not bin_count or bin_count <= 0:
            bin_count = 10

        if method == "dp_privtree":
            # TODO: integrate true PrivTree binning; for now fall back to uniform edges
            logger.warning("PrivTree binning selected; using uniform edges as a placeholder until DP support is wired in.")

        if method == "exponential" and growth_rate and growth_rate > 1:
            weights = np.array([growth_rate ** i for i in range(bin_count)], dtype=float)
            weights_sum = weights.sum()
            cumulative = np.cumsum(weights) / weights_sum
            edges = [min_bound] + list(min_bound + cumulative * span)
        else:
            edges = np.linspace(min_bound, max_bound, bin_count + 1).tolist()
            if method != "dp_privtree":
                method = "uniform"

        return edges, method, bin_count

    for column, config in domain_data.items():
        col_type = config.get("type")
        if col_type == "categorical":
            cat_col_names.append(column)
            categories_from_data = config.get("categories_from_data", [])
            selected_categories = config.get("selected_categories", categories_from_data)
            custom_categories = config.get("custom_categories", [])
            special_token = config.get("special_token") or "__OTHER__"
            excluded_strategy = config.get("excluded_strategy", "map_to_special")

            if column in processed_df.columns:
                canonical_series = processed_df[column].apply(canonicalize_category)
            else:
                logger.warning(f"Column '{column}' missing from dataframe during categorical processing.")
                canonical_series = pd.Series([], dtype=str)

            excluded_categories = config.get("excluded_categories")
            if excluded_categories is None:
                excluded_categories = [cat for cat in categories_from_data if cat not in selected_categories]

            selected_set = set(selected_categories + custom_categories)
            excluded_set = set(excluded_categories)

            if excluded_strategy == "map_to_special":
                def map_value(value: str) -> str:
                    if value in selected_set or value in custom_categories:
                        return value
                    return special_token

                canonical_series = canonical_series.apply(map_value)
                final_categories = list({*selected_set, *custom_categories, special_token})
            else:
                final_categories = list({*selected_set, *custom_categories, *excluded_set})

            if CATEGORY_NULL_TOKEN in canonical_series.values and CATEGORY_NULL_TOKEN not in final_categories:
                final_categories.append(CATEGORY_NULL_TOKEN)

            config["categories"] = final_categories
            config["size"] = len(final_categories)
            config["excluded_categories"] = list(excluded_set)
            processed_df[column] = canonical_series.astype(str)
        elif col_type == "numerical":
            num_col_names.append(column)
            bounds = config.get("bounds", {})
            col_series = pd.to_numeric(processed_df[column], errors="coerce") if column in processed_df.columns else pd.Series([], dtype=float)

            min_bound = bounds.get("min")
            max_bound = bounds.get("max")
            summary = config.get("numeric_summary") or config.get("numeric_candidate_summary") or {}
            if min_bound is None:
                min_bound = summary.get("min")
            if max_bound is None:
                max_bound = summary.get("max")

            if min_bound is not None:
                col_series = col_series.clip(lower=min_bound)
            if max_bound is not None:
                col_series = col_series.clip(upper=max_bound)

            fill_value = min_bound if min_bound is not None else 0
            col_series = col_series.fillna(fill_value)

            edges, resolved_method, resolved_count = compute_bin_edges(col_series, min_bound, max_bound, config.get("binning"))
            if edges is not None:
                config.setdefault("binning", {})
                config["binning"]["edges"] = edges
                config["binning"]["method"] = resolved_method
                config["binning"]["bin_count"] = resolved_count

            processed_df[column] = col_series
            config["bounds"] = {"min": min_bound, "max": max_bound}
            if config.get("binning", {}).get("bin_count"):
                config["size"] = int(config["binning"]["bin_count"])
            else:
                config["size"] = int(col_series.nunique()) if not col_series.empty else 0
        else:
            logger.debug(f"Column '{column}' has unsupported type '{col_type}'.")

    info_data["num_columns"] = num_col_names
    info_data["cat_columns"] = cat_col_names
    info_data["n_num_features"] = len(num_col_names)
    info_data["n_cat_features"] = len(cat_col_names)

    X_num = processed_df[num_col_names].to_numpy(dtype=float) if num_col_names else None
    X_cat = processed_df[cat_col_names].astype(str).to_numpy() if cat_col_names else None

    try:
        # Create a persistent directory for this synthesis run (still needed for synthesized_csv_path)
        synthesis_run_dir = os.path.join(project_root, "temp_synthesis_output", "runs", unique_id)
        os.makedirs(synthesis_run_dir, exist_ok=True)

        # Write domain.json and info.json to disk for run_synthesis
        with open(os.path.join(synthesis_run_dir, "domain.json"), "w") as f:
            json.dump(domain_data, f, indent=4)
        with open(os.path.join(synthesis_run_dir, "info.json"), "w") as f:
            json.dump(info_data, f, indent=4)

        original_df = processed_df[num_col_names + cat_col_names]
        logger.info(f"Successfully constructed in-memory original_df with shape: {original_df.shape}")

        # Prepare arguments for run_synthesis
        args_dict = {
            "method": method,
            "dataset": dataset_name,
            "epsilon": epsilon,
            "delta": delta,
            "num_preprocess": num_preprocess,
            "rare_threshold": rare_threshold,
            "n_sample": n_sample,
            "consist_iterations": consist_iterations,
            "non_negativity": non_negativity,
            "append": append,
            "sep_syn": sep_syn,
            "initialize_method": initialize_method,
            "update_method": update_method,
            "update_rate_method": update_rate_method,
            "update_rate_initial": update_rate_initial,
            "update_iterations": update_iterations,
            "test": False, # This is a synthesis run, not a test
            "syn_test": False, # Not a synthesis test
            "device": "cpu", # Default for synthesis
            "sample_device": "cpu", # Default for synthesis
        }
        args = Args(**args_dict)
        log_memory_usage("confirm_synthesis_before_run_synthesis")

        # Run synthesis
        synthesized_csv_path, _ = await run_synthesis( # original_data_dir_for_eval is no longer needed
            args=args,
            data_dir=synthesis_run_dir, # data_dir is still needed for synthesis output
            X_num_raw=X_num,
            X_cat_raw=X_cat,
            confirmed_domain_data=domain_data,
            confirmed_info_data=info_data
        )

        # Store data for evaluation
        logger.info(f"Populating data_storage for dataset: {dataset_name}")
        data_storage[dataset_name] = {
            "synthesized_csv_path": synthesized_csv_path,
            "original_df": original_df, # Store original_df directly
            "method": synthesis_params["method"],
            "epsilon": synthesis_params["epsilon"],
            "delta": synthesis_params["delta"],
            "num_preprocess": synthesis_params["num_preprocess"],
            "rare_threshold": synthesis_params["rare_threshold"],
        }
        logger.info(f"Current data_storage keys: {data_storage.keys()}")
        log_memory_usage("confirm_synthesis_before_return")

        return JSONResponse(content={"message": "Data synthesis initiated successfully!", "dataset_name": dataset_name})
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error during synthesis confirmation.")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")

@app.get("/download_synthesized_data/{dataset_name}")
async def download_synthesized_data(dataset_name: str):
    """
    Provides the synthesized data CSV for download.
    """
    if dataset_name not in data_storage:
        raise HTTPException(status_code=404, detail="Synthesized data not found for this dataset name.")

    synthesized_csv_path = data_storage[dataset_name]["synthesized_csv_path"]
    if not os.path.exists(synthesized_csv_path):
        raise HTTPException(status_code=404, detail="Synthesized data file not found on server.")

    return StreamingResponse(
        io.FileIO(synthesized_csv_path, "rb"),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={dataset_name}_synthesized.csv"}
    )

@app.post("/evaluate")
async def evaluate_data_fidelity(
    dataset_name: str = Form(...),
):
    logger.debug(f"sys.path: {sys.path}")
    """
    Triggers evaluation of synthesized data fidelity using selected methods.
    """
    if dataset_name not in data_storage:
        raise HTTPException(status_code=404, detail="Synthesized data not found for this dataset name. Please synthesize first.")

    data_info = data_storage[dataset_name]
    synthesized_csv_path = data_info["synthesized_csv_path"]
    original_df = data_info["original_df"] # Access original_df directly

    if not os.path.exists(synthesized_csv_path):
        raise HTTPException(status_code=404, detail="Synthesized data file not found for evaluation.")
    # No need to check for original_data_dir existence anymore as original_df is in memory

    results = {}

    try:
        logger.info("Calculating TVD metrics...")
        # original_df is already available
        synthesized_df = pd.read_csv(synthesized_csv_path)

        tvd_results = calculate_tvd_metrics(original_df, synthesized_df)
        results["tvd_metrics"] = tvd_results
        logger.info("TVD metrics calculated successfully.")
    except Exception as e:
        logger.exception(f"Error calculating TVD metrics: {e}")
        results["tvd_metrics"] = f"Error: Failed to calculate TVD metrics. Details: {str(e)}"


    return JSONResponse(content={"message": "Evaluation complete.", "results": results})

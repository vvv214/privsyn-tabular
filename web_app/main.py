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
from .data_comparison import calculate_tvd_metrics
import sys # For sys.path modification
from fastapi.staticfiles import StaticFiles

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



# Add the project root to the sys.path to allow importing project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from .synthesis_service import run_synthesis, Args # Import Args from synthesis_service
import uuid # For generating unique IDs for temporary storage
from .data_inference import infer_data_metadata, load_dataframe_from_uploaded_file

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "https://www.privsyn.com"],  # Allow your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/synthesize")
async def synthesize_data(
    method: str = Form(...),
    dataset_name: str = Form(...),
    epsilon: float = Form(...),
    delta: float = Form(...),
    num_preprocess: str = Form(...),
    rare_threshold: float = Form(...),
    n_sample: int = Form(...),
    consist_iterations: int = Form(...),
    non_negativity: str = Form(...),
    append: bool = Form(...),
    sep_syn: bool = Form(...),
    initialize_method: str = Form(...),
    update_method: str = Form(...),
    update_rate_method: str = Form(...),
    update_rate_initial: float = Form(...),
    update_iterations: int = Form(...),
    data_file: UploadFile = File(..., description="Upload your dataset as a CSV or ZIP file."),
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
        logger.info(f"Domain data sent to frontend: {inferred_data_temp_storage[unique_id]["domain_data"]}")
        logger.info(f"Info data sent to frontend: {inferred_data_temp_storage[unique_id]["info_data"]}")
        return JSONResponse(content={
            "message": "Metadata inferred. Please confirm.",
            "unique_id": unique_id,
            "domain_data": inferred_data_temp_storage[unique_id]["domain_data"],
            "info_data": inferred_data_temp_storage[unique_id]["info_data"]
        })

    try:
        if data_file is None:
            raise HTTPException(status_code=400, detail="Data file is required for synthesis unless in debug mode.")
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

        # Save X_cat and X_num to temporary npy files if they exist
        X_cat_path = None
        if inferred_data['X_cat'] is not None:
            X_cat_path = os.path.join(temp_dir, "X_cat.npy")
            np.save(X_cat_path, inferred_data['X_cat'])
            logger.info(f"Saved X_cat to {X_cat_path}")

        X_num_path = None
        if inferred_data['X_num'] is not None:
            X_num_path = os.path.join(temp_dir, "X_num.npy")
            np.save(X_num_path, inferred_data['X_num'])
            logger.info(f"Saved X_num to {X_num_path}")

        inferred_data_temp_storage[unique_id] = {
            "df_path": df_path,
            "X_cat_path": X_cat_path,
            "X_num_path": X_num_path,
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
    consist_iterations: int = Form(...),
    non_negativity: str = Form(...),
    append: bool = Form(...),
    sep_syn: bool = Form(...),
    initialize_method: str = Form(...),
    update_method: str = Form(...),
    update_rate_method: str = Form(...),
    update_rate_initial: float = Form(...),
    update_iterations: int = Form(...),
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
    X_cat_path = temp_data["X_cat_path"]
    X_num_path = temp_data["X_num_path"]
    temp_dir = temp_data["temp_dir"]

    # Load df from temporary parquet file
    df = pd.read_parquet(df_path)
    logger.info(f"Loaded df from {df_path}")
    log_memory_usage("confirm_synthesis_after_df_load")

    # Load X_cat and X_num from temporary npy files if they exist
    X_cat = np.load(X_cat_path, allow_pickle=True) if X_cat_path else None
    if X_cat_path: logger.info(f"Loaded X_cat from {X_cat_path}")
    X_num = np.load(X_num_path, allow_pickle=True) if X_num_path else None
    if X_num_path: logger.info(f"Loaded X_num from {X_num_path}")

    target_column = temp_data["target_column"]
    synthesis_params = temp_data["synthesis_params"] # Retrieve synthesis parameters

    # Parse the JSON strings back into dictionaries
    domain_data = json.loads(confirmed_domain_data)
    info_data = json.loads(confirmed_info_data)

    try:
        # Create a persistent directory for this synthesis run (still needed for synthesized_csv_path)
        synthesis_run_dir = os.path.join(project_root, "temp_synthesis_output", "runs", unique_id)
        os.makedirs(synthesis_run_dir, exist_ok=True)

        # Write domain.json and info.json to disk for run_synthesis
        with open(os.path.join(synthesis_run_dir, "domain.json"), "w") as f:
            json.dump(domain_data, f, indent=4)
        with open(os.path.join(synthesis_run_dir, "info.json"), "w") as f:
            json.dump(info_data, f, indent=4)

        # Construct original_df from in-memory data
        df_parts = []
        
        num_col_names = [col for col in domain_data if domain_data[col]['type'] == 'numerical']
        if X_num is not None and num_col_names:
            if len(num_col_names) == X_num.shape[1]:
                df_parts.append(pd.DataFrame(X_num, columns=num_col_names))
            else:
                logger.error(f"Mismatch in numerical columns during original_df construction: domain.json has {len(num_col_names)}, X_num has {X_num.shape[1]}")
                df_parts.append(pd.DataFrame(X_num, columns=[f'num_col_{i}' for i in range(X_num.shape[1])]))

        cat_col_names = [col for col in domain_data if domain_data[col]['type'] == 'categorical']
        if X_cat is not None and cat_col_names:
            if len(cat_col_names) == X_cat.shape[1]:
                df_parts.append(pd.DataFrame(X_cat, columns=cat_col_names))
            else:
                logger.error(f"Mismatch in categorical columns during original_df construction: domain.json has {len(cat_col_names)}, X_cat has {X_cat.shape[1]}")
                df_parts.append(pd.DataFrame(X_cat, columns=[f'cat_col_{i}' for i in range(X_cat.shape[1])]))

        if not df_parts:
            raise ValueError("No numerical or categorical data found to construct original_df.")

        # Concatenate parts, ensuring column order based on domain.json
        all_domain_cols_ordered = list(domain_data.keys())
        combined_df = pd.concat(df_parts, axis=1)
        final_columns = [col for col in all_domain_cols_ordered if col in combined_df.columns]
        original_df = combined_df[final_columns]
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
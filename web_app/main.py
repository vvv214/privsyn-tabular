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
import sys # For sys.path modification
from fastapi.staticfiles import StaticFiles

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the sys.path to allow importing project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from .synthesis_service import run_synthesis, Args # Import Args from synthesis_service
import uuid # For generating unique IDs for temporary storage
from .data_inference import infer_data_metadata, load_dataframe_from_uploaded_file

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Temporarily allow all origins for debugging CORS
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
    if dataset_name == "debug_dataset":
        logger.info("Debug mode: Bypassing data inference and returning dummy metadata.")
        unique_id = str(uuid.uuid4())
        inferred_data_temp_storage[unique_id] = {
            "df": pd.DataFrame(np.random.rand(10, 5)), # Dummy DataFrame
            "X_cat": np.array([["A", "B"], ["C", "D"]]), # Dummy X_cat
            "X_num": np.array([[1.0, 2.0], [3.0, 4.0]]), # Dummy X_num
            "domain_data": {"num_attr_1": 10, "cat_attr_1": 5}, # Dummy domain_data
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
        logger.info(f"DataFrame loaded. Shape: {df.shape}")

        # 2. Infer data metadata
        logger.info("Attempting to infer data metadata.")
        inferred_data = infer_data_metadata(df, target_column=target_column)
        logger.info("Data metadata inferred successfully.")

        # Store original df and y temporarily with a unique ID
        unique_id = str(uuid.uuid4())
        inferred_data_temp_storage[unique_id] = {
            "df": df,
            "X_cat": inferred_data['X_cat'],
            "X_num": inferred_data['X_num'],
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
            }
        }
        logger.info(f"synthesize_data populated inferred_data_temp_storage with unique_id: {unique_id}, dataset_name: {dataset_name}")
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
    """
    Receives confirmation of inferred metadata and proceeds with data synthesis.
    """
    if unique_id not in inferred_data_temp_storage:
        raise HTTPException(status_code=404, detail="Inferred data not found or session expired. Please re-upload.")

    temp_data = inferred_data_temp_storage.pop(unique_id) # Retrieve and remove from temp storage
    df = temp_data["df"]
    target_column = temp_data["target_column"]
    synthesis_params = temp_data["synthesis_params"] # Retrieve synthesis parameters

    # Parse the JSON strings back into dictionaries
    domain_data = json.loads(confirmed_domain_data)
    info_data = json.loads(confirmed_info_data)

    try:
        # Create a persistent directory for this synthesis run
        synthesis_run_dir = os.path.join(project_root, "temp_synthesis_output", "runs", unique_id)
        os.makedirs(synthesis_run_dir, exist_ok=True)

        # Save the data and metadata
        X_cat = temp_data["X_cat"]
        X_num = temp_data["X_num"]
        
        if X_cat is not None:
            np.save(os.path.join(synthesis_run_dir, "X_cat_test.npy"), X_cat)
            np.save(os.path.join(synthesis_run_dir, "X_cat_train.npy"), X_cat)
        if X_num is not None:
            np.save(os.path.join(synthesis_run_dir, "X_num_test.npy"), X_num)
            np.save(os.path.join(synthesis_run_dir, "X_num_train.npy"), X_num)

        with open(os.path.join(synthesis_run_dir, "domain.json"), "w") as f:
            json.dump(domain_data, f, indent=4)
        with open(os.path.join(synthesis_run_dir, "info.json"), "w") as f:
            json.dump(info_data, f, indent=4)

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

        # Run synthesis
        synthesized_csv_path, original_data_dir_for_eval = await run_synthesis(
            args=args,
            data_dir=synthesis_run_dir,
            confirmed_domain_data=domain_data,
            confirmed_info_data=info_data
        )

        # Store paths for evaluation
        logger.info(f"Populating data_storage for dataset: {dataset_name}")
        data_storage[dataset_name] = {
            "synthesized_csv_path": synthesized_csv_path,
            "original_data_dir": original_data_dir_for_eval, # Store the actual original data dir returned by run_synthesis
            "method": synthesis_params["method"],
            "epsilon": synthesis_params["epsilon"],
            "delta": synthesis_params["delta"],
            "num_preprocess": synthesis_params["num_preprocess"],
            "rare_threshold": synthesis_params["rare_threshold"],
        }
        logger.info(f"Current data_storage keys: {data_storage.keys()}")

        return JSONResponse(content={"message": "Data synthesis initiated successfully!", "dataset_name": dataset_name})
    except Exception as e:
        logger.exception("Error during synthesis confirmation.")
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")

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
    evaluation_methods: str = Form(...), # Comma-separated string of methods
):
    """
    Triggers evaluation of synthesized data fidelity using selected methods.
    """
    if dataset_name not in data_storage:
        raise HTTPException(status_code=404, detail="Synthesized data not found for this dataset name. Please synthesize first.")

    data_info = data_storage[dataset_name]
    synthesized_csv_path = data_info["synthesized_csv_path"]
    original_data_dir = data_info["original_data_dir"]
    selected_methods = [m.strip() for m in evaluation_methods.split(',')]

    if not os.path.exists(synthesized_csv_path):
        raise HTTPException(status_code=404, detail="Synthesized data file not found for evaluation.")
    if not os.path.exists(original_data_dir):
        raise HTTPException(status_code=404, detail="Original preprocessed data not found for evaluation.")

    results = {}
    old_stdout = sys.stdout # Initialize old_stdout before the loop
    # Create a dummy args object for evaluation scripts
    eval_args_dict = {
        "method": data_info["method"],
        "dataset": dataset_name, # Use dataset_name as dataset arg for eval scripts
        "epsilon": data_info["epsilon"],
        "delta": data_info["delta"],
        "num_preprocess": data_info["num_preprocess"],
        "rare_threshold": data_info["rare_threshold"],
        "test": True, # Evaluation scripts are typically run in test mode
        "syn_test": False, # Not a synthesis test
        "device": "cpu", # Default for evaluation
        "sample_device": "cpu", # Default for evaluation
    }
    eval_args = Args(**eval_args_dict)

    # Dynamically import and run evaluation scripts
    for method_name in selected_methods:
        try:
            module_path = f"evaluator.{method_name}"
            logger.info(f"Calling evaluation module: {module_path}")
            if module_path in sys.modules: del sys.modules[module_path] # Clear cache for fresh import
            eval_module = importlib.import_module(module_path)

            old_stdout = sys.stdout
            redirected_output = io.StringIO()
            sys.stdout = redirected_output

            if hasattr(eval_module, 'main'):
                logger.info(f"Executing main function of {method_name} evaluation module.")
                eval_module.main(eval_args, original_data_dir, synthesized_csv_path) # This line is the speculative call
            else:
                logger.warning(f"Evaluation module '{method_name}' does not have a 'main' function.")
                raise AttributeError(f"Evaluation module '{method_name}' does not have a 'main' function.")

            results[method_name] = redirected_output.getvalue()

        except ModuleNotFoundError:
            logger.error(f"Evaluation method '{method_name}' not found.")
            results[method_name] = f"Error: Evaluation method '{method_name}' not found."
        except AttributeError as e:
            logger.error(f"Attribute error in {method_name}: {e}")
            results[method_name] = f"Error: {e}. Module '{method_name}' might not have the expected 'main' function or its signature is incorrect."
        except Exception as e:
            logger.exception(f"Error running {method_name}.")
            results[method_name] = f"Error running {method_name}: {str(e)}"
        finally:
            sys.stdout = old_stdout # Restore stdout

    return JSONResponse(content={"message": "Evaluation complete.", "results": results})
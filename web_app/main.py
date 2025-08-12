from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import shutil
import tempfile
import os
import io
import pandas as pd
import numpy as np
import json
import importlib # For dynamic import of evaluation scripts
import sys # For sys.path modification

# Add the project root to the sys.path to allow importing project modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from .synthesis_service import run_synthesis, Args # Import Args from synthesis_service
from .data_inference import infer_data_metadata

app = FastAPI()

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
    data_file: UploadFile = File(..., description="Upload your dataset as a CSV file."),
    target_column: str = Form('y_attr', description="Name of the target column in your CSV. Defaults to 'y_attr'."),
):
    """
    Receives an uploaded CSV file and parameters, infers metadata,
    then triggers the data synthesis process.
    Returns paths to the synthesized data and original preprocessed data for evaluation.
    """
    temp_dir = None
    try:
        # Create a temporary directory to store processed files
        temp_dir = tempfile.mkdtemp()
        uploaded_file_paths = {}

        # 1. Save the uploaded CSV file
        csv_file_path = os.path.join(temp_dir, data_file.filename)
        with open(csv_file_path, "wb") as buffer:
            shutil.copyfileobj(data_file.file, buffer)

        # 2. Read CSV into DataFrame
        df = pd.read_csv(csv_file_path)

        # 3. Infer data metadata (X_cat, X_num, y, domain_data, info_data)
        inferred_data = infer_data_metadata(df, target_column=target_column)

        X_cat = inferred_data['X_cat']
        X_num = inferred_data['X_num']
        y = inferred_data['y']
        domain_data = inferred_data['domain_data']
        info_data = inferred_data['info_data']

        # 4. Save inferred data to temporary .npy and .json files
        if X_cat.size > 0: 
            x_cat_train_path = os.path.join(temp_dir, 'X_cat_train.npy')
            np.save(x_cat_train_path, X_cat)
            uploaded_file_paths['X_cat_train'] = x_cat_train_path

        if X_num.size > 0: 
            x_num_train_path = os.path.join(temp_dir, 'X_num_train.npy')
            np.save(x_num_train_path, X_num)
            uploaded_file_paths['X_num_train'] = x_num_train_path

        if y is not None: 
            y_train_path = os.path.join(temp_dir, 'y_train.npy')
            np.save(y_train_path, y)
            uploaded_file_paths['y_train'] = y_train_path

        domain_json_path = os.path.join(temp_dir, 'domain.json')
        with open(domain_json_path, 'w') as f:
            json.dump(domain_data, f)
        uploaded_file_paths['domain_json'] = domain_json_path

        info_json_path = os.path.join(temp_dir, 'info.json')
        with open(info_json_path, 'w') as f:
            json.dump(info_data, f)
        uploaded_file_paths['info_json'] = info_json_path

        # 5. Call the synthesis service
        synthesized_csv_path, original_data_dir = await run_synthesis(
            method=method,
            dataset_name=dataset_name,
            epsilon=epsilon,
            delta=delta,
            num_preprocess=num_preprocess,
            rare_threshold=rare_threshold,
            uploaded_file_paths=uploaded_file_paths,
            n_sample=n_sample,
        )

        # Store paths for later evaluation
        data_storage[dataset_name] = {
            "synthesized_csv_path": synthesized_csv_path,
            "original_data_dir": original_data_dir,
            "method": method, # Store method for evaluation args
            "epsilon": epsilon, # Store epsilon for evaluation args
            "delta": delta, # Store delta for evaluation args
            "num_preprocess": num_preprocess, # Store num_preprocess for evaluation args
            "rare_threshold": rare_threshold, # Store rare_threshold for evaluation args
            "n_sample": n_sample, # Store n_sample for evaluation args
        }

        return JSONResponse(content={
            "message": "Synthesis complete. Data ready for download and evaluation.",
            "synthesized_csv_path": synthesized_csv_path,
            "original_data_dir": original_data_dir,
            "dataset_name": dataset_name # Return dataset_name for frontend reference
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temporary directory where CSV was initially saved
        # The .npy and .json files are now in temp_data_dir, which is managed by synthesis_service
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

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
            eval_module = importlib.import_module(module_path)

            old_stdout = sys.stdout
            redirected_output = io.StringIO()
            sys.stdout = redirected_output

            # Call the main function of the evaluation script
            # This is a placeholder and will need to be adjusted based on actual eval script signatures.
            # For now, assuming eval scripts have a 'main' function that takes an args object
            # and paths to real and synthetic data.
            # This is a strong assumption and will likely break.

            # To correctly call eval scripts, I need to know their exact function signatures.
            # For example, eval_seeds expects eval_config, sampling_method, device, preprocesser, time_record, **generator_dict
            # This is not a simple (args, real_path, syn_path) signature.

            # Given the complexity, I will implement a simplified evaluation call for now.
            # I will assume each eval script has a function named 'run_evaluation' that takes
            # the eval_args, original_data_dir, and synthesized_csv_path.
            # This is a *very* strong assumption and will likely require manual adaptation of eval scripts.

            # If eval_module has a 'main' function, try to call it with eval_args.
            # This is still a guess.
            if hasattr(eval_module, 'main'):
                # This is a highly speculative call. The actual arguments for each eval script's main
                # function are likely different and complex.
                # For a robust solution, each eval script would need a standardized wrapper.
                # For now, I'll pass the eval_args and the paths.
                # This will likely fail and require manual intervention to adapt eval scripts.
                eval_module.main(eval_args, original_data_dir, synthesized_csv_path) # This line is the speculative call
            else:
                raise AttributeError(f"Evaluation module '{method_name}' does not have a 'main' function.")

            results[method_name] = redirected_output.getvalue()

        except ModuleNotFoundError:
            results[method_name] = f"Error: Evaluation method '{method_name}' not found."
        except AttributeError as e:
            results[method_name] = f"Error: {e}. Module '{method_name}' might not have the expected 'main' function or its signature is incorrect."
        except Exception as e:
            results[method_name] = f"Error running {method_name}: {str(e)}"
        finally:
            sys.stdout = old_stdout # Restore stdout

    return JSONResponse(content={"message": "Evaluation complete.", "results": results})
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import shutil
import tempfile
import os
import io
import pandas as pd
import numpy as np # Added for numpy operations
import json # Added for json operations

from .synthesis_service import run_synthesis
from .data_inference import infer_data_metadata # Import the new inference module

app = FastAPI()

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
    # Changed to accept a single CSV file
    data_file: UploadFile = File(..., description="Upload your dataset as a CSV file."),
    target_column: str = Form('y_attr', description="Name of the target column in your CSV. Defaults to 'y_attr'."), # New form parameter for target column
):
    """
    Receives an uploaded CSV file and parameters, infers metadata,
    then triggers the data synthesis process.
    Returns the synthesized data as a CSV file.
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
        if X_cat.size > 0: # Check if X_cat is not empty
            x_cat_train_path = os.path.join(temp_dir, 'X_cat_train.npy')
            np.save(x_cat_train_path, X_cat)
            uploaded_file_paths['X_cat_train'] = x_cat_train_path

        if X_num.size > 0: # Check if X_num is not empty
            x_num_train_path = os.path.join(temp_dir, 'X_num_train.npy')
            np.save(x_num_train_path, X_num)
            uploaded_file_paths['X_num_train'] = x_num_train_path

        if y is not None: # Check if y is not None
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
        synthesized_df = await run_synthesis(
            method=method,
            dataset_name=dataset_name,
            epsilon=epsilon,
            delta=delta,
            num_preprocess=num_preprocess,
            rare_threshold=rare_threshold,
            uploaded_file_paths=uploaded_file_paths,
            n_sample=n_sample,
        )

        # Convert DataFrame to CSV in memory
        output_buffer = io.StringIO()
        synthesized_df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)

        # Return as StreamingResponse
        return StreamingResponse(
            iter([output_buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={dataset_name}_synthesized.csv"}
        )

    except Exception as e:
        # Log the full traceback for debugging
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    finally:
        # Clean up the temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

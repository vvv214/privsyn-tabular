from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import shutil
import tempfile
import os
import io
import pandas as pd

from .synthesis_service import run_synthesis

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
    x_cat_train: UploadFile = File(None),
    x_num_train: UploadFile = File(None),
    y_train: UploadFile = File(None),
    domain_json: UploadFile = File(...),
    info_json: UploadFile = File(...),
):
    """
    Receives uploaded dataset files and parameters, then triggers the data synthesis process.
    Returns the synthesized data as a CSV file.
    """
    temp_dir = None
    try:
        # Create a temporary directory to store uploaded files
        temp_dir = tempfile.mkdtemp()
        uploaded_file_paths = {}

        # Save uploaded files to the temporary directory
        if x_cat_train:
            x_cat_train_path = os.path.join(temp_dir, x_cat_train.filename)
            with open(x_cat_train_path, "wb") as buffer:
                shutil.copyfileobj(x_cat_train.file, buffer)
            uploaded_file_paths['X_cat_train'] = x_cat_train_path

        if x_num_train:
            x_num_train_path = os.path.join(temp_dir, x_num_train.filename)
            with open(x_num_train_path, "wb") as buffer:
                shutil.copyfileobj(x_num_train.file, buffer)
            uploaded_file_paths['X_num_train'] = x_num_train_path

        if y_train:
            y_train_path = os.path.join(temp_dir, y_train.filename)
            with open(y_train_path, "wb") as buffer:
                shutil.copyfileobj(y_train.file, buffer)
            uploaded_file_paths['y_train'] = y_train_path

        domain_json_path = os.path.join(temp_dir, domain_json.filename)
        with open(domain_json_path, "wb") as buffer:
            shutil.copyfileobj(domain_json.file, buffer)
        uploaded_file_paths['domain_json'] = domain_json_path

        info_json_path = os.path.join(temp_dir, info_json.filename)
        with open(info_json_path, "wb") as buffer:
            shutil.copyfileobj(info_json.file, buffer)
        uploaded_file_paths['info_json'] = info_json_path

        # Call the synthesis service
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
        return {"error": str(e)}
    finally:
        # Clean up the temporary directory
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
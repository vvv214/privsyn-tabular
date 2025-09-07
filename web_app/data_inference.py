import pandas as pd
import numpy as np
import json
import zipfile
import io
from fastapi import UploadFile

def load_dataframe_from_uploaded_file(file: UploadFile) -> pd.DataFrame:
    """
    Loads a Pandas DataFrame from an uploaded CSV or ZIP file.
    If a ZIP file, it extracts the first CSV file found within.
    """
    if file.filename.endswith('.csv'):
        return pd.read_csv(io.StringIO(file.file.read().decode('utf-8')))
    elif file.filename.endswith('.zip'):
        with zipfile.ZipFile(io.BytesIO(file.file.read()), 'r') as zf:
            csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
            if not csv_files:
                raise ValueError("No CSV file found inside the ZIP archive.")
            # Take the first CSV file found
            with zf.open(csv_files[0]) as csv_file:
                return pd.read_csv(io.StringIO(csv_file.read().decode('utf-8')))
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or ZIP file.")

def infer_data_metadata(df: pd.DataFrame, target_column: str = 'y_attr') -> dict:
    """
    Infers data types, creates X_cat, X_num, y arrays, and generates domain.json and info.json content.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.

    Returns:
        dict: A dictionary containing:
            - 'X_cat': numpy array of categorical features.
            - 'X_num': numpy array of numerical features.
            - 'y': numpy array of the target variable.
            - 'domain_data': Dictionary for domain.json.
            - 'info_data': Dictionary for info.json.
    """
    X_cat_cols = []
    X_num_cols = []

    domain_data = {}
    info_data = {
        "name": "UploadedDataset", # Placeholder, can be refined later
        "id": "uploaded-dataset-default", # Placeholder
        "task_type": "unknown", # Default, as target column is ignored
        "n_num_features": 0,
        "n_cat_features": 0,
        "n_classes": 0, # No target column, so n_classes is 0
        "train_size": len(df),
        "test_size": 0, # Cannot infer from single file
        "val_size": 0 # Cannot infer from single file
    }

    if target_column in df.columns:
        df = df.drop(columns=[target_column])

    num_feature_count = 0
    cat_feature_count = 0

    for col in df.columns:
        # Heuristic for categorical vs. numerical
        # If unique values are few (e.g., < 50 and < 5% of total rows), treat as categorical
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 50 and df[col].nunique() > len(df) * 0.05:
            # Treat as numerical
            num_feature_count += 1
            X_num_cols.append(col)
            # For domain.json, use nunique for numerical as per bank example
            domain_data[col] = {"type": "numerical", "size": df[col].nunique()}
        else:
            # Treat as categorical (strings, objects, or numerical with few unique values)
            cat_feature_count += 1
            X_cat_cols.append(col)
            domain_data[col] = {"type": "categorical", "size": df[col].nunique()}

    info_data["n_num_features"] = num_feature_count
    info_data["n_cat_features"] = cat_feature_count
    info_data["num_columns"] = X_num_cols
    info_data["cat_columns"] = X_cat_cols

    # Reorder df columns to match X_num_cols and X_cat_cols for consistent numpy array creation
    df_num = df[X_num_cols] if X_num_cols else pd.DataFrame()
    df_cat = df[X_cat_cols] if X_cat_cols else pd.DataFrame()

    # Return None when there are no corresponding columns to avoid ambiguous empty arrays downstream
    X_num_np = df_num.values if not df_num.empty else None
    X_cat_np = df_cat.astype(str).values if not df_cat.empty else None

    return {
        'X_cat': X_cat_np,
        'X_num': X_num_np,
        'domain_data': domain_data,
        'info_data': info_data
    }

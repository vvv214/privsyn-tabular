import pandas as pd
import numpy as np
import json

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
    X_cat = []
    X_num = []
    y = None

    domain_data = {}
    info_data = {
        "train_size": len(df),
        "n_classes": 0, # Will be updated if target_column is found
        "columns": [] # To store column names and their inferred types
    }

    # Separate target column
    if target_column in df.columns:
        y = df[target_column].values
        info_data["n_classes"] = len(np.unique(y)) # Assuming classification task
        df = df.drop(columns=[target_column])
    else:
        print(f"Warning: Target column '{target_column}' not found. Proceeding without a target variable.")

    for col in df.columns:
        # Try to infer data type
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if it's truly numerical or categorical represented as numbers
            # A simple heuristic: if unique values are few, treat as categorical
            if df[col].nunique() < len(df) * 0.05 and df[col].nunique() < 50: # Heuristic for categorical
                X_cat.append(df[col].astype(str).values) # Convert to string for categorical
                domain_data[col] = len(df[col].unique())
                info_data["columns"].append({"name": col, "type": "categorical"})
            else:
                X_num.append(df[col].values)
                domain_data[col] = {"min": df[col].min(), "max": df[col].max()}
                info_data["columns"].append({"name": col, "type": "numerical"})
        else:
            # Treat as categorical (e.g., strings, objects)
            X_cat.append(df[col].astype(str).values)
            domain_data[col] = len(df[col].unique())
            info_data["columns"].append({"name": col, "type": "categorical"})

    X_cat_np = np.array(X_cat).T if X_cat else np.array([])
    X_num_np = np.array(X_num).T if X_num else np.array([])

    return {
        'X_cat': X_cat_np,
        'X_num': X_num_np,
        'y': y,
        'domain_data': domain_data,
        'info_data': info_data
    }

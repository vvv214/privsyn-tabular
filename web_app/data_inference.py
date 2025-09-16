import pandas as pd
import numpy as np
import json
import zipfile
import io
from typing import Dict, Tuple

from fastapi import UploadFile


# Heuristics used for column type inference. These values are exported to the
# frontend so we can explain decisions like "age" being inferred as categorical.
INTEGER_UNIQUE_THRESHOLD = 20
INTEGER_UNIQUE_RATIO_THRESHOLD = 0.05
INTEGER_UNIQUE_MAX = 100
FLOAT_UNIQUE_THRESHOLD = 10
MAX_CATEGORICAL_PREVIEW_VALUES = 50
CATEGORY_NULL_TOKEN = "__NULL__"


def _serialize_value(value):
    """Convert numpy types into native Python types for JSON serialization."""
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    return value


def canonicalize_category(value) -> str:
    """Normalize category values to strings; NaNs become a sentinel token."""
    if pd.isna(value):
        return CATEGORY_NULL_TOKEN
    if isinstance(value, str):
        return value
    return str(value)


def _infer_column_type(col_name: str, series: pd.Series) -> Tuple[str, Dict[str, object]]:
    """Infer column type and capture the reasoning for the frontend."""

    is_numeric = pd.api.types.is_numeric_dtype(series)
    unique_values = int(series.nunique(dropna=False))
    total_rows = len(series)
    details: Dict[str, object] = {
        "column": col_name,
        "total_rows": total_rows,
        "unique_values": unique_values,
        "is_numeric_dtype": bool(is_numeric),
        "reasons": [],
    }

    # Default assumption
    inferred_type = "categorical" if not is_numeric else "numerical"

    if is_numeric:
        if pd.api.types.is_integer_dtype(series):
            if unique_values < INTEGER_UNIQUE_THRESHOLD:
                inferred_type = "categorical"
                details["reasons"].append(
                    {
                        "code": "integer_unique_below_threshold",
                        "message": (
                            f"Integer column has only {unique_values} distinct values, below the "
                            f"threshold of {INTEGER_UNIQUE_THRESHOLD}."
                        ),
                    }
                )
            elif (unique_values / max(total_rows, 1)) < INTEGER_UNIQUE_RATIO_THRESHOLD and unique_values < INTEGER_UNIQUE_MAX:
                inferred_type = "categorical"
                details["reasons"].append(
                    {
                        "code": "integer_sparse_unique",
                        "message": (
                            f"Integer column has sparse uniques ({unique_values}/{total_rows}); treating as categorical."
                        ),
                    }
                )
            else:
                inferred_type = "numerical"
                details["reasons"].append(
                    {
                        "code": "integer_dense_unique",
                        "message": "Integer column has sufficient variety; treating as numerical.",
                    }
                )
        elif pd.api.types.is_float_dtype(series):
            if unique_values < FLOAT_UNIQUE_THRESHOLD:
                inferred_type = "categorical"
                details["reasons"].append(
                    {
                        "code": "float_unique_low",
                        "message": (
                            f"Float column has only {unique_values} distinct values (< {FLOAT_UNIQUE_THRESHOLD})."
                        ),
                    }
                )
            else:
                inferred_type = "numerical"
                details["reasons"].append(
                    {
                        "code": "float_unique_high",
                        "message": "Float column has many unique values; treating as numerical.",
                    }
                )
        else:
            inferred_type = "numerical"
            details["reasons"].append(
                {
                    "code": "other_numeric",
                    "message": "Numeric dtype detected; defaulting to numerical.",
                }
            )
    else:
        details["reasons"].append(
            {
                "code": "non_numeric",
                "message": "Non-numeric dtype detected; treating as categorical.",
            }
        )

    details["inferred_type"] = inferred_type
    return inferred_type, details

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

    inference_report = {}

    for col in df.columns:
        series = df[col]
        inferred_type, reason_details = _infer_column_type(col, series)
        inference_report[col] = reason_details

        numeric_candidate = pd.to_numeric(series, errors="coerce")
        numeric_candidate_summary = None
        if not numeric_candidate.isna().all():
            numeric_candidate_summary = {
                "min": _serialize_value(numeric_candidate.min()),
                "max": _serialize_value(numeric_candidate.max()),
            }

        if inferred_type == "categorical":
            cat_feature_count += 1
            X_cat_cols.append(col)

            canonical_series = series.apply(canonicalize_category)
            value_counts = canonical_series.value_counts(dropna=False)
            categories_preview = value_counts.head(MAX_CATEGORICAL_PREVIEW_VALUES).index.tolist()
            categories_full = canonical_series.drop_duplicates().tolist()
            value_counts_dict = {str(idx): int(cnt) for idx, cnt in value_counts.items()}

            domain_data[col] = {
                "type": "categorical",
                "size": int(series.nunique()),
                "categories_preview": categories_preview,
                "value_counts_preview": {
                    str(idx): int(cnt)
                    for idx, cnt in value_counts.head(MAX_CATEGORICAL_PREVIEW_VALUES).items()
                },
                "categories": categories_full,
                "value_counts": value_counts_dict,
                "category_null_token": CATEGORY_NULL_TOKEN,
                "numeric_candidate_summary": numeric_candidate_summary,
            }
        else:
            num_feature_count += 1
            X_num_cols.append(col)
            series_numeric = pd.to_numeric(series, errors="coerce")
            col_min = _serialize_value(series_numeric.min()) if not series_numeric.isna().all() else None
            col_max = _serialize_value(series_numeric.max()) if not series_numeric.isna().all() else None
            domain_data[col] = {
                "type": "numerical",
                "size": int(series.nunique()),
                "numeric_summary": {
                    "min": col_min,
                    "max": col_max,
                    "mean": _serialize_value(series_numeric.mean()) if not series_numeric.isna().all() else None,
                    "std": _serialize_value(series_numeric.std()) if not series_numeric.isna().all() else None,
                },
                "numeric_candidate_summary": numeric_candidate_summary,
            }

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

    info_data["inference_settings"] = {
        "integer_unique_threshold": INTEGER_UNIQUE_THRESHOLD,
        "integer_unique_ratio_threshold": INTEGER_UNIQUE_RATIO_THRESHOLD,
        "integer_unique_max": INTEGER_UNIQUE_MAX,
        "float_unique_threshold": FLOAT_UNIQUE_THRESHOLD,
        "max_categorical_preview_values": MAX_CATEGORICAL_PREVIEW_VALUES,
    }
    info_data["inference_report"] = inference_report

    return {
        'X_cat': X_cat_np,
        'X_num': X_num_np,
        'domain_data': domain_data,
        'info_data': info_data
    }

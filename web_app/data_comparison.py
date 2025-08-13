# web_app/data_comparison.py

import pandas as pd
import numpy as np

def _calculate_single_column_tvd(original_series: pd.Series, synthesized_series: pd.Series) -> float:
    """
    Calculates the Total Variation Distance (TVD) for a single column.

    Args:
        original_series (pd.Series): Original data column.
        synthesized_series (pd.Series): Synthesized data column.

    Returns:
        float: The TVD for the column.
    """
    # Combine unique values from both series to ensure all categories are covered
    all_values = pd.concat([original_series, synthesized_series]).unique()

    # Calculate value counts and normalize to get probabilities
    original_counts = original_series.value_counts(normalize=True)
    synthesized_counts = synthesized_series.value_counts(normalize=True)

    tvd = 0.0
    for value in all_values:
        original_prob = original_counts.get(value, 0.0)
        synthesized_prob = synthesized_counts.get(value, 0.0)
        tvd += abs(original_prob - synthesized_prob)

    return tvd / 2.0 # TVD is half the sum of absolute differences

def calculate_tvd_metrics(original_df: pd.DataFrame, synthesized_df: pd.DataFrame) -> dict:
    """
    Calculates Total Variation Distance (TVD) for each column and the average TVD
    across all columns between original and synthesized dataframes.

    Args:
        original_df (pd.DataFrame): Original data.
        synthesized_df (pd.DataFrame): Synthesized data.

    Returns:
        dict: A dictionary containing TVD for each column and the average TVD.
    """
    tvd_per_column = {}
    common_columns = list(set(original_df.columns) & set(synthesized_df.columns))

    if not common_columns:
        return {"error": "No common columns found between original and synthesized dataframes."}

    for col in common_columns:
        tvd_per_column[col] = _calculate_single_column_tvd(original_df[col], synthesized_df[col])

    average_tvd = np.mean(list(tvd_per_column.values())) if tvd_per_column else 0.0

    return {
        "tvd_per_column": tvd_per_column,
        "average_tvd": average_tvd,
        "notes": "TVD is calculated for common columns. Lower TVD indicates better similarity."
    }

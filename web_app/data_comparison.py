# web_app/data_comparison.py

import pandas as pd
import numpy as np


def _categorical_tvd(original_series: pd.Series, synthesized_series: pd.Series) -> float:
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

def _numeric_tvd(original_series: pd.Series, synthesized_series: pd.Series, edges: list | None = None) -> float:
    original_numeric = pd.to_numeric(original_series, errors="coerce").dropna()
    synthesized_numeric = pd.to_numeric(synthesized_series, errors="coerce").dropna()

    combined = pd.concat([original_numeric, synthesized_numeric])
    if combined.empty:
        return 0.0

    if edges is not None and len(edges) >= 2:
        bins = np.asarray(edges, dtype=float)
    else:
        min_val = combined.min()
        max_val = combined.max()
        if min_val == max_val:
            bins = np.array([min_val, min_val + 1])
        else:
            bins = np.linspace(min_val, max_val, num=11)

    original_counts, _ = np.histogram(original_numeric, bins=bins)
    synthesized_counts, _ = np.histogram(synthesized_numeric, bins=bins)

    total_original = original_counts.sum()
    total_synth = synthesized_counts.sum()
    if total_original == 0 and total_synth == 0:
        return 0.0

    original_prob = original_counts / total_original if total_original else np.zeros_like(original_counts, dtype=float)
    synthesized_prob = synthesized_counts / total_synth if total_synth else np.zeros_like(synthesized_counts, dtype=float)

    return float(0.5 * np.abs(original_prob - synthesized_prob).sum())


def calculate_tvd_metrics(
    original_df: pd.DataFrame,
    synthesized_df: pd.DataFrame,
    domain_data: dict | None = None,
    info_data: dict | None = None,
) -> dict:
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

    domain_types = {}
    column_edges = {}
    if isinstance(domain_data, dict):
        for col, details in domain_data.items():
            if isinstance(details, dict):
                col_type = details.get("type")
                if col_type:
                    domain_types[col] = col_type
                binning = details.get("binning")
                if isinstance(binning, dict) and "edges" in binning:
                    column_edges[col] = binning["edges"]

    for col in common_columns:
        col_type = domain_types.get(col)
        if col_type == "numerical":
            edges = column_edges.get(col)
            tvd_per_column[col] = _numeric_tvd(original_df[col], synthesized_df[col], edges=edges)
        else:
            tvd_per_column[col] = _categorical_tvd(original_df[col], synthesized_df[col])

    average_tvd = np.mean(list(tvd_per_column.values())) if tvd_per_column else 0.0

    return {
        "tvd_per_column": tvd_per_column,
        "average_tvd": average_tvd,
        "notes": "TVD is calculated for common columns. Lower TVD indicates better similarity."
    }

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def split_df_by_type(
    df: pd.DataFrame, info: Dict[str, Any]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Return numerical and categorical numpy views based on metadata."""
    num_cols: Sequence[str] = info.get("num_columns", []) or []
    cat_cols: Sequence[str] = info.get("cat_columns", []) or []

    x_num = df[num_cols].to_numpy(dtype=float) if num_cols else None
    x_cat = df[cat_cols].astype(str).to_numpy() if cat_cols else None
    return x_num, x_cat


def enforce_dataframe_schema(
    out: pd.DataFrame,
    original_dtypes: Optional[Dict[str, Any]],
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
) -> pd.DataFrame:
    """Ensure sampled output matches expected column ordering and dtypes."""
    expected_cols = list(num_cols) + list(cat_cols)
    if len(expected_cols) == out.shape[1]:
        out.columns = expected_cols

    if original_dtypes:
        for col, dtype in original_dtypes.items():
            if col in out.columns:
                try:
                    if pd.api.types.is_integer_dtype(dtype):
                        out[col] = (
                            pd.to_numeric(out[col], errors="coerce")
                            .fillna(0)
                            .astype(dtype)
                        )
                    else:
                        out[col] = out[col].astype(dtype)
                except (ValueError, TypeError):
                    out[col] = pd.to_numeric(out[col], errors="coerce")
    else:
        for col in num_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        for col in cat_cols:
            if col in out.columns:
                out[col] = out[col].astype(str)

    return out

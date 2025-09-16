import pandas as pd
import numpy as np

from method.api.utils import enforce_dataframe_schema, split_df_by_type


def test_split_df_by_type_respects_info_order():
    df = pd.DataFrame({
        "num": [1.0, 2.0],
        "cat": ["a", "b"],
    })
    info = {"num_columns": ["num"], "cat_columns": ["cat"]}

    x_num, x_cat = split_df_by_type(df, info)

    assert x_num.shape == (2, 1)
    assert x_cat.shape == (2, 1)
    assert x_num.dtype == float
    assert x_cat.dtype == object


def test_enforce_dataframe_schema_restores_order_and_types():
    raw = pd.DataFrame([["1", "x"], ["2", "y"]])
    original = {"num": np.dtype("int64"), "cat": np.dtype("O")}

    reconciled = enforce_dataframe_schema(raw, original, ["num"], ["cat"])

    assert list(reconciled.columns) == ["num", "cat"]
    assert pd.api.types.is_integer_dtype(reconciled["num"])  # coerced to int
    assert reconciled["cat"].tolist() == ["x", "y"]


def test_enforce_dataframe_schema_handles_missing_dtypes():
    raw = pd.DataFrame([["10", "foo"], ["20", "bar"]])
    reconciled = enforce_dataframe_schema(raw, None, ["value"], ["label"])

    assert list(reconciled.columns) == ["value", "label"]
    assert pd.api.types.is_numeric_dtype(reconciled["value"])  # falls back to numeric coercion
    assert reconciled["label"].dtype == object

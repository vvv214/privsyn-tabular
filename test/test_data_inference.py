import os
import sys

import pandas as pd

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from web_app.data_inference import infer_data_metadata


def test_numeric_candidate_summary_available_for_sparse_integer():
    df = pd.DataFrame({
        "age": [20, 20, 30],  # low variety -> inferred categorical
        "city": ["NY", "SF", "NY"],
    })

    result = infer_data_metadata(df)
    domain = result["domain_data"]

    assert domain["age"]["type"] == "categorical"
    summary = domain["age"].get("numeric_candidate_summary")
    assert summary["min"] == 20
    assert summary["max"] == 30


def test_numeric_candidate_summary_missing_for_non_numeric():
    df = pd.DataFrame({"label": ["a", "b"]})
    result = infer_data_metadata(df)
    domain = result["domain_data"]
    assert domain["label"]["numeric_candidate_summary"] is None

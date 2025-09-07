import os, sys, io, json
import pandas as pd
import numpy as np
import pytest

# Ensure project root on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from web_app.main import app


@pytest.mark.slow
def test_full_flow_confirm_synthesis_small():
    client = TestClient(app)

    # Build a tiny DataFrame and upload as CSV
    df = pd.DataFrame({
        'num_a': np.random.randint(0, 5, size=80),
        'cat_b': np.random.choice(['A','B','C'], size=80),
    })
    csv_bytes = df.to_csv(index=False).encode('utf-8')

    synth_form = {
        "method": "privsyn",
        "dataset_name": "mini",
        "epsilon": "0.5",
        "delta": "1e-5",
        "num_preprocess": "uniform_kbins",
        "rare_threshold": "0.002",
        "n_sample": "20",
        "consist_iterations": "3",
        "non_negativity": "N3",
        "append": "true",
        "sep_syn": "false",
        "initialize_method": "singleton",
        "update_method": "S5",
        "update_rate_method": "U4",
        "update_rate_initial": "1.0",
        "update_iterations": "2",
        "target_column": "y_attr",
    }

    files = {"data_file": ("mini.csv", io.BytesIO(csv_bytes), "text/csv")}
    r = client.post("/synthesize", data=synth_form, files=files)
    assert r.status_code == 200
    payload = r.json()
    unique_id = payload["unique_id"]
    domain_data = payload["domain_data"]
    info_data = payload["info_data"]

    # Confirm and synthesize
    confirm_form = {
        "unique_id": unique_id,
        "method": "privsyn",
        "dataset_name": "mini",
        "epsilon": "0.5",
        "delta": "1e-5",
        "num_preprocess": "uniform_kbins",
        "rare_threshold": "0.002",
        "n_sample": "20",
        "consist_iterations": "3",
        "non_negativity": "N3",
        "append": "true",
        "sep_syn": "false",
        "initialize_method": "singleton",
        "update_method": "S5",
        "update_rate_method": "U4",
        "update_rate_initial": "1.0",
        "update_iterations": "2",
        "confirmed_domain_data": json.dumps(domain_data),
        "confirmed_info_data": json.dumps(info_data),
    }

    r2 = client.post("/confirm_synthesis", data=confirm_form)
    assert r2.status_code == 200
    out = r2.json()
    assert out["message"].lower().startswith("data synthesis initiated")
    assert out["dataset_name"] == "mini"


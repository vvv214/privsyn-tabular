import os, sys
import json
import pytest

# Skip if httpx is unavailable (needed by TestClient)
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

# Ensure project root on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from web_app.main import app


def test_root_and_hello():
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "message" in r.json()

    r = client.get("/hello/world")
    assert r.status_code == 200
    assert r.json()["message"] == "Hello, world"


def test_synthesize_debug_mode():
    client = TestClient(app)
    # Use dataset_name=debug_dataset to trigger debug branch (no upload needed)
    data = {
        "method": "privsyn",
        "dataset_name": "debug_dataset",
        "epsilon": "1.0",
        "delta": "1e-5",
        "num_preprocess": "uniform_kbins",
        "rare_threshold": "0.002",
        "n_sample": "100",
        "consist_iterations": "5",
        "non_negativity": "N3",
        "append": "true",
        "sep_syn": "false",
        "initialize_method": "singleton",
        "update_method": "S5",
        "update_rate_method": "U4",
        "update_rate_initial": "1.0",
        "update_iterations": "5",
        "target_column": "y_attr",
    }
    r = client.post("/synthesize", data=data)
    assert r.status_code == 200
    payload = r.json()
    assert "unique_id" in payload
    assert "domain_data" in payload
    assert "info_data" in payload

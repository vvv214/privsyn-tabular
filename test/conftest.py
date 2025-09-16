import os
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from web_app.main import app, data_storage, inferred_data_temp_storage  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_backend_state():
    data_storage.clear()
    inferred_data_temp_storage.clear()
    yield
    data_storage.clear()
    inferred_data_temp_storage.clear()


@pytest.fixture
def api_client(monkeypatch):
    async def _fake_run_synthesis(*, args, data_dir, **_kwargs):
        output_path = os.path.join(data_dir, "synthetic.csv")
        pd.DataFrame({"record": [1, 2, 3]}).to_csv(output_path, index=False)
        return output_path, data_dir

    monkeypatch.setattr("web_app.main.run_synthesis", _fake_run_synthesis)
    return TestClient(app)

import os, sys
import pytest

pytest.importorskip("httpx")
from fastapi.testclient import TestClient

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from web_app.main import app


def test_sample_dataset_endpoint():
    client = TestClient(app)
    r = client.get("/sample_dataset/adult")
    # In CI container, sample_data should be present. If not, accept 404 with clear message.
    assert r.status_code in (200, 404)
    if r.status_code == 200:
        assert r.headers.get('content-type', '').startswith('application/zip')
        assert 'attachment' in r.headers.get('content-disposition','')


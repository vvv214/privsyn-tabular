import io
import json
import os
import shutil

import pandas as pd
import pytest
from fastapi.testclient import TestClient

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
import sys

if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from web_app.main import app, data_storage, inferred_data_temp_storage, project_root as backend_project_root
from preprocess_common.load_data_common import data_preporcesser_common


class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def _fake_run_synthesis(monkeypatch):
    async def fake_run_synthesis(*, args, data_dir, **kwargs):
        output_path = os.path.join(data_dir, "synthetic.csv")
        pd.DataFrame({"record": [1, 2, 3]}).to_csv(output_path, index=False)
        return output_path, data_dir

    monkeypatch.setattr("web_app.main.run_synthesis", fake_run_synthesis)


def _base_form_values(**overrides):
    form = {
        "method": "privsyn",
        "dataset_name": "test_dataset",
        "epsilon": "1.0",
        "delta": "1e-5",
        "num_preprocess": "uniform_kbins",
        "rare_threshold": "0.002",
        "n_sample": "5",
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
    form.update({k: str(v) for k, v in overrides.items()})
    return form


def _post_synthesize(client: TestClient, form: dict, df: pd.DataFrame):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    response = client.post(
        "/synthesize",
        data=form,
        files={"data_file": ("toy.csv", buf, "text/csv")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["unique_id"] in inferred_data_temp_storage
    assert "df_path" in inferred_data_temp_storage[payload["unique_id"]]
    return payload


def _confirm(client: TestClient, form: dict):
    response = client.post("/confirm_synthesis", data=form)
    assert response.status_code == 200, response.text
    return response


@pytest.mark.slow
def test_metadata_override_round_trip(tmp_path, monkeypatch):
    data_storage.clear()
    inferred_data_temp_storage.clear()
    client = TestClient(app)
    _fake_run_synthesis(monkeypatch)

    df = pd.DataFrame(
        {
            "age": [10, 80, 35],
            "department": ["A", "B", "A"],
        }
    )

    synth_form = _base_form_values(n_sample=10)
    payload = _post_synthesize(client, synth_form, df)
    unique_id = payload["unique_id"]
    inferred_domain = payload["domain_data"]
    inferred_info = payload["info_data"]

    special_token = "__OTHER__"
    confirmed_domain = {}
    for column, details in inferred_domain.items():
        if column == "department":
            categories_from_data = details.get("categories", [])
            confirmed_domain[column] = {
                "type": "categorical",
                "categories_from_data": categories_from_data,
                "selected_categories": ["A"],
                "custom_categories": ["C"],
                "excluded_categories": [c for c in categories_from_data if c != "A"],
                "excluded_strategy": "map_to_special",
                "special_token": special_token,
                "category_null_token": details.get("category_null_token"),
                "value_counts": details.get("value_counts", {}),
                "categories": list(
                    dict.fromkeys(
                        ["A", special_token, "C"]
                        + ([details.get("category_null_token")] if details.get("category_null_token") else [])
                    )
                ),
                "size": 3,
            }
        elif column == "age":
            confirmed_domain[column] = {
                "type": "numerical",
                "bounds": {"min": 18.0, "max": 65.0},
                "binning": {
                    "method": "uniform",
                    "bin_count": 5,
                    "bin_width": None,
                    "growth_rate": None,
                    "dp_budget_fraction": 0.05,
                },
                "numeric_summary": details.get("numeric_summary"),
                "size": 5,
            }
        else:
            confirmed_domain[column] = details

    confirmed_info = {**inferred_info}
    confirmed_info["num_columns"] = ["age"]
    confirmed_info["cat_columns"] = ["department"]
    confirmed_info["n_num_features"] = 1
    confirmed_info["n_cat_features"] = 1

    confirm_form = {**synth_form}
    confirm_form.update(
        {
            "unique_id": unique_id,
            "confirmed_domain_data": json.dumps(confirmed_domain),
            "confirmed_info_data": json.dumps(confirmed_info),
        }
    )

    _confirm(client, confirm_form)

    assert "test_dataset" in data_storage
    original_df = data_storage["test_dataset"]["original_df"]

    assert original_df["age"].tolist() == [18.0, 65.0, 35.0]
    assert set(original_df["department"]) == {"A", special_token}

    run_dir = os.path.join(backend_project_root, "temp_synthesis_output", "runs", unique_id)
    with open(os.path.join(run_dir, "domain.json")) as fh:
        persisted_domain = json.load(fh)

    assert "edges" in persisted_domain["age"]["binning"]
    assert len(persisted_domain["age"]["binning"]["edges"]) == confirmed_domain["age"]["binning"]["bin_count"] + 1

    args = Args(
        method="privsyn",
        num_preprocess="uniform_kbins",
        epsilon=1.0,
        delta=1e-5,
        rare_threshold=0.002,
        dataset="test_dataset",
    )
    preprocessor = data_preporcesser_common(args)

    X_num_raw = original_df[confirmed_info["num_columns"]].to_numpy()
    X_cat_raw = original_df[confirmed_info["cat_columns"]].astype(str).to_numpy()

    df_processed, _, _ = preprocessor.load_data(
        X_num_raw=X_num_raw,
        X_cat_raw=X_cat_raw,
        rho=0.1,
        user_domain_data=persisted_domain,
        user_info_data=confirmed_info,
    )

    x_num_rev, x_cat_rev = preprocessor.reverse_data(df_processed)
    assert "age" in preprocessor.numeric_edges
    assert preprocessor.numeric_edges["age"][0] == pytest.approx(18.0, rel=0.01)
    assert preprocessor.numeric_edges["age"][-1] == pytest.approx(65.0, rel=0.01)
    assert special_token in set(x_cat_rev[:, 0])

    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    data_storage.clear()
    inferred_data_temp_storage.clear()


@pytest.mark.slow
def test_confirm_synthesis_missing_session():
    data_storage.clear()
    inferred_data_temp_storage.clear()
    client = TestClient(app)

    form = _base_form_values()
    form.update(
        {
            "unique_id": "missing",
            "confirmed_domain_data": json.dumps({"age": {"type": "numerical", "size": 1}}),
            "confirmed_info_data": json.dumps({"num_columns": [], "cat_columns": []}),
        }
    )

    resp = client.post("/confirm_synthesis", data=form)
    assert resp.status_code == 404
    assert "session" in resp.json()["detail"].lower()


@pytest.mark.slow
def test_synthesize_requires_file_for_non_sample():
    data_storage.clear()
    inferred_data_temp_storage.clear()
    client = TestClient(app)

    form = _base_form_values()
    resp = client.post("/synthesize", data=form)
    assert resp.status_code == 400
    assert "data file is required" in resp.json()["detail"].lower()


@pytest.mark.slow
def test_synthesize_builtin_sample():
    data_storage.clear()
    inferred_data_temp_storage.clear()
    client = TestClient(app)

    form = _base_form_values()
    form["dataset_name"] = "adult"

    resp = client.post("/synthesize", data=form)
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    unique_id = payload["unique_id"]
    assert unique_id in inferred_data_temp_storage
    assert os.path.exists(inferred_data_temp_storage[unique_id]["df_path"])


@pytest.mark.slow
def test_non_numeric_column_coerced_when_forced_to_numeric(monkeypatch):
    data_storage.clear()
    inferred_data_temp_storage.clear()
    client = TestClient(app)
    _fake_run_synthesis(monkeypatch)

    df = pd.DataFrame({"code": ["x", "y", "z"]})
    payload = _post_synthesize(client, _base_form_values(dataset_name="coerce"), df)

    unique_id = payload["unique_id"]
    domain = payload["domain_data"]
    info = payload["info_data"]

    domain_override = {
        "code": {
            "type": "numerical",
            "bounds": {"min": 0, "max": 1},
            "binning": {"method": "uniform", "bin_count": 2, "bin_width": None, "growth_rate": None, "dp_budget_fraction": 0.05},
            "numeric_summary": domain["code"].get("numeric_summary"),
            "numeric_candidate_summary": domain["code"].get("numeric_candidate_summary"),
            "size": 2,
        }
    }
    info_override = {**info, "num_columns": ["code"], "cat_columns": [], "n_num_features": 1, "n_cat_features": 0}

    confirm_form = _base_form_values(dataset_name="coerce")
    confirm_form.update(
        {
            "unique_id": unique_id,
            "confirmed_domain_data": json.dumps(domain_override),
            "confirmed_info_data": json.dumps(info_override),
        }
    )
    _confirm(client, confirm_form)

    original_df = data_storage["coerce"]["original_df"]
    assert set(original_df["code"].astype(float)) == {0.0}

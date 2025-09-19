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
from method.preprocess_common.load_data_common import data_preporcesser_common


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
    return response.json()


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

    confirm_payload = _confirm(client, confirm_form)
    session_id = confirm_payload["session_id"]

    assert session_id in data_storage
    original_df = data_storage[session_id]["original_df"]

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


def test_confirm_synthesis_rejects_mismatched_edges(monkeypatch):
    data_storage.clear()
    inferred_data_temp_storage.clear()
    client = TestClient(app)
    _fake_run_synthesis(monkeypatch)

    df = pd.DataFrame({"value": [1, 2, 3]})
    synth_form = _base_form_values(n_sample=5)
    payload = _post_synthesize(client, synth_form, df)

    confirmed_domain = payload["domain_data"]
    confirmed_info = payload["info_data"]

    numeric = confirmed_domain["value"]
    numeric["type"] = "numerical"
    numeric["binning"] = {
        "method": "uniform",
        "bin_count": 3,
        "edges": [0, 5],  # only two edges -> mismatch with bin_count
    }

    confirm_form = {
        **synth_form,
        "unique_id": payload["unique_id"],
        "confirmed_domain_data": json.dumps(confirmed_domain),
        "confirmed_info_data": json.dumps(confirmed_info),
    }

    response = client.post("/confirm_synthesis", data=confirm_form)
    assert response.status_code == 400
    body = response.json()
    assert body["detail"]["error"] == "invalid_binning"

@pytest.mark.slow
def test_categorical_resample_strategy_persists_domain(monkeypatch):
    data_storage.clear()
    inferred_data_temp_storage.clear()
    client = TestClient(app)
    _fake_run_synthesis(monkeypatch)

    df = pd.DataFrame(
        {
            "department": ["Sales", "Ops", "Sales", "HR"],
            "tenure": [1, 3, 5, 2],
        }
    )

    synth_form = _base_form_values(dataset_name="resample_dataset", n_sample=8)
    payload = _post_synthesize(client, synth_form, df)
    unique_id = payload["unique_id"]

    inferred_domain = payload["domain_data"]
    inferred_info = payload["info_data"]

    categories_from_data = inferred_domain["department"].get("categories", [])
    confirmed_domain = {
        "department": {
            "type": "categorical",
            "categories_from_data": categories_from_data,
            "selected_categories": ["Sales"],
            "custom_categories": ["Remote"],
            "excluded_categories": [c for c in categories_from_data if c != "Sales"],
            "excluded_strategy": "resample",
            "special_token": "__OTHER__",
            "category_null_token": inferred_domain["department"].get("category_null_token"),
            "value_counts": inferred_domain["department"].get("value_counts", {}),
            "categories": list(
                {"Sales", "Remote", *categories_from_data}
            ),
            "size": len({"Sales", "Remote", *categories_from_data}),
        },
        "tenure": inferred_domain["tenure"],
    }

    confirm_form = {**synth_form}
    confirm_form.update(
        {
            "unique_id": unique_id,
            "confirmed_domain_data": json.dumps(confirmed_domain),
            "confirmed_info_data": json.dumps(inferred_info),
        }
    )

    confirm_payload = _confirm(client, confirm_form)
    session_id = confirm_payload["session_id"]

    run_dir = os.path.join(backend_project_root, "temp_synthesis_output", "runs", unique_id)
    with open(os.path.join(run_dir, "domain.json")) as fh:
        persisted_domain = json.load(fh)

    dept = persisted_domain["department"]
    assert dept["excluded_strategy"] == "resample"
    assert "__OTHER__" not in dept["categories"]
    assert set(dept["categories"]) == set(confirmed_domain["department"]["categories"])

    data_entry = data_storage[session_id]
    original_df = data_entry["original_df"]
    assert set(original_df["department"]) == set(df["department"])

    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    data_storage.clear()
    inferred_data_temp_storage.clear()


@pytest.mark.slow
def test_force_categorical_to_numeric_generates_range(monkeypatch):
    data_storage.clear()
    inferred_data_temp_storage.clear()
    client = TestClient(app)
    _fake_run_synthesis(monkeypatch)

    df = pd.DataFrame(
        {
            "score_band": ["low", "medium", "high", "medium"],
        }
    )

    synth_form = _base_form_values(dataset_name="numeric_override", n_sample=6)
    payload = _post_synthesize(client, synth_form, df)
    unique_id = payload["unique_id"]
    inferred_domain = payload["domain_data"]
    inferred_info = payload["info_data"]

    confirmed_domain = {
        "score_band": {
            "type": "numerical",
            "bounds": {"min": 0.0, "max": 100.0},
            "binning": {
                "method": "uniform",
                "bin_count": 4,
                "bin_width": None,
                "growth_rate": None,
                "dp_budget_fraction": 0.05,
            },
            "numeric_summary": inferred_domain["score_band"].get("numeric_summary"),
            "numeric_candidate_summary": inferred_domain["score_band"].get("numeric_candidate_summary"),
            "size": 4,
        }
    }

    confirmed_info = {**inferred_info}
    confirmed_info["num_columns"] = ["score_band"]
    confirmed_info["cat_columns"] = []
    confirmed_info["n_num_features"] = 1
    confirmed_info["n_cat_features"] = 0

    confirm_form = {**synth_form}
    confirm_form.update(
        {
            "unique_id": unique_id,
            "confirmed_domain_data": json.dumps(confirmed_domain),
            "confirmed_info_data": json.dumps(confirmed_info),
        }
    )

    confirm_payload = _confirm(client, confirm_form)
    session_id = confirm_payload["session_id"]

    run_dir = os.path.join(backend_project_root, "temp_synthesis_output", "runs", unique_id)
    with open(os.path.join(run_dir, "domain.json")) as fh:
        persisted_domain = json.load(fh)

    score_meta = persisted_domain["score_band"]
    assert score_meta["type"] == "numerical"
    assert score_meta["bounds"] == {"min": 0.0, "max": 100.0}

    synthetic_entry = data_storage[session_id]
    coerced_values = synthetic_entry["original_df"]["score_band"]
    assert coerced_values.dtype.kind in {"f", "i"}
    assert (coerced_values >= 0.0).all() and (coerced_values <= 100.0).all()
    # Ensure coercion produced varied values rather than a constant array.
    assert len(set(coerced_values.round(5))) > 1

    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    data_storage.clear()
    inferred_data_temp_storage.clear()


@pytest.mark.slow
def test_custom_categories_deduplicate_case(monkeypatch):
    data_storage.clear()
    inferred_data_temp_storage.clear()
    client = TestClient(app)
    _fake_run_synthesis(monkeypatch)

    df = pd.DataFrame(
        {
            "status": ["VIP", "vip", "Standard", "standard"],
        }
    )

    synth_form = _base_form_values(dataset_name="case_normalize", n_sample=6)
    payload = _post_synthesize(client, synth_form, df)
    unique_id = payload["unique_id"]
    domain = payload["domain_data"]
    info = payload["info_data"]

    status_config = domain["status"].copy()
    status_config.update(
        {
            "selected_categories": ["VIP"],
            "custom_categories": ["Premium", "premium"],
            "excluded_categories": ["Standard"],
            "excluded_strategy": "map_to_special",
            "special_token": "__OTHER__",
        }
    )

    confirmed_domain = {"status": status_config}
    confirmed_info = {**info, "cat_columns": ["status"], "num_columns": [], "n_cat_features": 1, "n_num_features": 0}

    confirm_form = {**synth_form}
    confirm_form.update(
        {
            "unique_id": unique_id,
            "confirmed_domain_data": json.dumps(confirmed_domain),
            "confirmed_info_data": json.dumps(confirmed_info),
        }
    )

    confirm_payload = _confirm(client, confirm_form)
    session_id = confirm_payload["session_id"]
    entry = data_storage[session_id]
    status_meta = entry["domain_data"]["status"]

    assert status_meta["custom_categories"] == ["Premium"]
    assert status_meta["special_token"] == "__OTHER__"
    assert status_meta["categories"].count("Premium") == 1
    assert status_meta["categories"].count("VIP") == 1
    assert "__OTHER__" in status_meta["categories"]

    normalized_values = entry["original_df"]["status"].tolist()
    assert normalized_values.count("VIP") == 2
    assert normalized_values.count("__OTHER__") == 2

    run_dir = os.path.join(backend_project_root, "temp_synthesis_output", "runs", unique_id)
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    data_storage.clear()
    inferred_data_temp_storage.clear()



@pytest.mark.slow
def test_numerical_exponential_binning_edges_are_monotonic(monkeypatch):
    data_storage.clear()
    inferred_data_temp_storage.clear()
    client = TestClient(app)
    _fake_run_synthesis(monkeypatch)

    df = pd.DataFrame(
        {
            "value": [5, 10, 15, 20, 25, 30],
        }
    )

    synth_form = _base_form_values(dataset_name="exp_bins", n_sample=6)
    payload = _post_synthesize(client, synth_form, df)
    unique_id = payload["unique_id"]
    inferred_domain = payload["domain_data"]
    inferred_info = payload["info_data"]

    confirmed_domain = {
        "value": {
            "type": "numerical",
            "bounds": {"min": 0.0, "max": 40.0},
            "binning": {
                "method": "exponential",
                "bin_count": 4,
                "bin_width": None,
                "growth_rate": 1.5,
                "dp_budget_fraction": 0.05,
            },
            "numeric_summary": inferred_domain["value"].get("numeric_summary"),
            "size": 4,
        }
    }

    confirm_form = {**synth_form}
    confirm_form.update(
        {
            "unique_id": unique_id,
            "confirmed_domain_data": json.dumps(confirmed_domain),
            "confirmed_info_data": json.dumps(inferred_info),
        }
    )

    confirm_payload = _confirm(client, confirm_form)
    session_id = confirm_payload["session_id"]

    run_dir = os.path.join(backend_project_root, "temp_synthesis_output", "runs", unique_id)
    with open(os.path.join(run_dir, "domain.json")) as fh:
        persisted_domain = json.load(fh)

    edges = persisted_domain["value"]["binning"]["edges"]
    assert len(edges) == 5
    assert edges[0] == pytest.approx(0.0)
    assert edges[-1] == pytest.approx(40.0)
    assert all(left < right for left, right in zip(edges, edges[1:]))
    assert persisted_domain["value"]["binning"]["method"] == "exponential"

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
    detail = resp.json()["detail"]
    if isinstance(detail, dict):
        assert detail.get("error") in {"session_expired", "session_missing"}
    else:
        assert "session" in detail.lower()


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
    confirm_payload = _confirm(client, confirm_form)
    session_id = confirm_payload["session_id"]

    original_df = data_storage[session_id]["original_df"]
    coerced_values = original_df["code"].astype(float)
    assert len(set(coerced_values.round(5))) > 1

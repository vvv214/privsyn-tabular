import os
import pickle
import sys

import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from method.synthesis.privsyn import config
from method.synthesis.privsyn.lib_dataset.data_store import DataStore


def _build_args():
    return {
        "dataset": "unit_test_ds",
        "dataset_name": "unit_test_ds",
        "method": "privsyn",
        "epsilon": 1.0,
        "num_preprocess": "uniform_kbins",
        "rare_threshold": 0.002,
    }


def test_datastore_save_and_load_marginal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = _build_args()

    store = DataStore(args)
    marginals = [("a", "b"), ("b", "c")]
    store.save_marginal(marginals)

    loaded = store.load_marginal()
    assert loaded == marginals


def test_datastore_processed_and_records_paths(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    synthesized = tmp_path / "synth"
    marginal = tmp_path / "marg"
    dependency = tmp_path / "dep"

    monkeypatch.setattr(config, "RAW_DATA_PATH", str(raw) + os.sep, raising=False)
    monkeypatch.setattr(config, "PROCESSED_DATA_PATH", str(processed) + os.sep, raising=False)
    monkeypatch.setattr(config, "SYNTHESIZED_RECORDS_PATH", str(synthesized) + os.sep, raising=False)
    monkeypatch.setattr(config, "MARGINAL_PATH", str(marginal) + os.sep, raising=False)
    monkeypatch.setattr(config, "DEPENDENCY_PATH", str(dependency) + os.sep, raising=False)
    monkeypatch.setattr(
        config,
        "ALL_PATH",
        [str(raw), str(processed), str(synthesized), str(marginal), str(dependency)],
        raising=False,
    )

    monkeypatch.chdir(tmp_path)
    args = _build_args()

    store = DataStore(args)
    for path in config.ALL_PATH:
        assert os.path.isdir(path)
    assert os.path.isdir(store.parent_dir)

    processed_file = os.path.join(config.PROCESSED_DATA_PATH, args["dataset_name"])
    os.makedirs(os.path.dirname(processed_file), exist_ok=True)
    expected = {"records": [1, 2, 3]}
    with open(processed_file, "wb") as fh:
        pickle.dump(expected, fh)

    assert store.load_processed_data() == expected

    store.save_synthesized_records({"foo": 1})
    assert os.path.exists(store.synthesized_records_file)

    custom_dir = tmp_path / "custom"
    custom_dir.mkdir()
    store.save_synthesized_records({"bar": 2}, save_path=str(custom_dir))
    expected_name = f"{args['dataset_name']}_{args['epsilon']}"
    assert os.path.exists(custom_dir / expected_name)

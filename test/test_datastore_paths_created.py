import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from method.synthesis.privsyn.lib_dataset.data_store import DataStore


def test_datastore_generates_parent_dir(tmp_path, monkeypatch):
    exp_root = tmp_path / "exp_root"
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    synth = tmp_path / "synth"
    marginal = tmp_path / "marg"
    dependency = tmp_path / "dep"

    monkeypatch.setattr("method.synthesis.privsyn.config.EXPERIMENT_BASE_PATH", str(exp_root), raising=False)
    monkeypatch.setattr("method.synthesis.privsyn.config.RAW_DATA_PATH", str(raw), raising=False)
    monkeypatch.setattr("method.synthesis.privsyn.config.PROCESSED_DATA_PATH", str(processed), raising=False)
    monkeypatch.setattr("method.synthesis.privsyn.config.SYNTHESIZED_RECORDS_PATH", str(synth), raising=False)
    monkeypatch.setattr("method.synthesis.privsyn.config.MARGINAL_PATH", str(marginal), raising=False)
    monkeypatch.setattr("method.synthesis.privsyn.config.DEPENDENCY_PATH", str(dependency), raising=False)
    monkeypatch.setattr(
        "method.synthesis.privsyn.config.ALL_PATH",
        [str(raw), str(processed), str(synth), str(marginal), str(dependency)],
        raising=False,
    )
    args = {
        'dataset': 'unit_ds_tmp',
        'dataset_name': 'unit_ds_tmp',
        'method': 'privsyn',
        'epsilon': 0.9,
        'num_preprocess': 'uniform_kbins',
        'rare_threshold': 0.002,
    }
    ds = DataStore(args)
    assert os.path.isdir(ds.parent_dir)

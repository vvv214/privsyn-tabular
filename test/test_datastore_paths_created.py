import os, sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from method.privsyn.lib_dataset.data_store import DataStore


def test_datastore_generates_parent_dir(tmp_path):
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


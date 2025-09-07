import os, sys
import tempfile
import pickle

# Ensure project root on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from privsyn.lib_dataset.data_store import DataStore


def test_datastore_save_and_load_marginal(tmp_path):
    # Prepare args dict expected by DataStore
    args = {
        'dataset': 'unit_test_ds',
        'dataset_name': 'unit_test_ds',
        'method': 'privsyn',
        'epsilon': 1.0,
        'num_preprocess': 'uniform_kbins',
        'rare_threshold': 0.002,
        'output_dir': str(tmp_path),
    }

    ds = DataStore(args)
    # Fake marginals object
    marginals = [('a','b'), ('b','c')]
    ds.save_marginal(marginals)

    loaded = ds.load_marginal()
    assert loaded == marginals

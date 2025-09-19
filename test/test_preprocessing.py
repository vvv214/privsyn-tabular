import pandas as pd
import numpy as np
import os
import sys
import zipfile
import io
import json
import pytest # Import pytest

# Add project root to sys.path
# Assuming this script is in project_root/test/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import modules from web_app and method/preprocess_common
from web_app.data_inference import infer_data_metadata
from method.preprocess_common.load_data_common import data_preporcesser_common
from method.preprocess_common.preprocess import discretizer # Import discretizer for type checking

# Dummy Args class to mimic the behavior of args object
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# --- Configuration ---
DATA_DIR = os.path.join(project_root, "sample_data") # Use os.path.join for robustness
ZIP_FILE_NAME = "adult.csv.zip"
CSV_FILE_NAME = "adult.csv"

# --- Test Case ---
def test_preprocessing_pipeline():
    print("\n--- Starting Preprocessing Test (Pytest) ---")

    # 1. Load adult.csv.zip
    zip_file_path = os.path.join(DATA_DIR, ZIP_FILE_NAME)
    print(f"Loading data from: {zip_file_path}")
    assert os.path.exists(zip_file_path), f"ZIP file not found: {zip_file_path}"
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zf:
            with zf.open(CSV_FILE_NAME) as csv_file:
                df_original = pd.read_csv(io.StringIO(csv_file.read().decode('utf-8')))
        print(f"Original DataFrame loaded. Shape: {df_original.shape}")
        assert not df_original.empty, "Original DataFrame is empty."
        assert df_original.shape[1] > 0, "Original DataFrame has no columns."
    except Exception as e:
        pytest.fail(f"Error loading ZIP file: {e}")

    # 2. Simulate infer_data_metadata
    print("Simulating data inference...")
    inferred_data = infer_data_metadata(df_original.copy())
    X_num_raw = inferred_data['X_num']
    X_cat_raw = inferred_data['X_cat']
    domain_data = inferred_data['domain_data']
    info_data = inferred_data['info_data']
    
    assert X_num_raw is not None or X_cat_raw is not None, "No numerical or categorical data inferred."
    print(f"Inferred X_num_raw shape: {X_num_raw.shape if X_num_raw is not None else None}")
    print(f"Inferred X_cat_raw shape: {X_cat_raw.shape if X_cat_raw is not None else None}")
    # print(f"Inferred domain_data: {json.dumps(domain_data, indent=2)}") # Too verbose for pytest output

    # 3. Simulate data_preporcesser_common.load_data
    print("Simulating data preprocessing (fitting encoder)...")
    args_preprocesser = Args(
        method='privsyn',
        num_preprocess='uniform_kbins',
        epsilon=1.0,
        delta=1e-5,
        rare_threshold=0.002, # Added rare_threshold
        dataset='adult'
    )
    data_preprocesser = data_preporcesser_common(args_preprocesser)
    
    df_processed, domain_processed, _ = data_preprocesser.load_data(
        X_num_raw=X_num_raw,
        X_cat_raw=X_cat_raw,
        rho=0.1, # Dummy rho value
        user_domain_data=domain_data,
        user_info_data=info_data
    )
    assert not df_processed.empty, "Processed DataFrame is empty."
    assert df_processed.shape[1] > 0, "Processed DataFrame has no columns."
    print(f"Processed DataFrame shape: {df_processed.shape}")
    assert data_preprocesser.num_encoder is not None, "Numerical encoder not fitted."
    if data_preprocesser.num_encoder and hasattr(data_preprocesser.num_encoder, 'encoder'):
        assert hasattr(data_preprocesser.num_encoder.encoder, 'n_bins_'), "Num encoder n_bins_ not found."
        # print(f"Num encoder n_bins_: {data_preprocesser.num_encoder.encoder.n_bins_}")
        # print(f"Num encoder categories_: {data_preprocesser.num_encoder.encoder.categories_}")

    # 4. Create dummy synthesized_df that mimics problematic output
    print("Creating dummy synthesized DataFrame...")
    num_cols = info_data.get('num_columns', [])
    cat_cols = info_data.get('cat_columns', [])

    synthesized_df_dummy = df_processed.copy() # Start with processed df structure
    for col in df_processed.columns:
        if col in num_cols:
            # For numerical columns, create values that are out of bounds for bins
            # Use a mix, including some valid and some problematic values
            # The max value should be significantly larger than any expected bin index
            problematic_values = np.array([0, 1, 79, 80, 200, 28224, 10000, 50000], dtype=np.uint32)
            num_elements = df_processed.shape[0]
            synthesized_df_dummy[col] = np.resize(problematic_values, num_elements)
        elif col in cat_cols:
            # For categorical columns, create valid values (e.g., 0 to num_categories - 1)
            num_categories = domain_data[col]['size']
            synthesized_df_dummy[col] = np.random.randint(0, num_categories, df_processed.shape[0])
        # else: columns not in num_cols or cat_cols will retain their values from df_processed.copy() # This line was commented out

    print(f"Dummy Synthesized DataFrame created. Shape: {synthesized_df_dummy.shape}")
    print(f"Dummy Synthesized DF numerical columns min/max (before reverse):\n{synthesized_df_dummy[num_cols].min()}\n{synthesized_df_dummy[num_cols].max()}")

    # 5. Call data_preporcesser_common.reverse_data
    print("Calling reverse_data...")
    try:
        x_num_rev, x_cat_rev = data_preprocesser.reverse_data(synthesized_df_dummy)
        print("Reverse data successful!")
        assert x_num_rev is not None or x_cat_rev is not None, "Reversed data is empty."
        print(f"Reversed X_num shape: {x_num_rev.shape if x_num_rev is not None else None}")
        print(f"Reversed X_cat shape: {x_cat_rev.shape if x_cat_rev is not None else None}")
    except Exception as e:
        pytest.fail(f"Error during reverse_data: {e}")

    print("--- Test Finished ---")


def test_numeric_preprocessing_none_preserves_variation():
    args_preprocesser = Args(
        method='privsyn',
        num_preprocess='none',
        epsilon=1.0,
        delta=1e-5,
        rare_threshold=0.002,
        dataset='toy'
    )
    data_preprocesser = data_preporcesser_common(args_preprocesser)

    X_num_raw = np.array([[0.0], [5.0], [10.0]], dtype=float)
    domain_data = {
        'feature': {
            'type': 'numerical',
            'size': 3,
            'bounds': {'min': 0.0, 'max': 10.0},
        }
    }
    info_data = {
        'num_columns': ['feature'],
        'cat_columns': [],
        'n_num_features': 1,
        'n_cat_features': 0,
    }

    df_processed, _, _ = data_preprocesser.load_data(
        X_num_raw=X_num_raw,
        X_cat_raw=None,
        rho=0.1,
        user_domain_data=domain_data,
        user_info_data=info_data,
    )

    processed_values = df_processed['feature'].tolist()
    assert processed_values == pytest.approx([0.0, 0.5, 1.0])

    x_num_rev, x_cat_rev = data_preprocesser.reverse_data(df_processed.values)
    assert x_cat_rev is None
    assert x_num_rev.shape == (3, 1)
    assert x_num_rev[:, 0].tolist() == pytest.approx([0.0, 5.0, 10.0])


def test_numeric_edges_transform_into_bins():
    args_preprocesser = Args(
        method='privsyn',
        num_preprocess='uniform_kbins',
        epsilon=1.0,
        delta=1e-5,
        rare_threshold=0.002,
        dataset='toy'
    )
    data_preprocesser = data_preporcesser_common(args_preprocesser)

    X_num_raw = np.array([[1.0], [3.0], [7.9], [12.1]])
    domain_data = {
        'feature': {
            'type': 'numerical',
            'size': 4,
            'bounds': {'min': 0.0, 'max': 16.0},
            'binning': {
                'method': 'uniform',
                'bin_count': 4,
                'edges': [0.0, 4.0, 8.0, 12.0, 16.0],
            },
        }
    }
    info_data = {
        'num_columns': ['feature'],
        'cat_columns': [],
        'n_num_features': 1,
        'n_cat_features': 0,
    }

    df_processed, _, _ = data_preprocesser.load_data(
        X_num_raw=X_num_raw,
        X_cat_raw=None,
        rho=0.1,
        user_domain_data=domain_data,
        user_info_data=info_data,
    )

    processed_vals = df_processed['feature'].tolist()
    assert processed_vals == [0, 0, 1, 3]
    assert data_preprocesser.numeric_edges['feature'] == [0.0, 4.0, 8.0, 12.0, 16.0]


def test_numeric_edges_none_discretizer_preserve_values():
    args_preprocesser = Args(
        method='privsyn',
        num_preprocess='none',
        epsilon=1.0,
        delta=1e-5,
        rare_threshold=0.002,
        dataset='toy'
    )
    data_preprocesser = data_preporcesser_common(args_preprocesser)

    X_num_raw = np.array([[1.0], [3.0], [7.9], [12.1]])
    domain_data = {
        'feature': {
            'type': 'numerical',
            'size': 4,
            'bounds': {'min': 0.0, 'max': 16.0},
            'binning': {
                'method': 'uniform',
                'bin_count': 4,
                'edges': [0.0, 4.0, 8.0, 12.0, 16.0],
            },
        }
    }
    info_data = {
        'num_columns': ['feature'],
        'cat_columns': [],
        'n_num_features': 1,
        'n_cat_features': 0,
    }

    df_processed, _, _ = data_preprocesser.load_data(
        X_num_raw=X_num_raw,
        X_cat_raw=None,
        rho=0.1,
        user_domain_data=domain_data,
        user_info_data=info_data,
    )

    processed_vals = df_processed['feature'].tolist()
    # MinMaxScaler scales to [0,1]
    assert processed_vals == pytest.approx([
        0.0,
        (3.0 - 1.0) / (12.1 - 1.0),
        (7.9 - 1.0) / (12.1 - 1.0),
        1.0,
    ])
    assert data_preprocesser.numeric_edges['feature'] == [0.0, 4.0, 8.0, 12.0, 16.0]

    x_num_rev, _ = data_preprocesser.reverse_data(df_processed.values)
    assert x_num_rev[:, 0].tolist() == pytest.approx([1.0, 3.0, 7.9, 12.1], rel=1e-5)

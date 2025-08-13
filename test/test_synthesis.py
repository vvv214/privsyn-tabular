import pandas as pd
import numpy as np
import os
import sys
import zipfile
import io
import json
import pytest
import shutil
import tempfile

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import modules from web_app and preprocess_common
from web_app.data_inference import infer_data_metadata
from preprocess_common.load_data_common import data_preporcesser_common
from privsyn.privsyn import privsyn_main
from util.rho_cdp import cdp_rho

# Dummy Args class to mimic the behavior of args object
class Args:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# --- Configuration ---
DATA_DIR = os.path.join(project_root, "sample_data")
ZIP_FILE_NAME = "adult.csv.zip"
CSV_FILE_NAME = "adult.csv"

# --- Test Case ---
def test_gum_synthesis_pipeline():
    print("\n--- Starting GUM Synthesis Test (Pytest) ---")

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
    target_column = 'income' # Assuming 'income' is the target column in adult.csv
    inferred_data = infer_data_metadata(df_original.copy(), target_column=target_column)
    X_num_raw = inferred_data['X_num']
    X_cat_raw = inferred_data['X_cat']
    domain_data = inferred_data['domain_data']
    info_data = inferred_data['info_data']
    
    assert X_num_raw is not None or X_cat_raw is not None, "No numerical or categorical data inferred."
    print(f"Inferred X_num_raw shape: {X_num_raw.shape if X_num_raw is not None else None}")
    print(f"Inferred X_cat_raw shape: {X_cat_raw.shape if X_cat_raw is not None else None}")

    # 3. Simulate data_preporcesser_common.load_data
    print("Simulating data preprocessing (fitting encoder)...")
    args_preprocesser = Args(
        method='privsyn',
        num_preprocess='uniform_kbins',
        epsilon=1.0,
        delta=1e-5,
        rare_threshold=0.002,
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

    # 4. Simulate privsyn_main call
    print("Simulating privsyn_main call...")
    epsilon = 1.0
    delta = 1e-5
    total_rho = cdp_rho(epsilon, delta)

    args_privsyn = Args(
        method='privsyn',
        dataset='adult',
        epsilon=epsilon,
        delta=delta,
        num_preprocess='uniform_kbins',
        rare_threshold=0.002,
        is_cal_marginals=True,
        is_cal_depend=True,
        is_combine=True,
        marg_add_sensitivity=1.0,
        marg_sel_threshold=20000,
        non_negativity='N3',
        consist_iterations=501,
        initialize_method='singleton',
        update_method='S5',
        append=True,
        sep_syn=False,
        update_rate_method='U4',
        update_rate_initial=1.0,
        update_iterations=50
    )

    privsyn_result = privsyn_main(args_privsyn, df_processed, domain_processed, total_rho)
    privsyn_generator = privsyn_result["privsyn_generator"]
    assert privsyn_generator is not None, "PrivSyn generator not created."
    print("PrivSyn generator created.")

    # 5. Call privsyn_generator.syn
    print("Calling privsyn_generator.syn...")
    n_sample = 100 # Number of samples to synthesize
    temp_output_dir = tempfile.mkdtemp() # Create a temporary directory for output
    try:
        privsyn_generator.syn(n_sample, data_preprocesser, temp_output_dir)
        print("Synthesis successful!")

        # 6. Assertions on synthesized_df
        synthesized_df = privsyn_generator.synthesized_df
        assert synthesized_df is not None, "Synthesized DataFrame is None."
        assert not synthesized_df.empty, "Synthesized DataFrame is empty."
        assert synthesized_df.shape[0] == n_sample, f"Synthesized DataFrame has {synthesized_df.shape[0]} rows, expected {n_sample}."
        assert synthesized_df.shape[1] == df_processed.shape[1], "Synthesized DataFrame column count mismatch."
        print(f"Synthesized DataFrame shape: {synthesized_df.shape}")
        print(f"Synthesized DataFrame min: {synthesized_df.min().min()}, max: {synthesized_df.max().max()}")

    except Exception as e:
        pytest.fail(f"Error during synthesis: {e}")
    finally:
        shutil.rmtree(temp_output_dir) # Clean up temporary directory
        print(f"Cleaned up temporary directory: {temp_output_dir}")

    print("--- Test Finished ---")

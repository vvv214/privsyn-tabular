import sys
import os
import numpy as np
import pandas as pd
import json
import math
import copy
import shutil
from typing import Dict, Any, Tuple

# Add the project root to the sys.path to allow importing project modules
# Assuming the project root is two levels up from this file (web_app/synthesis_service.py)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import necessary modules from the main project
from privsyn.privsyn import privsyn_main, add_default_params
from preprocess_common.load_data_common import data_preprocessor_common
from util.rho_cdp import cdp_rho
from privsyn.lib_dataset.dataset import Dataset
from privsyn.lib_dataset.domain import Domain

class Args:
    """
    A dummy class to mimic the argparse.Namespace object.
    Attributes will be set dynamically based on the web request parameters.
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

async def run_synthesis(
    method: str,
    dataset_name: str, # This will be a placeholder for now, as we're using uploaded data
    epsilon: float,
    delta: float,
    num_preprocess: str,
    rare_threshold: float,
    uploaded_file_paths: Dict[str, str], # Dictionary of temporary file paths for uploaded data
    n_sample: int, # Number of samples to generate
    device: str = "cpu", # Default device, can be passed from frontend
    sample_device: str = "cpu", # Default sample device
    # Add other parameters that might be needed for `args`
) -> Tuple[str, str]: # Returns path to synthesized CSV and path to original preprocessed data dir
    """
    Runs the data synthesis process using the PrivSyn logic.

    Args:
        method (str): The synthesis method (e.g., 'privsyn').
        dataset_name (str): A name for the dataset (used for directory naming, not actual data loading).
        epsilon (float): Privacy parameter epsilon.
        delta (float): Privacy parameter delta.
        num_preprocess (str): Numerical preprocessing method.
        rare_threshold (float): Threshold for rare categories.
        uploaded_file_paths (Dict[str, str]): Dictionary mapping file type (e.g., 'X_cat_train', 'domain_json')
                                              to their temporary absolute paths.
        n_sample (int): Number of samples to generate.
        device (str): Device to use for computation (e.g., 'cpu', 'cuda').
        sample_device (str): Device to use for sampling.

    Returns:
        Tuple[str, str]: Path to the synthesized CSV file and path to the original preprocessed data directory.
    """
    print(f"Starting synthesis with method: {method}, dataset: {dataset_name}, epsilon: {epsilon}")

    # 1. Create a dummy args object
    args_dict = {
        "method": method,
        "dataset": dataset_name,
        "device": device,
        "epsilon": epsilon,
        "delta": delta,
        "num_preprocess": num_preprocess,
        "rare_threshold": rare_threshold,
        "sample_device": sample_device,
        "test": False, # Always False for synthesis via web app
        "syn_test": False, # Always False for synthesis via web app
    }
    args = Args(**args_dict)
    args = add_default_params(args) # Add default parameters as done in original main.py

    # 2. Create temporary directory for uploaded data
    temp_data_dir = os.path.join(project_root, "temp_uploaded_data", dataset_name)
    os.makedirs(temp_data_dir, exist_ok=True)

    # 3. Save uploaded files to temp_data_dir
    try:
        if 'X_cat_train' in uploaded_file_paths: np.save(os.path.join(temp_data_dir, 'X_cat_train.npy'), np.load(uploaded_file_paths['X_cat_train']))
        if 'X_num_train' in uploaded_file_paths: np.save(os.path.join(temp_data_dir, 'X_num_train.npy'), np.load(uploaded_file_paths['X_num_train']))
        if 'y_train' in uploaded_file_paths: np.save(os.path.join(temp_data_dir, 'y_train.npy'), np.load(uploaded_file_paths['y_train']))

        with open(uploaded_file_paths['domain_json'], 'r') as f_in:
            domain_data = json.load(f_in)
        with open(os.path.join(temp_data_dir, 'domain.json'), 'w') as f_out:
            json.dump(domain_data, f_out)

        with open(uploaded_file_paths['info_json'], 'r') as f_in:
            info_data = json.load(f_in)
        with open(os.path.join(temp_data_dir, 'info.json'), 'w') as f_out:
            json.dump(info_data, f_out)

    except KeyError as e:
        raise ValueError(f"Missing required uploaded file: {e}. Please ensure all necessary data files are provided.")
    except Exception as e:
        raise RuntimeError(f"Error processing uploaded data: {e}")

    # 4. Calculate total_rho
    total_rho = cdp_rho(args.epsilon, args.delta)

    # 5. Instantiate data_preprocessor_common and load data
    data_preprocesser = data_preprocessor_common(args)
    df_processed, domain_processed, preprocesser_divide = data_preprocesser.load_data(temp_data_dir + '/', total_rho)

    # 6. Call privsyn_main
    privsyn_result = privsyn_main(args, df_processed, domain_processed, total_rho)
    privsyn_generator = privsyn_result["privsyn_generator"]

    # 7. Create temporary output directory for synthesis results
    temp_output_dir = os.path.join(project_root, "temp_synthesis_output", dataset_name)
    os.makedirs(temp_output_dir, exist_ok=True)

    # 8. Call privsyn_generator.syn
    privsyn_generator.syn(n_sample, data_preprocesser, temp_output_dir)

    # 9. Retrieve synthesized_df and save to CSV
    synthesized_df = privsyn_generator.synthesized_df
    synthesized_csv_path = os.path.join(temp_output_dir, f"{dataset_name}_synthesized.csv")
    synthesized_df.to_csv(synthesized_csv_path, index=False)

    # 10. Return paths to synthesized CSV and original preprocessed data dir
    return synthesized_csv_path, temp_data_dir
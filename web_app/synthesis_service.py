import sys
import os
import numpy as np
import pandas as pd
import json
import math
import copy
import shutil
import logging
from typing import Dict, Any, Tuple, Callable, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to the sys.path to allow importing project modules
# Assuming the project root is two levels up from this file (web_app/synthesis_service.py)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import necessary modules from the main project
from privsyn.privsyn import privsyn_main, add_default_params
from preprocess_common.load_data_common import data_preporcesser_common
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
    args: Args, # Pass the Args object directly
    data_dir: str, # Path to the directory containing X_cat.npy, X_num.npy, domain.json, info.json
    X_num_raw: np.ndarray,
    X_cat_raw: np.ndarray,
    confirmed_domain_data: dict, # User-confirmed domain data
    confirmed_info_data: dict, # User-confirmed info data
    consist_iterations: int = 501,
    non_negativity: str = 'N3',
    append: bool = True,
    sep_syn: bool = False,
    initialize_method: str = 'singleton',
    update_method: str = 'S5',
    update_rate_method: str = 'U4',
    update_rate_initial: float = 1.0,
    update_iterations: int = 50,
    progress_report: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[str, str]: # Returns path to synthesized CSV and path to original preprocessed data dir
    """
    Runs the data synthesis process using the PrivSyn logic.

    Args:
        args (Args): An Args object containing all necessary synthesis parameters.
        data_dir (str): The absolute path to the directory containing the preprocessed
                        data files (X_cat.npy, X_num.npy, y.npy) and metadata (domain.json, info.json).

    Returns:
        Tuple[str, str]: Path to the synthesized CSV file and path to the original preprocessed data directory.
    """
    dataset_name = args.dataset # Get dataset_name from args
    n_sample = args.n_sample # Get n_sample from args

    if dataset_name == "debug_dataset":
        logger.info("Debug mode: Returning dummy synthesized data for evaluation.")
        dummy_synthesized_path = os.path.join(project_root, "temp_synthesis_output", "debug_data", "debug_synthesized.csv")
        dummy_original_data_dir = os.path.join(project_root, "temp_synthesis_output", "debug_data", "original_data_for_eval")
        return dummy_synthesized_path, dummy_original_data_dir

    logger.info(f"Starting synthesis with method: {args.method}, dataset: {args.dataset}, epsilon: {args.epsilon}")

    # No need to create dummy args or copy files, as data_dir is already prepared
    args = add_default_params(args) # Add default parameters as done in original main.py

    # 2. Data is already in data_dir, so no need to create temp_data_dir or save files here.

    # 3. Load domain.json and info.json from data_dir
    logger.info("Loading domain.json and info.json from data_dir.")
    with open(os.path.join(data_dir, 'domain.json'), 'r') as f:
        domain_data = json.load(f)
    with open(os.path.join(data_dir, 'info.json'), 'r') as f:
        info_data = json.load(f)
    logger.info("Metadata loaded successfully.")

    # 4. Calculate total_rho
    logger.info("Calculating total_rho.")
    total_rho = cdp_rho(args.epsilon, args.delta)
    logger.info(f"Total rho calculated: {total_rho}")

    # 5. Instantiate data_preprocessor_common and load data
    logger.info("Instantiating data_preprocessor_common and loading data.")
    if progress_report:
        progress_report({"status": "running", "stage": "preprocess", "overall_step": 1, "overall_total": 5, "message": "Preprocessing data"})
    data_preprocesser = data_preporcesser_common(args)
    df_processed, domain_processed, preprocesser_divide = data_preprocesser.load_data(
        X_num_raw,
        X_cat_raw,
        total_rho,
        user_domain_data=confirmed_domain_data,
        user_info_data=confirmed_info_data
    )
    logger.info("Data loaded and preprocessed.")

    # 6. Call privsyn_main
    logger.info("Calling privsyn_main to initialize PrivSyn generator.")
    privsyn_result = privsyn_main(args, df_processed, domain_processed, total_rho)
    privsyn_generator = privsyn_result["privsyn_generator"]
    logger.info("PrivSyn generator initialized.")
    if progress_report:
        progress_report({"status": "running", "stage": "marginal_selection", "overall_step": 2, "overall_total": 5, "message": "Marginal selection complete"})

    # 7. Create temporary output directory for synthesis results
    temp_output_dir = os.path.join(project_root, "temp_synthesis_output", dataset_name)
    logger.info(f"Creating temporary output directory for synthesis results: {temp_output_dir}")
    os.makedirs(temp_output_dir, exist_ok=True)

    # 8. Call privsyn_generator.syn
    logger.info(f"Calling privsyn_generator.syn to perform synthesis for {n_sample} samples.")
    if progress_report:
        progress_report({"status": "running", "stage": "consistency", "overall_step": 3, "overall_total": 5, "message": "Consistency + update starting"})
    privsyn_generator.syn(n_sample, data_preprocesser, temp_output_dir, progress_report=progress_report)
    logger.info("Synthesis complete.")

    # 9. Retrieve synthesized_df, rename columns, and save to CSV
    logger.info("Retrieving synthesized_df and saving to CSV.")
    synthesized_df = privsyn_generator.synthesized_df
    
    # Rename columns to original names
    num_cols = info_data.get('num_columns', [])
    cat_cols = info_data.get('cat_columns', [])
    original_cols = num_cols + cat_cols
    synthesized_df.columns = original_cols

    synthesized_csv_path = os.path.join(temp_output_dir, f"{dataset_name}_synthesized.csv")
    synthesized_df.to_csv(synthesized_csv_path, index=False)
    logger.info(f"Synthesized data saved to: {synthesized_csv_path}")

    # 10. Return paths to synthesized CSV and original preprocessed data dir
    if progress_report:
        progress_report({"status": "running", "stage": "save", "overall_step": 5, "overall_total": 5, "message": "Saving outputs"})
    logger.info("Synthesis process finished. Returning paths.")
    return synthesized_csv_path, data_dir

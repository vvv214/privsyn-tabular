import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Tuple, Callable, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Import necessary modules from the main project
from .methods_dispatcher import synthesize as dispatch_synthesize

class Args:
    """
    A dummy class to mimic the argparse.Namespace object.
    Attributes will be set dynamically based on the web request parameters.
    """
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

async def run_synthesis(
    args: Args,  # synthesis parameters
    data_dir: str,  # output directory for run artifacts
    X_num_raw: np.ndarray,
    X_cat_raw: np.ndarray,
    confirmed_domain_data: dict,  # user-confirmed domain data
    confirmed_info_data: dict,  # user-confirmed info data
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
) -> Tuple[str, str]:
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

    # 2. Create temporary output directory for synthesis results
    temp_output_dir = os.path.join(project_root, "temp_synthesis_output", dataset_name)
    logger.info(f"Creating temporary output directory for synthesis results: {temp_output_dir}")
    os.makedirs(temp_output_dir, exist_ok=True)

    # 3. Reconstruct original DataFrame (before preprocessing) from raw arrays and confirmed info
    num_cols = confirmed_info_data.get('num_columns', []) or []
    cat_cols = confirmed_info_data.get('cat_columns', []) or []
    parts = []
    if X_num_raw is not None and len(num_cols) > 0:
        parts.append(pd.DataFrame(X_num_raw, columns=num_cols))
    if X_cat_raw is not None and len(cat_cols) > 0:
        parts.append(pd.DataFrame(X_cat_raw, columns=cat_cols))
    if not parts:
        raise ValueError("No input features provided for synthesis.")
    df_original = pd.concat(parts, axis=1)

    # 4. Build config and dispatch to selected method
    config = {
        'dataset': dataset_name,
        'epsilon': args.epsilon,
        'delta': args.delta,
        'num_preprocess': args.num_preprocess,
        'rare_threshold': args.rare_threshold,
    }

    # Surface advanced knobs so native synthesizers (e.g., PrivSyn) can honour
    # overrides from the UI instead of silently falling back to defaults.
    advanced_keys = (
        'consist_iterations',
        'non_negativity',
        'append',
        'sep_syn',
        'initialize_method',
        'update_method',
        'update_rate_method',
        'update_rate_initial',
        'update_iterations',
        'degree',
        'max_cells',
        'max_iters',
        'max_model_size',
        'num_marginals',
    )
    for key in advanced_keys:
        if hasattr(args, key):
            value = getattr(args, key)
            if value is not None:
                config[key] = value

    logger.info(f"Dispatching synthesis method: {args.method}")
    synth_df = dispatch_synthesize(
        method=args.method,
        df=df_original,
        user_domain_data=confirmed_domain_data,
        user_info_data=confirmed_info_data,
        config=config,
        n_sample=n_sample,
    )

    # 5. Save synthesized CSV
    synthesized_csv_path = os.path.join(temp_output_dir, f"{dataset_name}_synthesized.csv")
    synth_df.to_csv(synthesized_csv_path, index=False)
    logger.info(f"Synthesized data saved to: {synthesized_csv_path}")

    if progress_report:
        progress_report({"status": "running", "stage": "save", "overall_step": 5, "overall_total": 5, "message": "Saving outputs"})
    return synthesized_csv_path, data_dir

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from preprocess_common.load_data_common import data_preporcesser_common
from util.rho_cdp import cdp_rho
from method.privsyn.privsyn import privsyn_main, add_default_params
from method.api import register_adapter


logger = logging.getLogger(__name__)


@dataclass
class AdapterArgs:
    dataset: str = "adapter_dataset"
    method: str = "privsyn"
    epsilon: float = 1.0
    delta: float = 1e-5
    num_preprocess: str = "uniform_kbins"
    rare_threshold: float = 0.002


def _split_df(df: pd.DataFrame, info: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    num_cols = info.get("num_columns", []) or []
    cat_cols = info.get("cat_columns", []) or []

    X_num = df[num_cols].to_numpy(dtype=float) if len(num_cols) > 0 else None
    X_cat = df[cat_cols].astype(str).to_numpy() if len(cat_cols) > 0 else None
    return X_num, X_cat


def prepare(
    df: pd.DataFrame,
    user_domain_data: Dict[str, Any],
    user_info_data: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = config or {}
    args = AdapterArgs(
        dataset=config.get("dataset", "adapter_dataset"),
        method="privsyn",
        epsilon=float(config.get("epsilon", 1.0)),
        delta=float(config.get("delta", 1e-5)),
        num_preprocess=config.get("num_preprocess", "uniform_kbins"),
        rare_threshold=float(config.get("rare_threshold", 0.002)),
    )

    total_rho = cdp_rho(args.epsilon, args.delta)
    preprocesser = data_preporcesser_common(args)
    X_num_raw, X_cat_raw = _split_df(df, user_info_data)

    df_processed, domain_sizes, _ = preprocesser.load_data(
        X_num_raw,
        X_cat_raw,
        total_rho,
        user_domain_data=user_domain_data,
        user_info_data=user_info_data,
    )

    bundle = {
        "args": args,
        "df_processed": df_processed,
        "domain_list": domain_sizes,
        "preprocesser": preprocesser,
        "user_info": user_info_data,
        "user_domain": user_domain_data,
    }
    return bundle


def run(
    bundle: Dict[str, Any],
    epsilon: Optional[float] = None,
    delta: Optional[float] = None,
    seed: Optional[int] = None,
    n_sample: Optional[int] = None,
    progress_report=None,
) -> pd.DataFrame:
    args: AdapterArgs = bundle["args"]
    if epsilon is not None:
        args.epsilon = float(epsilon)
    if delta is not None:
        args.delta = float(delta)

    total_rho = cdp_rho(args.epsilon, args.delta)
    preprocesser = bundle["preprocesser"]
    df_processed = bundle["df_processed"]
    domain_list = bundle["domain_list"]

    args_obj = add_default_params(args)

    privsyn_result = privsyn_main(args_obj, df_processed, domain_list, total_rho)
    generator = privsyn_result["privsyn_generator"]

    if n_sample is None:
        n_sample = df_processed.shape[0]

    generator.syn(n_sample, preprocesser, parent_dir=None, progress_report=progress_report)
    out: pd.DataFrame = generator.synthesized_df
    # Ensure column names restored to original
    info = bundle["user_info"]
    num_cols = info.get("num_columns", []) or []
    cat_cols = info.get("cat_columns", []) or []
    expected_cols = num_cols + cat_cols
    if len(expected_cols) == out.shape[1]:
        out.columns = expected_cols
    return out

# Register this adapter into the unified registry at import time
try:
    register_adapter("privsyn", prepare, run)
except Exception as _e:
    # Avoid import hard-fail in environments that import modules in arbitrary order
    pass

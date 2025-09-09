import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from preprocess_common.load_data_common import data_preporcesser_common
from util.rho_cdp import cdp_rho
from method.AIM.aim import aim_main, add_default_params as aim_add_defaults
from method.api import register_adapter


logger = logging.getLogger(__name__)


@dataclass
class AdapterArgs:
    """Lightweight args container compatible with AIM add_default_params.

    Only includes fields we actually use; add more as needed.
    """

    dataset: str = "adapter_dataset"
    method: str = "aim"
    epsilon: float = 1.0
    delta: float = 1e-5
    num_preprocess: str = "uniform_kbins"
    rare_threshold: float = 0.002
    # AIM-specific knobs (keep defaults modest and CPU-friendly)
    degree: int = 2
    max_cells: int = 250000
    max_iters: int = 1000
    max_model_size: int = 80


def _split_df(df: pd.DataFrame, info: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    num_cols = info.get("num_columns", []) or []
    cat_cols = info.get("cat_columns", []) or []

    X_num = df[num_cols].to_numpy(dtype=float) if len(num_cols) > 0 else None
    X_cat = df[cat_cols].astype(str).to_numpy() if len(cat_cols) > 0 else None
    return X_num, X_cat


def _make_aim_domain_mapping(df_processed: pd.DataFrame, user_domain: Dict[str, Any]) -> Dict[str, int]:
    """Build AIM Domain mapping: {column_name: cardinality}.

    This uses the processed DataFrame's column order and looks up sizes from user_domain.
    """
    mapping: Dict[str, int] = {}
    for col in df_processed.columns:
        meta = user_domain.get(col, {})
        size = meta.get("size")
        if not isinstance(size, int):
            # Fallback: infer from data if size missing
            size = int(df_processed[col].max()) + 1
        mapping[col] = size
    return mapping


def prepare(
    df: pd.DataFrame,
    user_domain_data: Dict[str, Any],
    user_info_data: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Prepare inputs for AIM using the shared preprocessing utilities.

    Returns a bundle consumable by run().
    """
    config = config or {}
    if config.get("random_state") is not None:
        np.random.seed(config["random_state"])

    args = AdapterArgs(
        dataset=config.get("dataset", "adapter_dataset"),
        method="aim",
        epsilon=float(config.get("epsilon", 1.0)),
        delta=float(config.get("delta", 1e-5)),
        num_preprocess=config.get("num_preprocess", "uniform_kbins"),
        rare_threshold=float(config.get("rare_threshold", 0.002)),
        degree=int(config.get("degree", 2)),
        max_cells=int(config.get("max_cells", 250000)),
        max_iters=int(config.get("max_iters", 1000)),
        max_model_size=int(config.get("max_model_size", 80)),
    )

    total_rho = cdp_rho(args.epsilon, args.delta)
    preprocesser = data_preporcesser_common(args)
    X_num_raw, X_cat_raw = _split_df(df, user_info_data)

    df_processed, _domain_list, _ = preprocesser.load_data(
        X_num_raw,
        X_cat_raw,
        total_rho,
        user_domain_data=user_domain_data,
        user_info_data=user_info_data,
    )

    # AIM expects a mapping {name: size}
    aim_domain = _make_aim_domain_mapping(df_processed, user_domain_data)

    bundle = {
        "args": args,
        "df_processed": df_processed,
        "domain_map": aim_domain,
        "preprocesser": preprocesser,
        "user_info": user_info_data,
        "user_domain": user_domain_data,
        "original_dtypes": df.dtypes.to_dict(),
    }
    return bundle


def run(
    bundle: Dict[str, Any],
    epsilon: Optional[float] = None,
    delta: Optional[float] = None,
    seed: Optional[int] = None,
    n_sample: Optional[int] = None,
) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)
    """Run AIM on CPU and return a decoded DataFrame.

    - Uses CDP rho computed from epsilon/delta if provided; otherwise from bundle args.
    - Returns synthesized data decoded back to original numeric/categorical spaces.
    """
    args: AdapterArgs = bundle["args"]
    if epsilon is not None:
        args.epsilon = float(epsilon)
    if delta is not None:
        args.delta = float(delta)

    total_rho = cdp_rho(args.epsilon, args.delta)

    df_processed = bundle["df_processed"]
    domain_map = bundle["domain_map"]
    preprocesser = bundle["preprocesser"]

    # Prepare args object for AIM
    args_obj = aim_add_defaults(args)

    result = aim_main(args_obj, df_processed, domain_map, total_rho)
    mech = result["aim_generator"]

    if n_sample is None:
        n_sample = df_processed.shape[0]

    # Synthesize (returns a Dataset with encoded ints)
    synth_dataset = mech.syn_data(n_sample, path=None, preprocesser=None)
    synth_encoded_df: pd.DataFrame = synth_dataset.df

    # Decode back to original numeric/categorical values
    x_num_rev, x_cat_rev = preprocesser.reverse_data(synth_encoded_df)
    info = bundle["user_info"]
    num_cols = info.get("num_columns", []) or []
    cat_cols = info.get("cat_columns", []) or []

    if x_num_rev is not None and x_cat_rev is not None:
        out = pd.DataFrame(
            np.concatenate((x_num_rev, x_cat_rev), axis=1),
            columns=num_cols + cat_cols,
        )
    elif x_num_rev is not None:
        out = pd.DataFrame(x_num_rev, columns=num_cols)
    elif x_cat_rev is not None:
        out = pd.DataFrame(x_cat_rev, columns=cat_cols)
    else:
        # Fallback: return the encoded dataframe with original column names
        out = synth_encoded_df.copy()
        out.columns = num_cols + cat_cols

    # Enforce dtypes to match original
    original_dtypes = bundle.get("original_dtypes")
    if original_dtypes is not None:
        for col, dtype in original_dtypes.items():
            if col in out.columns:
                try:
                    if pd.api.types.is_integer_dtype(dtype):
                        # Coerce to numeric, fill NaNs, then cast to integer
                        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0).astype(dtype)
                    else:
                        out[col] = out[col].astype(dtype)
                except (ValueError, TypeError):
                    # Fallback for columns that can't be cast
                    out[col] = pd.to_numeric(out[col], errors='coerce')
    else:
        # Fallback for older bundles without dtypes
        for col in num_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors='coerce')

    return out

# Register this adapter into the unified registry at import time
try:
    register_adapter("aim", prepare, run)
except Exception as _e:
    # Avoid import hard-fail in environments that import modules in arbitrary order
    pass

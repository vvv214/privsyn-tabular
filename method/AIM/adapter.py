import argparse
from typing import Any, Dict, Optional

import pandas as pd

from method.api.base import register_adapter
from method.AIM.aim import aim_main, add_default_params
from method.AIM.cdp2adp import cdp_rho
from method.AIM.mbi.Dataset import Dataset
from method.AIM.mbi.Domain import Domain


def prepare(
    df: pd.DataFrame,
    user_domain_data: Dict[str, Any],
    user_info_data: Dict[str, Any],
    config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    config = config or {}
    args = argparse.Namespace(**config)
    args = add_default_params(args)

    rho = cdp_rho(config.get("epsilon", 1.0), config.get("delta", 1e-5))

    mappings = {}
    df_encoded = df.copy()
    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object' or pd.api.types.is_categorical_dtype(df_encoded[col].dtype):
            codes, uniques = pd.factorize(df_encoded[col])
            df_encoded[col] = codes
            mappings[col] = uniques

    bundle = aim_main(args, df_encoded, user_domain_data, rho)
    bundle["columns"] = df.columns.tolist()
    bundle["dtypes"] = df.dtypes.to_dict()
    bundle["mappings"] = mappings
    return bundle


def run(
    bundle: Dict[str, Any],
    epsilon: float,
    delta: float,
    n_sample: int,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    generator = bundle["aim_generator"]
    if seed is not None:
        generator.prng.seed(seed)

    synth_dataset = generator.syn_data(num_synth_rows=n_sample)

    df = pd.DataFrame(synth_dataset.df.values, columns=synth_dataset.domain.attrs)

    # Decode categorical columns
    for col, uniques in bundle["mappings"].items():
        # Ensure indices are within bounds
        max_idx = len(uniques) - 1
        df[col] = df[col].apply(lambda i: uniques[i] if 0 <= i <= max_idx else None)

    # Ensure dtypes and column order match original
    for col in bundle["columns"]:
        if col in df and df[col].dtype != bundle["dtypes"][col]:
            try:
                df[col] = df[col].astype(bundle["dtypes"][col])
            except (TypeError, ValueError):
                pass  # Keep as is if casting fails
    return df[bundle["columns"]]


# This is a lightweight adapter that registers the AIM method.
# The dispatcher will discover it at import time via the registry.
register_adapter(
    method="AIM",
    prepare_fn=prepare,
    run_fn=run,
)

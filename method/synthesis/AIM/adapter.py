import argparse
from typing import Any, Dict, Optional

import pandas as pd

from method.api.base import register_adapter
from method.synthesis.AIM.aim import aim_main, add_default_params
from method.synthesis.AIM.cdp2adp import cdp_rho

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

    # All logic is now inside aim_main
    bundle = aim_main(args, df, user_domain_data, rho)
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

    # All decoding logic is now inside syn_data
    df = generator.syn_data(num_synth_rows=n_sample)
    return df

# This is a lightweight adapter that registers the AIM method.
# The dispatcher will discover it at import time via the registry.
register_adapter(
    method="AIM",
    prepare_fn=prepare,
    run_fn=run,
)
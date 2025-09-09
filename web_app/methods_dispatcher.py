import logging
from typing import Any, Dict

import pandas as pd


logger = logging.getLogger(__name__)


def synthesize(
    method: str,
    df: pd.DataFrame,
    user_domain_data: Dict[str, Any],
    user_info_data: Dict[str, Any],
    config: Dict[str, Any],
    n_sample: int,
) -> pd.DataFrame:
    """Dispatch to the selected synthesis method and return synthesized DataFrame.

    Currently supports: 'privsyn', 'aim'.
    """
    m = method.lower()
    if m == "privsyn":
        from method.privsyn import adapter as privsyn_adapter

        bundle = privsyn_adapter.prepare(
            df,
            user_domain_data=user_domain_data,
            user_info_data=user_info_data,
            config=config,
        )
        synth_df = privsyn_adapter.run(
            bundle,
            epsilon=config.get("epsilon"),
            delta=config.get("delta"),
            n_sample=n_sample,
        )
        return synth_df

    if m == "aim":
        from method.AIM import adapter as aim_adapter

        bundle = aim_adapter.prepare(
            df,
            user_domain_data=user_domain_data,
            user_info_data=user_info_data,
            config=config,
        )
        synth_df = aim_adapter.run(
            bundle,
            epsilon=config.get("epsilon"),
            delta=config.get("delta"),
            n_sample=n_sample,
        )
        return synth_df

    raise ValueError(f"Unsupported method: {method}")


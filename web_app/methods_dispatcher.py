import logging
from typing import Any, Dict

import pandas as pd

from method.api import PrivacySpec, RunConfig, SynthRegistry


logger = logging.getLogger(__name__)


# Ensure adapters are imported so they self-register with the registry
try:  # pragma: no cover - import-time side effect
    from method.privsyn import adapter as _privsyn_adapter  # noqa: F401
    from method.AIM import adapter as _aim_adapter  # noqa: F401
except Exception:  # pragma: no cover - non-fatal
    pass


def synthesize(
    method: str,
    df: pd.DataFrame,
    user_domain_data: Dict[str, Any],
    user_info_data: Dict[str, Any],
    config: Dict[str, Any],
    n_sample: int,
) -> pd.DataFrame:
    """Dispatch to the selected synthesis method and return synthesized DataFrame.

    Uses the unified SynthRegistry (adapters registered at import time).
    """
    # Prepare privacy + run config
    privacy = PrivacySpec(
        epsilon=float(config.get("epsilon", 1.0)),
        delta=float(config.get("delta", 1e-5)),
    )
    extra_cfg = {k: v for k, v in config.items() if k not in ("epsilon", "delta")}
    run_cfg = RunConfig(device=str(config.get("device", "cpu")), extra=extra_cfg)

    # Get synthesizer and fit/sample
    synthesizer = SynthRegistry.get(method)
    fitted = synthesizer.fit(
        df=df,
        domain=user_domain_data,
        info=user_info_data,
        privacy=privacy,
        config=run_cfg,
    )
    synth_df = fitted.sample(n=n_sample)
    return synth_df

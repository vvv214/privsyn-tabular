import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from method.api.base import FittedSynth, PrivacySpec, RunConfig, Synthesizer
from method.privsyn.privsyn import PrivSyn, add_default_params
from preprocess_common.load_data_common import data_preporcesser_common
from util.rho_cdp import cdp_rho

logger = logging.getLogger(__name__)


def _split_df(
    df: pd.DataFrame, info: Dict[str, Any]
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    num_cols = info.get("num_columns", []) or []
    cat_cols = info.get("cat_columns", []) or []

    X_num = df[num_cols].to_numpy(dtype=float) if len(num_cols) > 0 else None
    X_cat = df[cat_cols].astype(str).to_numpy() if len(cat_cols) > 0 else None
    return X_num, X_cat


class FittedPrivSyn(FittedSynth):
    def __init__(
        self,
        privsyn_generator: PrivSyn,
        preprocesser,
        user_info: Dict[str, Any],
        original_dtypes: Dict[str, Any],
        privacy: PrivacySpec,
        config: RunConfig,
    ):
        self._privsyn_generator = privsyn_generator
        self._preprocesser = preprocesser
        self._user_info = user_info
        self._original_dtypes = original_dtypes
        self._privacy = privacy
        self._config = config

    @property
    def info(self) -> Dict[str, Any]:
        info_dict = {
            "method": "privsyn",
            "epsilon": self._privacy.epsilon,
            "delta": self._privacy.delta,
        }
        if self._config and self._config.device:
            info_dict["device"] = self._config.device
        return info_dict

    def sample(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        if seed is not None:
            self._privsyn_generator.set_seed(seed)

        self._privsyn_generator.syn(n, self._preprocesser, parent_dir=None)
        out: pd.DataFrame = self._privsyn_generator.synthesized_df

        # Ensure column names restored to original
        info = self._user_info
        num_cols = info.get("num_columns", []) or []
        cat_cols = info.get("cat_columns", []) or []
        expected_cols = num_cols + cat_cols
        if len(expected_cols) == out.shape[1]:
            out.columns = expected_cols

        # Enforce dtypes to match original
        original_dtypes = self._original_dtypes
        if original_dtypes is not None:
            for col, dtype in original_dtypes.items():
                if col in out.columns:
                    try:
                        if pd.api.types.is_integer_dtype(dtype):
                            # Coerce to numeric, fill NaNs, then cast to integer
                            out[col] = (
                                pd.to_numeric(out[col], errors="coerce")
                                .fillna(0)
                                .astype(dtype)
                            )
                        else:
                            out[col] = out[col].astype(dtype)
                    except (ValueError, TypeError):
                        # Fallback for columns that can't be cast
                        out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            # Fallback for older bundles without dtypes
            for col in num_cols:
                if col in out.columns:
                    out[col] = pd.to_numeric(out[col], errors="coerce")
            for col in cat_cols:
                if col in out.columns:
                    out[col] = out[col].astype(str)
        return out

    def metrics(self, original_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        n_synth = original_df.shape[0] if original_df is not None else 10
        synth_df = self.sample(n=n_synth)
        return {"record_count": float(synth_df.shape[0])}


class PrivSynSynthesizer(Synthesizer):
    @classmethod
    def method_id(cls) -> str:
        return "privsyn"

    def fit(
        self,
        df: pd.DataFrame,
        domain: Dict[str, Any],
        info: Dict[str, Any],
        privacy: PrivacySpec,
        config: Optional[RunConfig] = None,
    ) -> FittedSynth:
        config = config or RunConfig()
        if config.random_state is not None:
            np.random.seed(config.random_state)

        extra_config = (config.extra or {}).copy()

        @dataclass
        class TempArgs:
            dataset: str
            method: str
            epsilon: float
            delta: float
            num_preprocess: str
            rare_threshold: float

        args = TempArgs(
            dataset="native_dataset",
            method="privsyn",
            epsilon=privacy.epsilon,
            delta=privacy.delta,
            num_preprocess=extra_config.pop("num_preprocess", "uniform_kbins"),
            rare_threshold=extra_config.pop("rare_threshold", 0.002),
        )

        total_rho = cdp_rho(args.epsilon, args.delta)
        preprocesser = data_preporcesser_common(args)
        X_num_raw, X_cat_raw = _split_df(df, info)

        df_processed, domain_sizes, _ = preprocesser.load_data(
            X_num_raw,
            X_cat_raw,
            total_rho,
            user_domain_data=domain,
            user_info_data=info,
        )

        args_obj = add_default_params(args)
        args_obj.extra = extra_config
        args_dict = vars(args_obj)

        privsyn_generator = PrivSyn(args_dict, df_processed, domain_sizes, total_rho)
        privsyn_generator.marginal_selection()

        return FittedPrivSyn(
            privsyn_generator=privsyn_generator,
            preprocesser=preprocesser,
            user_info=info,
            original_dtypes=df.dtypes.to_dict(),
            privacy=privacy,
            config=config,
        )

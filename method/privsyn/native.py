import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from method.api.base import FittedSynth, PrivacySpec, RunConfig, Synthesizer
from method.api.utils import enforce_dataframe_schema, split_df_by_type
from method.privsyn.privsyn import PrivSyn, add_default_params
from preprocess_common.load_data_common import data_preporcesser_common
from util.rho_cdp import cdp_rho

logger = logging.getLogger(__name__)


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
        info = self._user_info
        num_cols = info.get("num_columns", []) or []
        cat_cols = info.get("cat_columns", []) or []
        out = enforce_dataframe_schema(
            out,
            self._original_dtypes,
            num_cols,
            cat_cols,
        )
        return out

    def _internal_metrics(self, original_df: pd.DataFrame) -> Dict[str, float]:
        """
        Internal metrics for debugging privsyn.
        This is a white-box evaluation that compares the marginals of the
        original and synthetic data using the marginals selected by privsyn.
        """
        from method.privsyn.lib_dataset.dataset import Dataset
        from method.privsyn.lib_marginal.marg import Marginal
        from util.rho_cdp import cdp_rho
        from method.api.utils import split_df_by_type

        n_synth = original_df.shape[0]
        synth_df = self.sample(n=n_synth)

        total_rho = cdp_rho(self._privacy.epsilon, self._privacy.delta)
        X_num_raw, X_cat_raw = split_df_by_type(original_df, self._user_info)
        
        domain_dict = {}
        for attr, size in self._privsyn_generator.original_dataset.domain.config.items():
            domain_dict[attr] = {'size': size}

        df_processed, _, _ = self._preprocesser.load_data(
            X_num_raw,
            X_cat_raw,
            total_rho,
            user_domain_data=domain_dict,
            user_info_data=self._user_info,
        )
        original_dataset = Dataset(df_processed, self._privsyn_generator.original_dataset.domain)

        synth_X_num_raw, synth_X_cat_raw = split_df_by_type(synth_df, self._user_info)
        synth_df_processed, _, _ = self._preprocesser.load_data(
            synth_X_num_raw,
            synth_X_cat_raw,
            total_rho,
            user_domain_data=domain_dict,
            user_info_data=self._user_info,
        )
        synth_dataset = Dataset(synth_df_processed, self._privsyn_generator.original_dataset.domain)

        errors = {}
        for marg_spec in self._privsyn_generator.sel_marg_name:
            original_marg = Marginal(original_dataset.domain.project(marg_spec), original_dataset.domain)
            original_marg.count_records(original_dataset.df.values)
            original_marg_norm = original_marg.calculate_normalize_count()

            synth_marg = Marginal(synth_dataset.domain.project(marg_spec), synth_dataset.domain)
            synth_marg.count_records(synth_dataset.df.values)
            synth_marg_norm = synth_marg.calculate_normalize_count()
            
            error = np.abs(original_marg_norm - synth_marg_norm).sum() / 2.0
            errors[str(marg_spec)] = error

        return errors

    


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
        X_num_raw, X_cat_raw = split_df_by_type(df, info)

        df_processed, domain_sizes, _ = preprocesser.load_data(
            X_num_raw,
            X_cat_raw,
            total_rho,
            user_domain_data=domain,
            user_info_data=info,
        )

        args_obj = add_default_params(args)

        override_keys = (
            "consist_iterations",
            "non_negativity",
            "append",
            "sep_syn",
            "initialize_method",
            "update_method",
            "update_rate_method",
            "update_rate_initial",
            "update_iterations",
        )
        for key in override_keys:
            if key in extra_config:
                setattr(args_obj, key, extra_config.pop(key))

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

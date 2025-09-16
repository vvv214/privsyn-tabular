import itertools
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from method.AIM.aim import AIM, add_default_params
from method.AIM.mbi.Dataset import Dataset
from method.AIM.mbi.Domain import Domain
from method.api.base import FittedSynth, PrivacySpec, RunConfig, Synthesizer
from method.api.utils import enforce_dataframe_schema, split_df_by_type
from preprocess_common.load_data_common import data_preporcesser_common
from util.rho_cdp import cdp_rho

logger = logging.getLogger(__name__)


def _make_aim_domain_mapping(
    df_processed: pd.DataFrame, user_domain: Dict[str, Any]
) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for col in df_processed.columns:
        meta = user_domain.get(col, {})
        size = meta.get("size")
        if not isinstance(size, int):
            size = int(df_processed[col].max()) + 1
        mapping[col] = size
    return mapping


class FittedAIM(FittedSynth):
    def __init__(
        self,
        aim_generator: AIM,
        preprocesser,
        user_info: Dict[str, Any],
        original_dtypes: Dict[str, Any],
        privacy: PrivacySpec,
        config: RunConfig,
    ):
        self._aim_generator = aim_generator
        self._preprocesser = preprocesser
        self._user_info = user_info
        self._original_dtypes = original_dtypes
        self._privacy = privacy
        self._config = config

    @property
    def info(self) -> Dict[str, Any]:
        info_dict = {
            "method": "aim",
            "epsilon": self._privacy.epsilon,
            "delta": self._privacy.delta,
        }
        if self._config and self._config.device:
            info_dict["device"] = self._config.device
        return info_dict

    def sample(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        if seed is not None:
            if hasattr(self._aim_generator, "prng"):
                self._aim_generator.prng.seed(seed)
            else:
                np.random.seed(seed)

        synth_dataset = self._aim_generator.syn_data(n, path=None, preprocesser=None)
        synth_encoded_df: pd.DataFrame = synth_dataset.df

        x_num_rev, x_cat_rev = self._preprocesser.reverse_data(synth_encoded_df)
        info = self._user_info
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
            out = synth_encoded_df.copy()
            out.columns = num_cols + cat_cols

        out = enforce_dataframe_schema(out, self._original_dtypes, num_cols, cat_cols)
        return out

    def metrics(self, original_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        n_synth = original_df.shape[0] if original_df is not None else 10
        synth_df = self.sample(n=n_synth)
        return {"record_count": float(synth_df.shape[0])}


class AIMSynthesizer(Synthesizer):
    @classmethod
    def method_id(cls) -> str:
        return "aim"

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
            degree: int
            max_cells: int
            max_iters: int
            max_model_size: int

        args = TempArgs(
            dataset="native_dataset",
            method="aim",
            epsilon=privacy.epsilon,
            delta=privacy.delta,
            num_preprocess=extra_config.pop("num_preprocess", "uniform_kbins"),
            rare_threshold=extra_config.pop("rare_threshold", 0.002),
            degree=extra_config.pop("degree", 2),
            max_cells=extra_config.pop("max_cells", 250000),
            max_iters=extra_config.pop("max_iters", 1000),
            max_model_size=extra_config.pop("max_model_size", 80),
        )

        total_rho = cdp_rho(args.epsilon, args.delta)
        preprocesser = data_preporcesser_common(args)
        X_num_raw, X_cat_raw = split_df_by_type(df, info)

        df_processed, _domain_list, _ = preprocesser.load_data(
            X_num_raw,
            X_cat_raw,
            total_rho,
            user_domain_data=domain,
            user_info_data=info,
        )

        aim_domain_map = _make_aim_domain_mapping(df_processed, domain)

        args_obj = add_default_params(args)
        if "num_marginals" in extra_config:
            args_obj.num_marginals = extra_config.pop("num_marginals")
        args_obj.extra = extra_config

        aim_domain = Domain(aim_domain_map.keys(), aim_domain_map.values())
        data = Dataset(df_processed, aim_domain)

        workload = list(itertools.combinations(data.domain, args_obj.degree))
        workload = [
            cl for cl in workload if data.domain.size(cl) <= args_obj.max_cells
        ]
        if hasattr(args_obj, "num_marginals") and args_obj.num_marginals is not None:
            workload = [
                workload[i]
                for i in np.random.choice(
                    len(workload), args_obj.num_marginals, replace=False
                )
            ]
        workload = [(cl, 1.0) for cl in workload]

        mech = AIM(
            rho=total_rho,
            max_model_size=args_obj.max_model_size,
            max_iters=args_obj.max_iters,
        )
        mech.run(data, workload)

        return FittedAIM(
            aim_generator=mech,
            preprocesser=preprocesser,
            user_info=info,
            original_dtypes=df.dtypes.to_dict(),
            privacy=privacy,
            config=config,
        )

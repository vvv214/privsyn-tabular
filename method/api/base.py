from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Type

import pandas as pd


@dataclass
class PrivacySpec:
    epsilon: float
    delta: float = 1e-5
    # Optional alternative accounting (rho-CDP); keep fields flexible for future
    rho: Optional[float] = None


@dataclass
class RunConfig:
    device: str = "cpu"  # "cpu" | "cuda" | device string
    n_threads: Optional[int] = None
    random_state: Optional[int] = None
    timeout_s: Optional[int] = None
    # free-form config passthrough
    extra: Dict[str, Any] = None


class FittedSynth:
    """A fitted synthesizer ready to sample synthetic data."""

    def sample(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:  # pragma: no cover - interface
        raise NotImplementedError

    @property
    def info(self) -> Dict[str, Any]:  # pragma: no cover - interface
        return {}

    def metrics(self, original_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        # By default, adapters do not provide metrics.
        # This hook is for native synthesizers to override.
        return {
            "record_count": self.sample(n=original_df.shape[0] if original_df is not None else 10).shape[0],
        }


class Synthesizer:
    """Abstract synthesizer interface.

    Implementations fit on encoded/decoded inputs depending on the method,
    and return a FittedSynth that can sample decoded DataFrames.
    """

    @classmethod
    def method_id(cls) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def fit(
        self,
        df: pd.DataFrame,
        domain: Dict[str, Any],
        info: Dict[str, Any],
        privacy: PrivacySpec,
        config: Optional[RunConfig] = None,
    ) -> FittedSynth:  # pragma: no cover - interface
        raise NotImplementedError


class _AdapterFitted(FittedSynth):
    def __init__(
        self,
        *,
        run_fn: Callable[..., pd.DataFrame],
        bundle: Dict[str, Any],
        privacy: PrivacySpec,
        config: Optional[RunConfig],
        method: str,
    ) -> None:
        self._run_fn = run_fn
        self._bundle = bundle
        self._privacy = privacy
        self._config = config
        self._method = method
        self._extra_info: Dict[str, Any] = {}

    def sample(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        # If a seed is passed to sample, it overrides the fit-time seed.
        # Otherwise, we fall back to the fit-time seed.
        run_seed = seed
        if run_seed is None and self._config and self._config.random_state is not None:
            run_seed = self._config.random_state

        df = self._run_fn(
            self._bundle,
            epsilon=self._privacy.epsilon,
            delta=self._privacy.delta,
            seed=run_seed,
            n_sample=n,
        )
        return df

    @property
    def info(self) -> Dict[str, Any]:
        info_dict = {
            "method": self._method,
            "epsilon": self._privacy.epsilon,
            "delta": self._privacy.delta,
            **self._extra_info,
        }
        if self._config and self._config.device:
            info_dict["device"] = self._config.device
        return info_dict

    def metrics(self, original_df: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Metrics hook for adapter-based synthesizers.

        As a default, we provide a simple record count metric. Native synthesizers
        can and should provide more meaningful metrics.
        """
        n_synth = original_df.shape[0] if original_df is not None else 10
        synth_df = self.sample(n=n_synth)
        return {"record_count": float(synth_df.shape[0])}


class _AdapterSynth(Synthesizer):
    def __init__(
        self,
        *,
        method: str,
        prepare_fn: Callable[[pd.DataFrame, Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]], Dict[str, Any]],
        run_fn: Callable[..., pd.DataFrame],
    ) -> None:
        self._method = method
        self._prepare_fn = prepare_fn
        self._run_fn = run_fn

    @classmethod
    def method_id(cls) -> str:
        # Adapter instances carry method id via instance; registry uses instance
        return "adapter"

    def fit(
        self,
        df: pd.DataFrame,
        domain: Dict[str, Any],
        info: Dict[str, Any],
        privacy: PrivacySpec,
        config: Optional[RunConfig] = None,
    ) -> FittedSynth:
        # Bridge RunConfig -> adapter config dict
        raw_cfg: Dict[str, Any] = {}
        if config is not None:
            raw_cfg = {
                "device": config.device,
                "random_state": config.random_state,
                "n_threads": config.n_threads,
                **(config.extra or {}),
            }
        # Adapter prepare accepts df, domain/info dicts and config dict
        bundle = self._prepare_fn(df, domain, info, raw_cfg)
        # Store privacy in bundle's args if needed via run(...)
        return _AdapterFitted(
            run_fn=self._run_fn,
            bundle=bundle,
            privacy=privacy,
            config=config,
            method=self._method,
        )


class SynthRegistry:
    _registry: Dict[str, Callable[[], Synthesizer]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[[], Synthesizer]) -> None:
        key = name.lower()
        cls._registry[key] = factory

    @classmethod
    def get(cls, name: str) -> Synthesizer:
        key = name.lower()
        if key not in cls._registry:
            raise KeyError(f"Unknown synthesizer: {name}")
        return cls._registry[key]()

    @classmethod
    def list(cls) -> List[str]:
        return sorted(cls._registry.keys())


def register_adapter(
    method: str,
    prepare_fn: Callable[[pd.DataFrame, Dict[str, Any], Dict[str, Any], Optional[Dict[str, Any]]], Dict[str, Any]],
    run_fn: Callable[..., pd.DataFrame],
) -> None:
    """Register an existing adapter (prepare/run) under the unified interface.

    This wraps the functions in a Synthesizer implementation and stores it in SynthRegistry.
    """

    def _factory() -> Synthesizer:
        return _AdapterSynth(method=method, prepare_fn=prepare_fn, run_fn=run_fn)

    SynthRegistry.register(method, _factory)


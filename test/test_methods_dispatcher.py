import pandas as pd

from method.api.base import FittedSynth, PrivacySpec, RunConfig, SynthRegistry, Synthesizer
from web_app import methods_dispatcher


class _CaptureFitted(FittedSynth):
    def __init__(self):
        self.requested = None

    def sample(self, n: int, seed=None):  # pragma: no cover - simple passthrough
        return pd.DataFrame({"rows": list(range(n))})


class _CaptureSynth(Synthesizer):
    def __init__(self, bucket):
        self.bucket = bucket

    @classmethod
    def method_id(cls) -> str:  # pragma: no cover - unused
        return "capture"

    def fit(self, df, domain, info, privacy: PrivacySpec, config: RunConfig):
        self.bucket["privacy"] = privacy
        self.bucket["extra"] = dict(config.extra or {})
        self.bucket["device"] = config.device
        fitted = _CaptureFitted()
        fitted.requested = self.bucket
        return fitted


def test_dispatcher_preserves_privacy_in_extra():
    bucket = {}

    def _factory():
        return _CaptureSynth(bucket)

    SynthRegistry.register("dispatcher_dummy", _factory)
    try:
        df = pd.DataFrame({"x": [1, 2]})
        info = {"num_columns": [], "cat_columns": []}
        config = {"epsilon": 2.5, "delta": 1e-6, "consist_iterations": 9}

        methods_dispatcher.synthesize(
            method="dispatcher_dummy",
            df=df,
            user_domain_data={},
            user_info_data=info,
            config=config,
            n_sample=3,
        )
    finally:
        SynthRegistry._registry.pop("dispatcher_dummy", None)

    assert bucket["extra"]["epsilon"] == 2.5
    assert bucket["extra"]["delta"] == 1e-6
    assert bucket["extra"]["consist_iterations"] == 9


def test_dispatcher_passes_device_when_present():
    bucket = {}

    def _factory():
        return _CaptureSynth(bucket)

    SynthRegistry.register("dispatcher_device", _factory)
    try:
        df = pd.DataFrame({"x": [0]})
        info = {"num_columns": [], "cat_columns": []}
        config = {"epsilon": 1.0, "delta": 1e-5, "device": "cuda"}

        methods_dispatcher.synthesize(
            method="dispatcher_device",
            df=df,
            user_domain_data={},
            user_info_data=info,
            config=config,
            n_sample=1,
        )
    finally:
        SynthRegistry._registry.pop("dispatcher_device", None)

    assert bucket["device"] == "cuda"

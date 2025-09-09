import pandas as pd
import pytest

from method.api import PrivacySpec, RunConfig, SynthRegistry


def _toy_df():
    # Tiny mixed DataFrame
    return pd.DataFrame(
        {
            "age": [21, 35, 44, 28],
            "income": [50000, 82000, 120000, 61000],
            "gender": ["M", "F", "F", "M"],
        }
    )


def _toy_domain_info(df: pd.DataFrame):
    # Minimal domain/info; adapters tolerate missing sizes by inferring
    domain = {
        "age": {"type": "num", "size": 256},
        "income": {"type": "num", "size": 256},
        "gender": {"type": "cat", "size": 3, "categories": ["M", "F", "U"]},
    }
    info = {
        "num_columns": ["age", "income"],
        "cat_columns": ["gender"],
    }
    return domain, info


def _fit_and_sample(name: str, n: int = 4):
    df = _toy_df()
    domain, info = _toy_domain_info(df)
    synth = SynthRegistry.get(name)
    fitted = synth.fit(
        df=df,
        domain=domain,
        info=info,
        privacy=PrivacySpec(epsilon=0.5, delta=1e-5),
        config=RunConfig(device="cpu"),
    )
    out = fitted.sample(n)
    return df, out


def _assert_df_shape_and_cols(df_in: pd.DataFrame, df_out: pd.DataFrame):
    assert list(df_out.columns) == list(df_in.columns), "Column order must be preserved"
    assert df_out.shape[0] >= 1
    assert df_out.shape[1] == df_in.shape[1]
    # Dtypes must be preserved (or be a reasonable equivalent, like int64 for int32)
    for col, expected_dtype in df_in.dtypes.items():
        actual_dtype = df_out[col].dtype
        assert pd.api.types.is_dtype_equal(actual_dtype, expected_dtype), \
            f"Dtype for column '{col}' must be {expected_dtype}, but was {actual_dtype}"


@pytest.mark.slow
def test_registry_lists_methods():
    names = SynthRegistry.list()
    # Expect at least privsyn and aim registered by adapters
    assert "privsyn" in names
    assert "aim" in names


@pytest.mark.slow
def test_contract_privsyn_fit_sample():
    df_in, df_out = _fit_and_sample("privsyn", n=4)
    _assert_df_shape_and_cols(df_in, df_out)


@pytest.mark.slow
def test_contract_aim_fit_sample():
    df_in, df_out = _fit_and_sample("aim", n=4)
    _assert_df_shape_and_cols(df_in, df_out)


@pytest.mark.slow
@pytest.mark.parametrize("method", ["privsyn", "aim"])
def test_contract_run_config_wiring(method: str):
    df = _toy_df()
    domain, info = _toy_domain_info(df)
    synth = SynthRegistry.get(method)
    fitted = synth.fit(
        df=df,
        domain=domain,
        info=info,
        privacy=PrivacySpec(epsilon=0.5, delta=1e-5),
        config=RunConfig(device="test_device"),
    )
    assert fitted.info.get("device") == "test_device"


@pytest.mark.slow
@pytest.mark.parametrize("method", ["privsyn", "aim"])
def test_contract_metrics_hook(method: str):
    df = _toy_df()
    domain, info = _toy_domain_info(df)
    synth = SynthRegistry.get(method)
    fitted = synth.fit(
        df=df,
        domain=domain,
        info=info,
        privacy=PrivacySpec(epsilon=0.5, delta=1e-5),
        config=RunConfig(device="cpu"),
    )
    metrics = fitted.metrics(original_df=df)
    assert "record_count" in metrics
    assert isinstance(metrics["record_count"], float)


@pytest.mark.slow
@pytest.mark.parametrize("method", ["privsyn", "aim"])
def test_contract_deterministic_sample(method: str):
    df = _toy_df()
    domain, info = _toy_domain_info(df)
    synth = SynthRegistry.get(method)
    fitted = synth.fit(
        df=df,
        domain=domain,
        info=info,
        privacy=PrivacySpec(epsilon=0.5, delta=1e-5),
        config=RunConfig(device="cpu", random_state=123),  # config seed for fit
    )
    # Two samples with same seed must be identical
    out1 = fitted.sample(n=5, seed=42)
    out2 = fitted.sample(n=5, seed=42)
    pd.testing.assert_frame_equal(out1, out2)

    # A third sample with a different seed may or may not differ,
    # as the synthesizer's fit() method may be fully deterministic
    # when configured with a seed.
    out3 = fitted.sample(n=5, seed=99)
    assert out3.shape == out1.shape

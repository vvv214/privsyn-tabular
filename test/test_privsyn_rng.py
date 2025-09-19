import numpy as np
import pandas as pd

from method.api.base import PrivacySpec, RunConfig
from method.synthesis.privsyn.native import PrivSynSynthesizer


def _states_equal(left, right):
    return all(
        np.array_equal(l, r) if isinstance(l, np.ndarray) else l == r
        for l, r in zip(left, right)
    )


def _build_domain_and_info():
    domain = {
        'num': {
            'type': 'numerical',
            'size': 4,
            'bounds': {'min': 0.0, 'max': 3.0},
            'binning': {'method': 'uniform', 'bin_count': 4},
        },
        'cat': {
            'type': 'categorical',
            'size': 3,
            'categories': ['A', 'B', 'C'],
            'selected_categories': ['A', 'B', 'C'],
            'custom_categories': [],
            'categories_from_data': ['A', 'B', 'C'],
            'excluded_categories': [],
            'special_token': '__OTHER__',
        },
    }
    info = {
        'num_columns': ['num'],
        'cat_columns': ['cat'],
        'n_num_features': 1,
        'n_cat_features': 1,
    }
    return domain, info


def test_privsyn_fit_and_sample_preserve_global_rng_state():
    df = pd.DataFrame({
        'num': [0.0, 1.0, 2.0, 3.0],
        'cat': ['A', 'B', 'A', 'C'],
    })
    domain, info = _build_domain_and_info()
    synthesizer = PrivSynSynthesizer()
    privacy = PrivacySpec(epsilon=1.0, delta=1e-5)
    run_cfg = RunConfig(random_state=1337)

    state_before_fit = np.random.get_state()
    fitted = synthesizer.fit(df=df, domain=domain, info=info, privacy=privacy, config=run_cfg)
    state_after_fit = np.random.get_state()
    assert _states_equal(state_before_fit, state_after_fit)

    state_before_sample = np.random.get_state()
    sample_a = fitted.sample(n=4, seed=2024)
    state_after_sample = np.random.get_state()
    assert _states_equal(state_before_sample, state_after_sample)

    sample_b = fitted.sample(n=4, seed=2024)
    pd.testing.assert_frame_equal(sample_a, sample_b)

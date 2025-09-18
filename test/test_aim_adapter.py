import os, sys
import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import pytest
from method.synthesis.AIM.cdp2adp import cdp_rho


def _build_simple_df(n=60):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'num_a': rng.integers(0, 10, size=n),
        'num_b': rng.integers(0, 5, size=n),
        'cat_c': rng.choice(['x', 'y', 'z'], size=n),
    })
    domain_data = {
        'num_a': {'type': 'numerical', 'size': df['num_a'].nunique()},
        'num_b': {'type': 'numerical', 'size': df['num_b'].nunique()},
        'cat_c': {'type': 'categorical', 'size': df['cat_c'].nunique()},
    }
    info_data = {
        'num_columns': ['num_a', 'num_b'],
        'cat_columns': ['cat_c'],
        'n_num_features': 2,
        'n_cat_features': 1,
        'n_classes': 0,
        'train_size': n,
        'test_size': 0,
        'val_size': 0,
    }
    return df, domain_data, info_data


@pytest.mark.slow
def test_aim_adapter_basic_synthesize():
    from method.synthesis.AIM import adapter as aim_adapter

    df, domain_data, info_data = _build_simple_df(80)

    bundle = aim_adapter.prepare(
        df,
        user_domain_data=domain_data,
        user_info_data=info_data,
        config={
            'dataset': 'aim_basic',
            'epsilon': 0.5,
            'delta': 1e-5,
            'num_preprocess': 'uniform_kbins',
            'rare_threshold': 0.002,
            # keep AIM light for tests
            'degree': 2,
            'max_cells': 5000,
            'max_iters': 50,
            'max_model_size': 50,
        },
    )

    generator = bundle["aim_generator"]
    assert generator.max_model_size == 50
    assert generator.max_iters == 50
    assert generator.rho == pytest.approx(cdp_rho(0.5, 1e-5))

    synth_df = aim_adapter.run(bundle, n_sample=20, epsilon=0.5, delta=1e-5, seed=123)

    assert isinstance(synth_df, pd.DataFrame)
    assert list(synth_df.columns) == ['num_a', 'num_b', 'cat_c']
    assert synth_df.shape[0] == 20
    # Value ranges should be plausible
    assert synth_df['num_a'].min() >= df['num_a'].min()
    assert synth_df['num_b'].max() <= df['num_b'].max()
    assert set(synth_df['cat_c'].unique()) <= set(df['cat_c'].unique())

    repeat_df = aim_adapter.run(bundle, n_sample=20, epsilon=0.5, delta=1e-5, seed=123)
    pd.testing.assert_frame_equal(synth_df, repeat_df)

import os, sys
import numpy as np
import pandas as pd
import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


def test_aim_adapter_numeric_dtypes():
    from method.AIM import adapter as aim_adapter
    from method.AIM.cdp2adp import cdp_rho

    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'x': rng.integers(0, 7, size=60),
        'y': rng.integers(0, 3, size=60),
        'z': rng.choice(['u','v'], size=60),
    })
    domain = {
        'x': {'type': 'numerical', 'size': df['x'].nunique()},
        'y': {'type': 'numerical', 'size': df['y'].nunique()},
        'z': {'type': 'categorical', 'size': df['z'].nunique()},
    }
    info = {
        'num_columns': ['x','y'],
        'cat_columns': ['z'],
        'n_num_features': 2,
        'n_cat_features': 1,
        'n_classes': 0,
        'train_size': 60,
        'test_size': 0,
        'val_size': 0,
    }

    bundle = aim_adapter.prepare(
        df,
        user_domain_data=domain,
        user_info_data=info,
        config={'dataset':'aim_dtype','epsilon':0.5,'delta':1e-5,
                'max_iters': 20, 'max_model_size': 50, 'max_cells': 5000}
    )
    generator = bundle["aim_generator"]
    assert generator.max_iters == 20
    assert generator.max_model_size == 50
    assert generator.rho == pytest.approx(cdp_rho(0.5, 1e-5))
    out = aim_adapter.run(bundle, n_sample=10, epsilon=0.5, delta=1e-5)
    # numeric columns must be numeric dtype
    assert pd.api.types.is_numeric_dtype(out['x'])
    assert pd.api.types.is_numeric_dtype(out['y'])
    # categorical should be present
    assert out['z'].dtype == object or pd.api.types.is_string_dtype(out['z'])

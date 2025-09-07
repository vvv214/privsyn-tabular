import os, sys
import numpy as np
import pandas as pd

from preprocess_common.load_data_common import data_preporcesser_common


class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_reverse_data_numeric_only():
    args = Args(method='privsyn', num_preprocess='uniform_kbins', epsilon=0.1, delta=1e-5, rare_threshold=0.002, dataset='unit')
    dp = data_preporcesser_common(args)
    X_num = np.random.rand(50, 2)
    X_cat = None
    domain_data = {
        'num_attr_1': {'type': 'numerical', 'size': 10, 'min': 0.0, 'max': 1.0},
        'num_attr_2': {'type': 'numerical', 'size': 10, 'min': 0.0, 'max': 1.0},
    }
    info_data = {'num_columns': ['num_attr_1','num_attr_2'], 'cat_columns': []}
    df, domain_processed, _ = dp.load_data(X_num_raw=X_num, X_cat_raw=X_cat, rho=0.1, user_domain_data=domain_data, user_info_data=info_data)
    x_num_rev, x_cat_rev = dp.reverse_data(df.values)
    assert x_num_rev is not None
    assert x_cat_rev is None


def test_reverse_data_categorical_only():
    args = Args(method='privsyn', num_preprocess='uniform_kbins', epsilon=0.1, delta=1e-5, rare_threshold=0.002, dataset='unit')
    dp = data_preporcesser_common(args)
    X_num = None
    X_cat = np.random.choice(['A','B','C'], size=(50, 2))
    domain_data = {
        'cat_attr_1': {'type': 'categorical', 'size': 3},
        'cat_attr_2': {'type': 'categorical', 'size': 3},
    }
    info_data = {'num_columns': [], 'cat_columns': ['cat_attr_1','cat_attr_2']}
    df, domain_processed, _ = dp.load_data(X_num_raw=X_num, X_cat_raw=X_cat, rho=0.1, user_domain_data=domain_data, user_info_data=info_data)
    x_num_rev, x_cat_rev = dp.reverse_data(df.values)
    assert x_num_rev is None
    assert x_cat_rev is not None


import numpy as np
import pandas as pd

from method.reconstruct_algo.reconstruct import reconstruct_to_original


class DummyPreprocessor:
    def __init__(self, num_columns=None, cat_columns=None):
        self._num_cols = num_columns or []
        self._cat_cols = cat_columns or []

    def reverse_data(self, df_encoded):
        num = df_encoded[self._num_cols].to_numpy() if self._num_cols else None
        cat = df_encoded[self._cat_cols].to_numpy() if self._cat_cols else None
        return num, cat


def test_reconstruct_with_numeric_and_categorical():
    df = pd.DataFrame({'n0': [0, 1], 'c0': [0, 1]})
    preprocesser = DummyPreprocessor(num_columns=['n0'], cat_columns=['c0'])
    info = {'num_columns': ['age'], 'cat_columns': ['city']}

    reconstructed = reconstruct_to_original(df, preprocesser, info)

    assert list(reconstructed.columns) == ['age', 'city']
    assert reconstructed.shape == (2, 2)


def test_reconstruct_numeric_only():
    df = pd.DataFrame({'n0': [0, 1], 'n1': [2, 3]})
    preprocesser = DummyPreprocessor(num_columns=['n0', 'n1'])
    info = {'num_columns': ['x', 'y'], 'cat_columns': []}

    reconstructed = reconstruct_to_original(df, preprocesser, info)
    assert list(reconstructed.columns) == ['x', 'y']
    assert np.array_equal(reconstructed.values, df[['n0', 'n1']].to_numpy())


def test_reconstruct_passes_through_when_no_reverse():
    df = pd.DataFrame({'raw0': [5, 6]})
    preprocesser = DummyPreprocessor()
    info = {'num_columns': ['raw0'], 'cat_columns': []}

    reconstructed = reconstruct_to_original(df, preprocesser, info)
    assert list(reconstructed.columns) == ['raw0']
    assert reconstructed.equals(df)

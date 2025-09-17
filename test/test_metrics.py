import pandas as pd
import pytest

from method.metrics import calculate_marginal_error


def test_calculate_marginal_error_zero_diff():
    df = pd.DataFrame(
        {
            'a': ['x', 'y', 'x', 'z'],
            'b': ['m', 'm', 'n', 'n'],
        }
    )
    errors = calculate_marginal_error(df, df.copy(), max_way=2)
    assert errors
    for value in errors.values():
        assert value == pytest.approx(0.0)


def test_calculate_marginal_error_detects_difference():
    original = pd.DataFrame(
        {
            'a': ['x', 'y', 'x', 'z'],
            'b': ['m', 'm', 'n', 'n'],
        }
    )
    synth = original.copy()
    synth.loc[0, 'a'] = 'y'

    errors = calculate_marginal_error(original, synth, max_way=2)
    # At least one marginal should report a non-zero deviation.
    assert any(val > 0 for val in errors.values())

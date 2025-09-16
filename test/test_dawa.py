
import numpy as np
import pytest

from preprocess_common import dawa


@pytest.fixture(autouse=True)
def deterministic_primitives(monkeypatch):
    monkeypatch.setattr(dawa.random, "random", lambda: 0.25)
    monkeypatch.setattr(np.random, "randint", lambda high: 1)
    monkeypatch.setattr(np.random, "laplace", lambda loc, scale, size: np.zeros(size))
    monkeypatch.setattr(np.random, "uniform", lambda low, high: (np.asarray(low) + np.asarray(high)) / 2.0)


def test_L1partition_histogram_modes():
    data = [1.0, 2.0, 3.0, 4.0]
    hist = dawa.L1partition_fn(data, epsilon=1.0, ratio=0.5, gethist=True)
    assert hist[0][0] == 0
    approx_hist = dawa.L1partition_approx_fn(data, epsilon=1.0, ratio=0.5, gethist=True)
    assert approx_hist[0][0] == 0


def test_L1partition_denoise_vector():
    data = [1.0, 2.0, 3.0, 4.0]
    denoised = dawa.L1partition_fn(data, epsilon=1.0, ratio=0.5, gethist=False)
    assert denoised.shape == (len(data),)
    approx = dawa.L1partition_approx_fn(data, epsilon=1.0, ratio=0.5, gethist=False)
    assert approx.shape == (len(data),)


def test_interval_transform_round_trip():
    interval = np.array([[0.0, 1.0], [1.0, 2.0]])
    values = np.array([0.2, 1.5])
    encoded = dawa.interval_transform(values, interval)
    assert encoded.tolist() == [0, 1]

    decoded = dawa.interval_inverse_transform(encoded.astype(int), interval)
    assert decoded.shape == values.shape
    assert np.all(decoded >= interval[encoded.astype(int), 0])
    assert np.all(decoded <= interval[encoded.astype(int), 1])


def test_registry_run_methods():
    data = [0.5, 0.75, 1.0]
    hist1 = dawa.l1_partition.Run(data, 1.0, 0.5)
    hist2 = dawa.l1_partition_approx.Run(data, 1.0, 0.5)
    assert hist1 and hist2

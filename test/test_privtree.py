
import numpy as np
import pytest

from preprocess_common import privtree


@pytest.fixture(autouse=True)
def deterministic_random(monkeypatch):
    monkeypatch.setattr(np.random, "laplace", lambda loc=0.0, scale=1.0, size=None: 0.0 if size is None else np.zeros(size))
    monkeypatch.setattr(np.random, "uniform", lambda low, high: (np.array(low) + np.array(high)) / 2.0)


def test_privtree_basic_split_and_transform():
    tree = privtree.privtree(rho=0.5, theta=0, domain_margin=0.1)
    data = np.array([0.0, 0.2, 0.4, 0.8])
    transformed = tree.fit_transform(data)

    assert transformed.dtype == int
    assert transformed.min() >= 0
    assert len(tree.split) > 0

    reconstructed = tree.inverse_transform(transformed)
    assert reconstructed.shape == data.shape
    assert np.all(reconstructed >= data.min() - 0.1)
    assert np.all(reconstructed <= data.max() + 0.1)


def test_privtree_multicolumn_transform():
    tree = privtree.privtree(rho=1.0, theta=0)
    data = np.array([[0.1, 1.0], [0.4, 1.5], [0.9, 2.0]])
    tree.fit(data)
    transformed = tree.transform(data)
    assert transformed.shape == data.shape
    assert set(np.unique(transformed[:, 0])) <= {0, 1, 2}


def test_helper_functions():
    q1, q2 = privtree.get_domain_subdomains([0.0, 1.0])
    assert q1[0] == 0.0 and q2[1] == 1.0

    xs = np.array([0.1, 0.4, 0.9])
    assert privtree.count_in_domain(xs, [0.0, 0.5]) == 2

    assert privtree.is_in_domain(0.3, 0.0, 0.5)
    assert not privtree.is_in_domain(0.9, 0.0, 0.5)


def test_calculate_param_positive():
    tree = privtree.privtree(rho=0.7)
    lam, delta = tree.calculate_param(0.7, 2)
    assert lam > 0
    assert delta > 0


import logging
import numpy as np
import pandas as pd
import pytest

from method.privsyn.lib_dataset.dataset import Dataset
from method.privsyn.lib_dataset.domain import Domain
from method.privsyn.lib_marginal.marg_determine import marginal_selection, marginal_combine
from method.privsyn.lib_marginal.marg_select_helper import calculate_indif, handle_isolated_attrs


# Mock logger to prevent console output during tests
class MockLogger:
    def info(self, *args, **kwargs): pass
    def debug(self, *args, **kwargs): pass


@pytest.fixture
def simple_dataset_independent():
    df = pd.DataFrame({
        'A': [0, 0, 1, 1],
        'B': [0, 1, 0, 1]
    })
    domain = Domain(['A', 'B'], [2, 2])
    return Dataset(df, domain)


@pytest.fixture
def simple_dataset_dependent():
    df = pd.DataFrame({
        'A': [0, 1, 2, 3],
        'B': [0, 1, 2, 3]
    })
    domain = Domain(['A', 'B'], [4, 4])
    return Dataset(df, domain)


@pytest.fixture
def simple_dataset_mixed():
    df = pd.DataFrame({
        'A': [0, 0, 1, 1],
        'B': [0, 1, 0, 1],
        'C': [0, 0, 0, 0]
    })
    domain = Domain(['A', 'B', 'C'], [2, 2, 1])
    return Dataset(df, domain)


@pytest.fixture
def simple_dataset_clique():
    df = pd.DataFrame({
        'A': [0, 1, 0, 1],
        'B': [0, 0, 1, 1],
        'C': [0, 0, 0, 0]
    })
    domain = Domain(['A', 'B', 'C'], [2, 2, 1])
    return Dataset(df, domain)


def test_calculate_indif_independent(simple_dataset_independent):
    logger = MockLogger()
    indif_df = calculate_indif(logger, simple_dataset_independent, "test_independent", rho=0.0)

    assert isinstance(indif_df, pd.DataFrame)
    assert len(indif_df) == 1  # Only one pair (A, B)
    assert indif_df.loc[0, "first_attr"] == 'A'
    assert indif_df.loc[0, "second_attr"] == 'B'
    assert indif_df.loc[0, "num_cells"] == 4
    assert indif_df.loc[0, "error"] == pytest.approx(0.0) # Perfectly independent


def test_calculate_indif_dependent(simple_dataset_dependent):
    logger = MockLogger()
    indif_df = calculate_indif(logger, simple_dataset_dependent, "test_dependent", rho=0.0)

    assert isinstance(indif_df, pd.DataFrame)
    assert len(indif_df) == 1  # Only one pair (A, B)
    assert indif_df.loc[0, "first_attr"] == 'A'
    assert indif_df.loc[0, "second_attr"] == 'B'
    assert indif_df.loc[0, "num_cells"] == 16
    assert indif_df.loc[0, "error"] == pytest.approx(6.0) # Perfectly dependent (L1 distance between uniform and diagonal)


def test_calculate_indif_with_noise(simple_dataset_independent):
    logger = MockLogger()
    indif_df = calculate_indif(logger, simple_dataset_independent, "test_noise", rho=1.0)

    assert isinstance(indif_df, pd.DataFrame)
    assert len(indif_df) == 1
    assert indif_df.loc[0, "error"] != pytest.approx(0.0) # Noise added


def test_handle_isolated_attrs_isolate(simple_dataset_mixed):
    logger = MockLogger()
    selected_attrs = set(['A', 'B'])
    indif_df = calculate_indif(logger, simple_dataset_mixed, "test_isolate", rho=0.0)
    marginals = [('A', 'B')]

    updated_marginals = handle_isolated_attrs(simple_dataset_mixed.domain, selected_attrs, indif_df, marginals, method="isolate")

    assert ('C',) in updated_marginals # C should be isolated
    assert len(updated_marginals) == 2 # (A,B) and (C,)


def test_handle_isolated_attrs_connect(simple_dataset_mixed):
    logger = MockLogger()
    selected_attrs = set(['A'])
    indif_df = calculate_indif(logger, simple_dataset_mixed, "test_connect", rho=0.0)
    marginals = [('A', 'B')]

    updated_marginals = handle_isolated_attrs(simple_dataset_mixed.domain, selected_attrs, indif_df, marginals, method="connect")

    # C should be connected to A or B, depending on which has lower error
    # In this case, A-C error is 0, B-C error is 0, A-B error is 0
    # So C should be connected to A or B
    assert ('C', 'A') in updated_marginals or ('A', 'C') in updated_marginals or \
           ('C', 'B') in updated_marginals or ('B', 'C') in updated_marginals
    assert len(updated_marginals) == 2 # (A,B) and (C,A) or (C,B)

def test_marginal_combine_simple_clique(simple_dataset_clique):
    logger = MockLogger()
    select_args = {'threshold': 1000} # A large threshold so cliques are not filtered
    marginals = [('A', 'B'), ('B', 'C'), ('A', 'C')]

    combined_marginals = marginal_combine(simple_dataset_clique, select_args, marginals)

    assert len(combined_marginals) == 1
    assert ('A', 'B', 'C') in combined_marginals or ('A', 'C', 'B') in combined_marginals or \
           ('B', 'A', 'C') in combined_marginals or ('B', 'C', 'A') in combined_marginals or \
           ('C', 'A', 'B') in combined_marginals or ('C', 'B', 'A') in combined_marginals
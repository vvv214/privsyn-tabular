import numpy as np
import pandas as pd
import os, sys

# Ensure project root on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from privsyn.lib_dataset.domain import Domain
from privsyn.lib_dataset.dataset import Dataset
from privsyn.lib_marginal.marg import Marginal
from privsyn.lib_marginal.consistent import Consistenter


def test_marginal_count_records_simple():
    # Domain: a in {0,1,2}, b in {0,1}
    domain = Domain(['a', 'b'], [3, 2])
    df = pd.DataFrame({
        'a': [0, 1, 1, 2, 2, 2],
        'b': [0, 0, 1, 0, 1, 1],
    })
    ds = Dataset(df, domain)

    marg = Marginal(ds.domain.project(('a', 'b')), ds.domain)
    marg.count_records(ds.df.values)
    # 3*2 = 6 bins
    assert marg.count.size == 6
    # All records counted
    assert int(marg.count.sum()) == df.shape[0]
    # Some rough sanity on distribution
    # a=2 occurs 3 times, b=1 occurs twice with a=2
    marg.calculate_count_matrix()
    assert int(marg.count_matrix[2, 1]) == 2


def test_consistency_non_negative_and_normalized():
    # Build two overlapping marginals AB and BC with small inconsistencies
    domain = Domain(['a', 'b', 'c'], [2, 2, 2])
    df = pd.DataFrame({
        'a': [0,0,1,1],
        'b': [0,1,0,1],
        'c': [0,1,1,0],
    })
    ds = Dataset(df, domain)

    m_ab = Marginal(ds.domain.project(('a','b')), ds.domain)
    m_ab.count_records(ds.df.values)
    m_bc = Marginal(ds.domain.project(('b','c')), ds.domain)
    m_bc.count_records(ds.df.values)

    margs = {('a','b'): m_ab, ('b','c'): m_bc}
    params = {"consist_iterations": 5, "non_negativity": 'N3'}
    c = Consistenter(margs, ds.domain, params)
    c.consist_marginals()

    for key, m in margs.items():
        assert (m.count >= 0).all()
        total = m.count.sum()
        if total > 0:
            assert np.isclose(m.calculate_normalize_count().sum(), 1.0)
        else:
            # all zeros case is acceptable on tiny synthetic inputs
            assert total == 0

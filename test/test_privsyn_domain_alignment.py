import os, sys
import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from method.synthesis.privsyn.privsyn import PrivSyn


def test_two_way_marginal_selection_domain_alignment():
    # df columns order intentionally shuffled vs domain dict iteration order
    df = pd.DataFrame({
        'b': np.random.randint(0, 3, size=50),
        'a': np.random.randint(0, 2, size=50),
        'c': np.random.randint(0, 4, size=50),
    })
    domain = {'a': 2, 'b': 3, 'c': 4}

    pairs = PrivSyn.two_way_marginal_selection(df, domain, rho_indif=0.0, rho_measure=0.1)

    # returns list of tuples of existing columns
    assert isinstance(pairs, list)
    assert all(isinstance(t, tuple) for t in pairs)
    for t in pairs:
        for col in t:
            assert col in df.columns


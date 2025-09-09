import numpy as np
import pandas as pd
import os, sys

# Ensure project root on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from method.privsyn.privsyn import PrivSyn


def test_two_way_marginal_selection_returns_pairs():
    # Small synthetic dataframe with 3 columns
    df = pd.DataFrame({
        'x': np.random.randint(0, 3, size=100),
        'y': np.random.randint(0, 2, size=100),
        'z': np.random.randint(0, 4, size=100),
    })
    # PrivSyn.two_way_marginal_selection expects a dict of sizes
    domain = {'x': 3, 'y': 2, 'z': 4}
    # Use small rho to keep selection quick
    pairs = PrivSyn.two_way_marginal_selection(df, domain, rho_indif=0.0, rho_measure=0.1)
    assert isinstance(pairs, list)
    # At least one pair or singleton
    assert len(pairs) > 0
    # Elements are tuples of attr names
    assert all(isinstance(t, tuple) for t in pairs)

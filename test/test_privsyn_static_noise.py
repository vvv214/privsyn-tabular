import numpy as np
import pandas as pd
import os, sys

# Ensure project root on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from privsyn.privsyn import PrivSyn


def test_two_way_marginal_selection_with_noise():
    # Small synthetic dataframe
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        'x': rng.integers(0, 3, size=200),
        'y': rng.integers(0, 2, size=200),
        'z': rng.integers(0, 4, size=200),
    })
    domain = {'x': 3, 'y': 2, 'z': 4}

    # Non-zero rho triggers noise branch in indif
    pairs = PrivSyn.two_way_marginal_selection(df, domain, rho_indif=0.5, rho_measure=0.1)
    assert isinstance(pairs, list)
    assert len(pairs) > 0
    assert all(isinstance(t, tuple) for t in pairs)


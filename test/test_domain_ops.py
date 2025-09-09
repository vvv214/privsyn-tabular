import os, sys

# Ensure project root on sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from method.privsyn.lib_dataset.domain import Domain


def test_domain_project_marginalize_merge():
    d = Domain(['a','b','c'], [2,3,4])
    # Project keeps order
    dp = d.project(['b','c'])
    assert dp.attrs == ('b','c')
    assert dp.shape == (3,4)

    # Marginalize removes given attrs
    dm = d.marginalize(['b'])
    assert dm.attrs == ('a','c')
    assert dm.shape == (2,4)

    # Merge domains
    d1 = Domain(['a','b'], [2,3])
    d2 = Domain(['b','c'], [3,4])
    dm = d1.merge(d2)
    assert dm.attrs == ('a','b','c')
    assert dm.shape == (2,3,4)

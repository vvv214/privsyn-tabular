from __future__ import annotations

from contextlib import contextmanager
import numpy as np


@contextmanager
def temporary_numpy_seed(seed: int | None):
    """Temporarily set NumPy's global RNG seed and restore it afterwards."""
    if seed is None:
        yield
        return

    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

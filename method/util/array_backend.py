"""
Lightweight array backend selector for optional GPU acceleration.

Usage:
- Set environment variable PRIVSYN_USE_CUPY=1 to enable CuPy if installed.
- Functions fall back to NumPy if CuPy is unavailable or not requested.
"""

import os
import numpy as _np

_USE_GPU = os.getenv("PRIVSYN_USE_CUPY") == "1"
_xp = _np
_cupy_available = False

if _USE_GPU:
    try:
        import cupy as _cp  # type: ignore
        _xp = _cp
        _cupy_available = True
    except Exception:
        _xp = _np
        _cupy_available = False


def xp():
    return _xp


def enabled() -> bool:
    return _USE_GPU and _cupy_available


def asarray(a):
    return _xp.asarray(a)


def to_numpy(a):
    if _xp is _np:
        return a
    try:
        return _xp.asnumpy(a)  # cupy
    except Exception:
        return a


def dot(a, b):
    return _xp.dot(a, b)


def clip(a, a_min, a_max):
    return _xp.clip(a, a_min, a_max)


def bincount(a, minlength=0, weights=None):
    if weights is not None:
        return _xp.bincount(a, weights=weights, minlength=minlength)
    return _xp.bincount(a, minlength=minlength)


"""Centralised helpers for sampling DP noise."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

ArrayLike = Union[int, Sequence[int], tuple]
RNGLike = Optional[np.random.Generator]


def _resolve_rng(rng: Optional[Union[np.random.Generator, np.random.RandomState]]):
    if rng is None:
        return np.random
    return rng


def gaussian_noise(
    scale: float,
    size: Optional[ArrayLike] = None,
    rng: Optional[Union[np.random.Generator, np.random.RandomState]] = None,
):
    """Draw Gaussian noise with the given scale."""
    resolved = _resolve_rng(rng)
    return resolved.normal(loc=0.0, scale=scale, size=size)


def laplace_noise(
    scale: float,
    size: Optional[ArrayLike] = None,
    rng: Optional[Union[np.random.Generator, np.random.RandomState]] = None,
):
    """Draw Laplace noise with the given scale."""
    resolved = _resolve_rng(rng)
    return resolved.laplace(loc=0.0, scale=scale, size=size)

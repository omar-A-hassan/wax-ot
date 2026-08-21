"""Evaluation metrics used in the paper.

- :func:`srg` implements the Symmetric Relevance Gain of equation (5), the
  faithfulness criterion that compares the effect of retaining versus
  excluding the ``pi`` most relevant features.
- :func:`cosine_similarity` scores attribution vectors against the
  ground-truth transport relevance of equation (6).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .coupling import Coupling

__all__ = ["srg", "cosine_similarity"]


def srg(
    X: np.ndarray,
    Y: np.ndarray,
    p: float,
    q: float,
    R_i: np.ndarray,
    coupling_factory: Callable[..., Coupling],
    fraction: float = 1.0,
) -> float:
    """Symmetric Relevance Gain (paper equation (5)).

    ``R_i`` must be a per-feature attribution (higher = more relevant).
    For each retention level ``pi`` from 0 to ``d`` the Wasserstein distance
    of the top-``pi`` retained features is compared with the distance of the
    excluded features; the mean signed gain is returned.

    ``coupling_factory(X, Y, p, q)`` is used to re-solve the transport problem
    on each feature subset.
    """
    if fraction < 1.0:
        n_keep = int(round(len(R_i) * fraction))
        order = np.argsort(-np.asarray(R_i))[:n_keep]
        Xr, Yr = X[:, order], Y[:, order]
        Rr = np.asarray(R_i)[order]
    else:
        Xr, Yr, Rr = X, Y, np.asarray(R_i)
    order = np.argsort(-Rr)
    n = Xr.shape[1]
    gains = np.empty(n + 1, dtype=float)
    for pi in range(n + 1):
        sel = order[:pi]
        ex = order[pi:]
        W_plus = _subset_w(Xr, Yr, sel, p, q, coupling_factory)
        W_minus = _subset_w(Xr, Yr, ex, p, q, coupling_factory)
        gains[pi] = W_plus - W_minus
    return float(gains.mean())


def _subset_w(
    X: np.ndarray,
    Y: np.ndarray,
    idx: np.ndarray,
    p: float,
    q: float,
    coupling_factory: Callable[..., Coupling],
) -> float:
    from .forward import wasserstein

    if idx.size == 0:
        return 0.0
    coupling = coupling_factory(X[:, idx], Y[:, idx], p, q)
    return wasserstein(X[:, idx], Y[:, idx], coupling, p, q)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two attribution vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / norm)
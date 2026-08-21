"""The neuralized two-layer computation of the Wasserstein distance.

Following Naumann et al. 2026 the Wasserstein distance with fixed optimal
coupling is rewritten as a two-layer network (equations (2a) and (2b)):

    layer 1:  z_kl   = ||x_k - y_l||_q
    layer 2:  W_p    = ( sum_kl gamma_kl z_kl^p )^{1/p}

Attribution is then a backward pass over this graph (see :mod:`wax.wax`).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist

from .coupling import Coupling

__all__ = ["pairwise_diffs", "pairwise_distance", "wasserstein", "detached_z"]


def pairwise_diffs(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Return the difference tensor ``Delta[k, l, :] = x_k - y_l`` of shape (N, M, d)."""
    return X[:, None, :] - Y[None, :, :]


def pairwise_distance(X: np.ndarray, Y: np.ndarray, q: float) -> np.ndarray:
    """Return ``z_kl = ||x_k - y_l||_q`` of shape (N, M)."""
    return np.asarray(cdist(X, Y, metric="minkowski", p=q), dtype=float)


def wasserstein(
    X: np.ndarray, Y: np.ndarray, coupling: Coupling, p: float, q: float
) -> float:
    """Return ``W_p(X, Y)`` for the fixed coupling, i.e. equation (2b)."""
    z = pairwise_distance(X, Y, q)
    gamma = coupling.gamma
    return float(np.sum(gamma * z**p) ** (1.0 / p))


def detached_z(X: np.ndarray, Y: np.ndarray, q: float, beta: float) -> np.ndarray:
    """Value of the 'detach trick' layer-1 activation of equation (4).

    The value equals ``||x_k - y_l||_q`` while the gradient-relevant norm is
    ``||x_k - y_l||_beta``; the paper uses this to make automatic-differentiation
    gradients coincide with the LRP rules.
    """
    dq = pairwise_distance(X, Y, q)
    db = pairwise_distance(X, Y, beta)
    return db * (dq / db)
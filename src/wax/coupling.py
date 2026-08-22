"""Optimal transport couplings between two empirical distributions.

The coupling matrix ``gamma`` is the joint distribution over source and target
indices that appears in the Wasserstein formulation (1) of Naumann et al. 2026.
It is obtained either as the exact optimal transport plan (solving (1)), as a
Sinkhorn-regularized plan, or set to the uniform coupling.

``gamma[i, j]`` is the probability mass transported from ``X[i]`` to ``Y[j]``
with row sums ``1 / N`` and column sums ``1 / M``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import ot
from scipy.spatial.distance import cdist

__all__ = [
    "Coupling",
    "cost_matrix",
    "exact",
    "sinkhorn",
    "uniform",
]


@dataclass
class Coupling:
    """A transport coupling ``gamma`` between ``X`` and ``Y``.

    Attributes
    ----------
    gamma : ndarray of shape (N, M)
        Joint probability matrix with marginal sums ``1 / N`` and ``1 / M``.
    kind : str
        ``"exact"``, ``"sinkhorn"``, ``"uniform"`` or ``"custom"``.
    p, q : float or None
        The Wasserstein model this coupling was built for.  :func:`wax.attribute`
        reads them so that the exponents cannot silently disagree between the
        transport problem and the explanation of it.  A coupling built by hand
        leaves them unset, and then the exponents must be passed explicitly.
    """

    gamma: np.ndarray
    kind: str = "custom"
    p: float | None = None
    q: float | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.gamma.shape

    def check_marginals(self, X: np.ndarray, Y: np.ndarray, atol: float = 1e-6) -> bool:
        """Return whether ``gamma`` has the correct source/target marginals."""
        n, m = len(X), len(Y)
        g = self.gamma
        ok_a = np.allclose(g.sum(axis=1), np.full(n, 1.0 / n), atol=atol)
        ok_b = np.allclose(g.sum(axis=0), np.full(m, 1.0 / m), atol=atol)
        return bool(ok_a and ok_b)


def cost_matrix(X: np.ndarray, Y: np.ndarray, p: float, q: float) -> np.ndarray:
    """The transportation cost matrix ``C[k, l] = ||x_k - y_l||_q^p``.

    This is the per-pair cost used in (1); its inner product with any coupling
    yields ``W_p^p``.
    """
    dist = cdist(X, Y, metric="minkowski", p=q)
    return np.asarray(dist, dtype=float) ** p


def exact(X: np.ndarray, Y: np.ndarray, p: float, q: float) -> Coupling:
    """Coupling from the exact optimal transport plan solving (1)."""
    n, m = len(X), len(Y)
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    C = cost_matrix(X, Y, p, q)
    gamma = ot.emd(a, b, C)
    return Coupling(gamma=gamma, kind="exact", p=float(p), q=float(q))


def sinkhorn(
    X: np.ndarray,
    Y: np.ndarray,
    p: float,
    q: float,
    reg: float = 1.0,
    max_iter: int = 5000,
    **kwargs,
) -> Coupling:
    """Coupling from the entropy-regularized (Sinkhorn) problem of Cuturi 2013."""
    n, m = len(X), len(Y)
    a = np.full(n, 1.0 / n)
    b = np.full(m, 1.0 / m)
    C = cost_matrix(X, Y, p, q)
    gamma = ot.sinkhorn(a, b, C, reg=reg, numItermax=max_iter, **kwargs)
    gamma = np.clip(gamma, 0.0, None)
    return Coupling(gamma=gamma, kind="sinkhorn", p=float(p), q=float(q))


def uniform(X: np.ndarray, Y: np.ndarray, p: float, q: float) -> Coupling:
    """The uniform coupling ``gamma[k, l] = 1 / (N M)``.

    This is the maximum-entropy limit of the Sinkhorn distance discussed in
    section IV-B of Naumann et al.  The coupling itself does not depend on ``p``
    or ``q``, but they are recorded because they still select the Wasserstein
    model that the coupling is used with.
    """
    gamma = np.full((len(X), len(Y)), 1.0 / (len(X) * len(Y)))
    return Coupling(gamma=gamma, kind="uniform", p=float(p), q=float(q))
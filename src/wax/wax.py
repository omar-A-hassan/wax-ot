"""WaX: LRP attribution of the Wasserstein distance to pairs and features.

Implements equations (3a), (3b) and the hyperparameter heuristic of Naumann et
al. 2026.  Attribution conserves the Wasserstein distance:

    sum_kl R_kl = sum_i R_i = W_p .

A gradient-identical formulation (Propositions 1 and 2 of the paper) is
provided through an automatic-differentiation path used for verification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coupling import Coupling, exact, sinkhorn, uniform
from .forward import pairwise_distance, wasserstein

__all__ = [
    "recommend_parameters",
    "instance_relevance",
    "feature_relevance",
    "Attribution",
    "attribute",
    "explain",
    "torch_gradient_attribution",
]

#: The coupling builders that :func:`explain` accepts by name.
COUPLINGS = {"exact": exact, "sinkhorn": sinkhorn, "uniform": uniform}


def recommend_parameters(p: float, q: float) -> tuple[float, float]:
    """The heuristic ``alpha = p``, ``beta = min(p + 2, q)`` of the paper."""
    return float(p), float(min(p + 2.0, q))


def instance_relevance(
    z: np.ndarray, gamma: np.ndarray, alpha: float, W: float
) -> np.ndarray:
    """Instance-pair relevance ``R_kl`` (equation (3a))."""
    num = gamma * np.power(z, alpha)
    denom = num.sum()
    if denom <= 0.0 or not np.isfinite(denom):
        return np.zeros_like(z)
    return (num / denom) * W


def feature_relevance(
    X: np.ndarray,
    Y: np.ndarray,
    R_kl: np.ndarray,
    beta: float,
    chunk_rows: int = 2048,
) -> np.ndarray:
    """Feature relevance ``R_i`` (equation (3b)).

    ``R_i = sum_kl R_kl * |x_ki - y_li|^beta / sum_i |x_ki - y_li|^beta``.
    ``X`` is processed in blocks so the ``(N, M, d)`` difference tensor never
    needs to be fully materialized.
    """
    n, d = X.shape
    result = np.zeros(d, dtype=float)
    for start in range(0, n, chunk_rows):
        xb = X[start : start + chunk_rows]
        Rb = R_kl[start : start + chunk_rows]
        diffs = xb[:, None, :] - Y[None, :, :]  # (nb, M, d)
        absd = np.abs(diffs)
        denom = np.power(absd, beta).sum(axis=2)  # (nb, M) = ||Delta||_beta^beta
        mask = denom > 0.0
        weights = np.zeros_like(denom)
        weights[mask] = Rb[mask] / denom[mask]
        result += np.einsum("kli,kl->i", np.power(absd, beta), weights)
    return result


@dataclass
class Attribution:
    """Result of a WaX attribution.

    Attributes
    ----------
    W : float
        The Wasserstein distance ``W_p(X, Y)``.
    z : ndarray of shape (N, M)
        Layer-1 activations ``||x_k - y_l||_q``.
    R_kl : ndarray of shape (N, M)
        Instance-pair relevance (3a).
    R_i : ndarray of shape (d,)
        Feature relevance (3b).
    alpha, beta : float
        The LRP hyperparameters actually used.
    conserved : bool
        Whether ``sum_i R_i`` matches ``W`` to ``role_tol``.
    """

    W: float
    z: np.ndarray
    R_kl: np.ndarray
    R_i: np.ndarray
    alpha: float
    beta: float
    conserved: bool = False


def attribute(
    X: np.ndarray,
    Y: np.ndarray,
    coupling: Coupling,
    p: float | None = None,
    q: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    check_conservation: bool = True,
    chunk_rows: int = 2048,
) -> Attribution:
    """Run the full WaX forward/backward pass (Algorithm 1 of the paper).

    ``p`` and ``q`` default to the values that the coupling was built with, so
    that the transport problem and the explanation of it cannot disagree by
    accident.  Pass them only to explain a Wasserstein model other than the one
    the coupling solves.  The paper does this on purpose for the maximally
    regularized coupling of Section IV-B.
    """
    if p is None:
        p = coupling.p
    if q is None:
        q = coupling.q
    if p is None or q is None:
        raise ValueError(
            "p and q are unknown. Give them to attribute(), or build the "
            "coupling with wax.exact, wax.sinkhorn or wax.uniform, which "
            "record them."
        )
    if alpha is None:
        alpha = p
    if beta is None:
        beta = min(p + 2.0, q)
    W = wasserstein(X, Y, coupling, p, q)
    z = pairwise_distance(X, Y, q)
    R_kl = instance_relevance(z, coupling.gamma, alpha, W)
    R_i = feature_relevance(X, Y, R_kl, beta, chunk_rows=chunk_rows)
    conserved = False
    if check_conservation:
        conserved = bool(np.isclose(R_i.sum(), W, rtol=1e-6, atol=1e-8) and
                         np.isclose(R_kl.sum(), W, rtol=1e-6, atol=1e-8))
    return Attribution(
        W=W,
        z=z,
        R_kl=R_kl,
        R_i=R_i,
        alpha=alpha,
        beta=beta,
        conserved=conserved,
    )


def explain(
    X: np.ndarray,
    Y: np.ndarray,
    p: float = 2.0,
    q: float = 2.0,
    coupling: str = "exact",
    **kwargs,
) -> Attribution:
    """Solve the transport problem and explain it in one call.

    This is the short form of the two-step sequence. The defaults are the model
    of the main experiments in the paper: the exact coupling with p = q = 2.

        a = wax.explain(X, Y)

        # the same operation, written out
        c = wax.exact(X, Y, p=2, q=2)
        a = wax.attribute(X, Y, c)

    ``coupling`` selects the transport plan. Use ``"exact"``, ``"sinkhorn"`` or
    ``"uniform"``. Other keyword arguments go to the coupling function, for
    example ``reg`` for Sinkhorn.
    """
    if coupling not in COUPLINGS:
        raise ValueError(f"unknown coupling {coupling!r}; use one of {sorted(COUPLINGS)}")
    return attribute(X, Y, COUPLINGS[coupling](X, Y, p, q, **kwargs))


def torch_gradient_attribution(
    X: np.ndarray,
    Y: np.ndarray,
    coupling: Coupling,
    p: float,
    q: float,
    alpha: float | None = None,
    beta: float | None = None,
) -> Attribution:
    """The authors' reference implementation (Supplementary Note D, Fig. S3).

    A transcription of the published code: two applications of the "detach
    trick" of (4) - one on layer 1 with ``beta``, one on layer 2 with ``alpha``
    - make plain automatic differentiation reproduce the LRP rules (3a)-(3b)
    for *arbitrary* alpha and beta, not only the gradient-identical case
    ``alpha = p, beta = q`` of Propositions 1 and 2::

        z = zbeta * (zq / zbeta).detach()          # value zq, gradient via zbeta
        W = Walpha * (Wp / Walpha).detach()        # value Wp, gradient via Walpha
        Ri = (source * source.grad).sum(0) + (target * target.grad).sum(0)

    This exists to cross-check the closed form in :func:`attribute`, which is
    what the package actually uses; it needs PyTorch and materializes the full
    (N, M) graph, so it is not the fast path.
    """
    import torch  # deferred import: optional dependency

    if alpha is None:
        alpha = p
    if beta is None:
        beta = min(p + 2.0, q)

    Xt = torch.tensor(X, dtype=torch.float64, requires_grad=True)
    Yt = torch.tensor(Y, dtype=torch.float64, requires_grad=True)
    g = torch.tensor(np.asarray(coupling.gamma), dtype=torch.float64)

    zq = torch.cdist(Xt, Yt, p=q)
    zbeta = torch.cdist(Xt, Yt, p=beta)
    z = zbeta * (zq / zbeta).detach()
    z.retain_grad()

    Wp = (g * z**p).sum() ** (1.0 / p)
    Walpha = (g * z**alpha).sum() ** (1.0 / alpha)
    W = Walpha * (Wp / Walpha).detach()
    W.backward()

    assert Xt.grad is not None and Yt.grad is not None and z.grad is not None
    R_i = (Xt * Xt.grad).sum(0) + (Yt * Yt.grad).sum(0)
    R_kl = z * z.grad
    return Attribution(
        W=float(Wp.detach()),
        z=zq.detach().numpy(),
        R_kl=R_kl.detach().numpy(),
        R_i=R_i.detach().numpy(),
        alpha=float(alpha),
        beta=float(beta),
    )

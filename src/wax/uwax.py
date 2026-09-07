"""U-WaX: attribution of the Wasserstein distance on input subspaces.

Section VI of Naumann et al. 2026.  The input space is rotated so that the
Wasserstein distance decomposes over ``C`` concept subspaces (columns of the
orthogonal matrix ``U``):

    z^c_kl = ||U_c^T (x_k - y_l)||_2                      (9a)
    S_c    = ( sum_kl gamma_kl (z^c_kl)^2 )^{1/2}         (9b)
    W_2    = ( sum_c S_c^2 )^{1/2}                        (9c)

Subspaces are learned to maximize the tailedness statistics ``Q_c``
(equations (11)-(12)) by alternating projected gradient ascent and
orthogonalization; the closed form for ``r = 2, C = 1`` is the top eigenvector
of ``A = sum_kl gamma_kl (x_k - y_l)(x_k - y_l)^T`` (Supplementary Note J).
Relevance is then propagated onto concepts (10a), pairs (10b) and features
(10c), conserving ``W_2``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .coupling import Coupling
from .wax import _chunk_rows

__all__ = ["eigen_subspace", "uwax_search", "uwax_attribute", "UwaxResult"]


def _diff_matrix(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return X[:, None, :] - Y[None, :, :]


def _block_gradient(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: np.ndarray,
    U_c: np.ndarray,
    Q: float,
    r: float,
) -> np.ndarray:
    """Gradient of ``Q_c`` w.r.t. ``U_c`` for general ``r``.

    ``dQ/dU = Q^{1-r} * sum_kl gamma_kl s^{r-2} D_kl D_kl^T U_c`` with
    ``s = ||U_c^T D_kl||``, i.e. per-pair weight ``gamma_kl * z2^{r/2 - 1}``.
    """
    if Q <= 0.0:
        return np.zeros_like(U_c)
    z2 = _squared_projections(X, Y, U_c)
    w = np.asarray(gamma) * np.power(z2, r / 2.0 - 1.0)
    A = np.zeros((X.shape[1], X.shape[1]), dtype=float)
    n = len(X)
    chunk = _chunk_rows(n, len(Y), X.shape[1])
    for start in range(0, n, chunk):
        D = _diff_matrix(X[start : start + chunk], Y)
        A += np.einsum("kli,kl,klj->ij", D, w[start : start + chunk], D)
    return (Q ** (1.0 - r)) * (A @ U_c)


def _squared_projections(X: np.ndarray, Y: np.ndarray, U_c: np.ndarray) -> np.ndarray:
    """``z2_kl = ||U_c^T (x_k - y_l)||^2``, row-blocked."""
    n = len(X)
    M = len(Y)
    z2 = np.empty((n, M), dtype=float)
    chunk = _chunk_rows(n, M, X.shape[1])
    for start in range(0, n, chunk):
        D = _diff_matrix(X[start : start + chunk], Y)
        proj = D @ U_c
        z2[start : start + chunk] = np.einsum("kli,kli->kl", proj, proj)
    return z2


def _tailedness(
    X: np.ndarray, Y: np.ndarray, gamma: np.ndarray, U_c: np.ndarray, r: float
) -> tuple[float, np.ndarray]:
    """``Q_c`` of a subspace block and its squared projections."""
    z2 = _squared_projections(X, Y, U_c)
    Q = float(np.sum(np.asarray(gamma) * np.power(z2, r / 2.0)) ** (1.0 / r))
    return Q, z2


def eigen_subspace(
    X: np.ndarray,
    Y: np.ndarray,
    coupling: Coupling,
    dims: list[int],
) -> np.ndarray:
    """Orthogonal subspace blocks from the eigenvectors of ``A``.

    Closed form of the ``r = 2`` objective: block columns of ``U`` are the top
    eigenvectors of ``A = sum_kl gamma_kl (x_k - y_l)(x_k - y_l)^T``.
    """
    gamma = np.asarray(coupling.gamma)
    n = len(X)
    A = np.zeros((X.shape[1], X.shape[1]), dtype=float)
    chunk = _chunk_rows(n, len(Y), X.shape[1])
    for start in range(0, n, chunk):
        D = _diff_matrix(X[start : start + chunk], Y)
        G = gamma[start : start + chunk]
        A += np.einsum("kli,kl,klj->ij", D, G, D)
    A = 0.5 * (A + A.T)
    _, V = np.linalg.eigh(A)  # ascending eigenvalues
    order = np.arange(X.shape[1])[::-1]
    cols = []
    pos = 0
    for c in dims:
        cols.append(V[:, order[pos : pos + c]])
        pos += c
    return np.hstack(cols)


def _project_u(U: np.ndarray, dims: list[int]) -> np.ndarray:
    """Orthonormalize stacked subspace blocks, preserving block identity.

    Each block is orthogonalized in sequence against all previously fixed
    blocks, then internally orthonormalized by QR.
    """
    blocks = []
    pos = 0
    for c in dims:
        blk = U[:, pos : pos + c]
        pos += c
        for prev in blocks:
            blk = blk - prev @ (prev.T @ blk)
        q, _ = np.linalg.qr(blk)
        blocks.append(q)
    return np.hstack(blocks)


def uwax_search(
    X: np.ndarray,
    Y: np.ndarray,
    coupling: Coupling,
    r: float,
    dims: list[int],
    seed: int | None = None,
    max_iter: int = 500,
    lr: float = 0.1,
    tol: float = 1e-6,
    init: str = "eigen",
    verbose: bool = False,
) -> tuple[np.ndarray, list[float]]:
    """Maximize ``sum_c Q_c`` over the orthogonal ``U`` (equation (12)).

    Alternating projected gradient ascent: each block is moved along the
    gradient of its ``Q_c``, then the stacked matrix is re-orthogonalized.
    ``init="eigen"`` starts from the closed-form ``r=2`` solution,
    ``init="random"`` from a random orthogonal matrix.

    Returns ``(U, objective_history)``.
    """
    gamma = np.asarray(coupling.gamma)
    d = X.shape[1]
    if init == "eigen":
        U = eigen_subspace(X, Y, coupling, dims)
    else:
        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
        blocks = []
        pos = 0
        for c in dims:
            blocks.append(Q[:, pos : pos + c])
            pos += c
        U = np.hstack(blocks)
    U = _project_u(U, dims)
    bounds = []
    pos = 0
    for c in dims:
        bounds.append((pos, pos + c))
        pos += c
    history: list[float] = []
    prev = -np.inf
    for it in range(max_iter):
        total = 0.0
        for s, e in bounds:
            U_c = U[:, s:e]
            Q_c, _ = _tailedness(X, Y, gamma, U_c, r)
            total += Q_c
            grad = _block_gradient(X, Y, gamma, U_c, Q_c, r)
            U[:, s:e] = U_c + lr * grad
        U = _project_u(U, dims)
        history.append(total)
        if it > 0 and abs(total - prev) <= tol * max(1.0, abs(total)):
            if verbose:
                print(f"uwax_search converged at iter {it}, sum Q = {total:.6f}")
            break
        prev = total
    return U, history


@dataclass
class UwaxResult:
    """Result of :func:`uwax_attribute`.

    ``R_c``, ``R_kl_c`` and ``R_i_c`` are the concept, pair and feature
    relevances of (10a)-(10c); ``W2`` is the decomposed distance (9c) and is
    conserved by all three relevance levels.
    """

    U: np.ndarray
    """Orthogonal rotation carrying the concept blocks in its first columns."""
    dims: tuple[int, ...]
    """Per-subspace dimensionalities."""
    W2: float
    """Decomposed Wasserstein distance, equal to ``W_2`` for ``p = q = 2``."""
    S_c: np.ndarray
    """Concept scores of (9b)."""
    R_c: np.ndarray
    """Concept relevance of (10a)."""
    R_kl_c: list[np.ndarray]
    """Instance-pair relevance within each subspace (10b)."""
    R_i_c: np.ndarray
    """Feature relevance within each subspace (10c), shape (C, d)."""
    R_i: np.ndarray
    """Total feature relevance, ``sum_c R_i_c``."""
    captured: float
    """Fraction of ``W2`` explained by the concept blocks ``sum S_c^2 / W2^2``."""
    residual: float
    """Relevance of the unexplained orthogonal complement, ``W2 - sum_c R_c``."""
    conserved: bool
    """Whether the relevance levels conserve ``W2``."""


def uwax_attribute(
    X: np.ndarray,
    Y: np.ndarray,
    coupling: Coupling,
    U: np.ndarray,
    r: float,
    dims: Sequence[int] | None = None,
) -> UwaxResult:
    """Compute the subspace attribution of equations (9)-(10).

    ``dims`` must be the block sizes given to :func:`uwax_search`. The blocks
    of (8) may have different sizes, and ``U`` alone does not give them: every
    column of an orthogonal matrix is orthogonal to every other column.
    ``None`` means one column for each concept.
    """
    gamma = np.asarray(coupling.gamma)
    n, d = X.shape

    dims = tuple(dims) if dims is not None else (1,) * U.shape[1]
    if sum(dims) != U.shape[1]:
        raise ValueError(f"dims sum to {sum(dims)}, but U has {U.shape[1]} columns")

    z2_c: list[np.ndarray] = []
    S_c = np.empty(len(dims))
    boundaries = []
    pos = 0
    for c in dims:
        boundaries.append((pos, pos + c))
        pos += c
    for ic, (s, e) in enumerate(boundaries):
        U_c = U[:, s:e]
        _, z2 = _tailedness(X, Y, gamma, U_c, 2.0)
        z2_c.append(z2)
        S_c[ic] = float(np.sqrt(np.sum(gamma * z2)))
    from .forward import wasserstein

    W2 = wasserstein(X, Y, coupling, 2.0, 2.0)
    R_c = S_c**2 / W2 if W2 > 0.0 else np.zeros_like(S_c)

    R_kl_c: list[np.ndarray] = []
    R_i_c = np.zeros((len(dims), d), dtype=float)
    for ic, (s, e) in enumerate(boundaries):
        U_c = U[:, s:e]
        z2 = z2_c[ic]
        num = gamma * z2
        denom = num.sum()
        rkl = np.full_like(gamma, 0.0)
        if denom > 0.0:
            rkl = (num / denom) * R_c[ic]
        R_kl_c.append(rkl)
        chunk = _chunk_rows(n, len(Y), d, arrays=3)
        for b0 in range(0, n, chunk):
            D = _diff_matrix(X[b0 : b0 + chunk], Y)
            # fused rank-k projection: (D @ U_c) @ U_c.T  — 64× cheaper than D @ (U_c@U_c.T)
            proj = (D @ U_c) @ U_c.T  # (nb, M, d) = D_hat of (10c)
            den = z2[b0 : b0 + chunk]
            mask = den > 0.0
            w = np.zeros_like(den)
            w[mask] = rkl[b0 : b0 + chunk][mask] / den[mask]
            R_i_c[ic] += np.einsum("kli,kl,kli->i", D, w, proj)

    R_i = R_i_c.sum(axis=0)
    captured = float(np.sum(S_c**2) / max(W2**2, 1e-300))
    residual = float(W2 - R_i.sum())
    per_concept_ok = all(
        bool(np.isclose(R_kl_c[ic].sum(), R_c[ic], rtol=1e-5, atol=1e-8)) and
        bool(np.isclose(R_i_c[ic].sum(), R_c[ic], rtol=1e-5, atol=1e-8))
        for ic in range(len(dims))
    )
    conserved = per_concept_ok
    return UwaxResult(
        U=U,
        dims=dims,
        W2=W2,
        S_c=S_c,
        R_c=R_c,
        R_kl_c=R_kl_c,
        R_i_c=R_i_c,
        R_i=R_i,
        captured=captured,
        residual=residual,
        conserved=conserved,
    )
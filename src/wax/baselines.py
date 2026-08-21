"""Baseline attribution methods contributed in the paper (Section IV)."""

from __future__ import annotations

import numpy as np

from .forward import wasserstein

__all__ = ["MeanShift", "Occlusion", "CouplingBaseline", "Uniform", "LogisticBaseline"]


class MeanShift:
    """``R_i = (E[x] - E[y])_i^2``, feature-wise squared mean shift."""

    def __call__(self, X: np.ndarray, Y: np.ndarray, p: float, q: float) -> np.ndarray:
        return (np.mean(X, axis=0) - np.mean(Y, axis=0)) ** 2


class Occlusion:
    """``R_i = W_p(x, y) - W_p(x_{:,-i}, y_{:,-i})``.

    Re-solves the transport problem with the ``i``-th feature removed, hence
    the (potentially expensive) ``d`` re-solves.
    """

    def __init__(self, coupling_factory):
        self.coupling_factory = coupling_factory

    def __call__(self, X: np.ndarray, Y: np.ndarray, p: float, q: float) -> np.ndarray:
        coupling = self.coupling_factory(X, Y, p, q)
        W = wasserstein(X, Y, coupling, p, q)
        d = X.shape[1]
        R = np.empty(d, dtype=float)
        for i in range(d):
            mask = np.arange(d) != i
            c = self.coupling_factory(X[:, mask], Y[:, mask], p, q)
            R[i] = W - wasserstein(X[:, mask], Y[:, mask], c, p, q)
        return R


class CouplingBaseline:
    """``R_i = sum_kl gamma_kl (x_ki - y_li)^2``.

    The per-feature cost of the transport plan (Euclidean), i.e. an ablation
    of WaX with ``alpha = 1, beta = 2`` up to a global scaling.
    """

    def __call__(self, X: np.ndarray, Y: np.ndarray, p: float, q: float) -> np.ndarray:
        coupling = self.coupling_factory(X, Y, p, q)
        diff = X[:, None, :] - Y[None, :, :]
        c = np.asarray(coupling.gamma)
        return np.einsum("kli,kl->i", diff**2, c)

    def __init__(self, coupling_factory):
        self.coupling_factory = coupling_factory


class Uniform:
    """Baseline assigning ``R_i = 1`` to every feature."""

    def __call__(self, X: np.ndarray, Y: np.ndarray, p: float, q: float) -> np.ndarray:
        return np.ones(X.shape[1], dtype=float)


class LogisticBaseline:
    """Attributions from a logistic regression trained to separate domains.

    Two score variants follow Supplementary Note H:

    - ``"gradient"``: ``R_i = w_i^2`` (squared weights, sensitivity),
    - ``"gi"``:       ``R_i = w_i (E[x] - E[y])_i`` (gradient times input).
    """

    def __init__(self, mode: str = "gradient", seed: int = 0, max_iter: int = 2000):
        if mode not in {"gradient", "gi"}:
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.seed = seed
        self.max_iter = max_iter

    def __call__(self, X: np.ndarray, Y: np.ndarray, p: float, q: float) -> np.ndarray:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # The source is class 1 so that ``w`` points towards X and the
        # gradient-times-input score below matches the sign of the paper's
        # R_i = w_i (E[x] - E[y])_i.  Inputs are scaled before fitting: on raw
        # units the L2 penalty crushes every coefficient and the attribution
        # collapses to zero.
        Z = np.vstack([X, Y])
        t = np.concatenate([np.ones(len(X)), np.zeros(len(Y))])
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=self.max_iter, random_state=self.seed),
        )
        clf.fit(Z, t)
        w = clf[-1].coef_[0] / clf[0].scale_  # undo the scaling, back to input units
        if self.mode == "gradient":
            return w**2
        shift = np.mean(X, axis=0) - np.mean(Y, axis=0)
        return w * shift
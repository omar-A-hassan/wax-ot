"""Matplotlib figures reproducing the paper's use-case visualizations.

Each plotting function saves a PNG under ``figs/``.  Figures follow the paper's
layout conventions (horizontal feature bars, instance transport arrows,
per-subspace bars, feature-removal curves).
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

__all__ = [
    "fig_feature_and_transport",
    "fig_subspaces",
    "fig_removal_curves",
    "fig_face_subspace_words",
]


def _save(fig, fname):
    fig.tight_layout()
    fig.savefig(fname, dpi=140)
    plt.close(fig)


def fig_feature_and_transport(
    fname: str,
    R_i: np.ndarray,
    feature_names: list[str],
    R_kl: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    gamma: np.ndarray,
    n_arrows: int = 14,
    cbar_label: str = "R_kl",
) -> None:
    """Fig-4(a)-style: feature relevance bars plus transport arrows.

    The transport plot projects the source and target points onto the two most
    relevant features and draws the strongest coupled pairs.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    order = np.argsort(R_i)
    ys = np.arange(len(R_i))
    ax1.barh(ys, R_i[order], color="#2c6fbb")
    ax1.set_yticks(ys, [feature_names[i] for i in order], fontsize=9)
    ax1.set_xlabel("feature relevance R_i")
    ax1.set_title("feature attribution (3b)", fontsize=10)

    top2 = np.argsort(-R_i)[:2]
    x0, x1 = top2[0], top2[1]
    ax2.scatter(X[:, x0], X[:, x1], s=8, label="source", alpha=0.7)
    ax2.scatter(Y[:, x0], Y[:, x1], s=8, label="target", alpha=0.7, marker="x")
    flat = np.column_stack([R_kl.ravel(), np.arange(R_kl.size)])
    flat = flat[np.argsort(-flat[:, 0])][:n_arrows]
    rmax = flat[:, 0].max()
    for r, idx in flat:
        k, ell = int(idx) // Y.shape[0], int(idx) % Y.shape[0]
        lw = 0.5 + 2.5 * (r / rmax)
        ax2.plot(
            [X[k, x0], Y[ell, x0]],
            [X[k, x1], Y[ell, x1]],
            color="#c0392b",
            linewidth=lw,
            alpha=0.85,
        )
    ax2.set_xlabel(feature_names[x0])
    ax2.set_ylabel(feature_names[x1])
    ax2.legend(fontsize=8)
    ax2.set_title("strongest couplings, top-2 features", fontsize=10)
    _save(fig, fname)


def fig_subspaces(
    fname: str,
    S_c: np.ndarray,
    R_c: np.ndarray,
    R_i_c: np.ndarray,
    feature_names: list[str],
    residual: float,
) -> None:
    """Fig-4(b)-style: per-subspace feature relevance bars (10c)."""
    C = len(S_c)
    fig, ax = plt.subplots(figsize=(8.6, 0.6 * C * R_i_c.shape[1] + 2.6))
    y = 0.0
    ticks, labels = [], []
    for c in range(C):
        group_start = y
        vals = R_i_c[c]
        order = np.argsort(vals)
        for i in order:
            ax.barh(y, vals[i], height=0.8, color=plt.cm.tab10(c))
            y += 1.0
        ticks.append((group_start + y - 1.0) / 2.0)
        labels.append(f"S{c + 1}  (R={R_c[c]:.3f})")
    ax.set_yticks(ticks, labels, fontsize=9)
    ax.set_xlabel("feature relevance within subspace")
    ax.set_title(f"U-WaX subspaces | S_c={np.round(S_c, 3)} | residual={residual:.3f}", fontsize=10)
    _save(fig, fname)


def fig_removal_curves(
    fname: str, pct: np.ndarray, curves: dict[str, np.ndarray], baseline: float
) -> None:
    """Fig-3(a)-style: accuracy after removing the most relevant features."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    colors = {"WaX": "#2c6fbb", "Coupling": "#c0392b", "MeanShift": "#27ae60"}
    for label, vals in curves.items():
        ax.plot(pct * 100, vals, "-o", ms=4, label=label, color=colors.get(label))
    ax.axhline(baseline, ls="--", color="gray", label="baseline (all features)")
    ax.set_xlabel("% features removed")
    ax.set_ylabel("1-NN accuracy on target")
    ax.legend(fontsize=9)
    _save(fig, fname)


def fig_face_subspace_words(
    fname: str,
    subspace_words: list[tuple[str, dict[str, float]]],
    per_subspace: dict[str, np.ndarray],
    cbar_label: str = "relevance ratio",
) -> None:
    """Fig-5-style: top aligned words per learned subspace."""
    n = len(subspace_words)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.6), squeeze=False)
    for ax, (label, word_scores) in zip(axes[0], subspace_words, strict=False):
        words = list(word_scores.keys())[:8]
        scores = [word_scores[w] for w in words]
        ax.barh(np.arange(len(words)), scores, color="#8e44ad")
        ax.set_yticks(np.arange(len(words)), words, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"subspace {label}", fontsize=10)
        ax.set_xlabel("cosine similarity to CLIP word")
    _save(fig, fname)
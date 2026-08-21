"""Script 03 - Section VII: exploring an aging phenomenon with U-WaX.

Simulates an abalone cohort observed twice (~one year apart) and decomposes
the Wasserstein distance into three concept subspaces with U-WaX (r=4),
contrasting the subspace analysis with a k-means clustering baseline.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from wax import attribute, exact, uwax_attribute, uwax_search
from wax.datasets import abalone_aging_split
from wax.plotting import fig_feature_and_transport, fig_subspaces

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs")
os.makedirs(FIGDIR, exist_ok=True)


def clustering_baseline(X, Y, gamma, k=3, seed=0):
    """Attribution from k-means prototypes on (source, transported) pairs.

    Analog of the clustering baseline of [6] used in the paper's Fig. 4c.
    """
    from sklearn.cluster import KMeans

    transported = np.empty_like(X)
    for krow in range(len(X)):
        transported[krow] = Y[np.argmax(gamma[krow])]
    Z = np.hstack([X, transported])
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    R_c = []
    R_i_c = np.zeros((k, X.shape[1]))
    for c in range(k):
        sel = np.nonzero(km.labels_ == c)[0]
        delta = transported[sel].mean(axis=0) - X[sel].mean(axis=0)
        R_c.append(float(np.sum(delta**2)))
        R_i_c[c] = delta**2
    return R_c, R_i_c


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=350)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X, Y, names = abalone_aging_split(n=args.n, seed=args.seed)
    coupling = exact(X, Y, 2, 2)
    a = attribute(X, Y, coupling, 2, 2)

    print(f"\n=== Abalone aging: W_2 = {a.W:.4f} ===")
    order = np.argsort(-a.R_i)
    print("WaX feature relevance (ranked):")
    for i in order:
        print(f"  {names[i]:<14} R = {a.R_i[i]:.4f}")

    fig_feature_and_transport(
        os.path.join(FIGDIR, "fig_03_abalone_wax.png"),
        a.R_i, names, a.R_kl, X, Y, coupling.gamma,
    )

    print("\nU-WaX (r=4, three 1-dimensional subspaces):")
    U, hist = uwax_search(X, Y, coupling, r=4, dims=[1, 1, 1], seed=args.seed,
                          lr=0.3, max_iter=1500, tol=1e-7)
    s = uwax_attribute(X, Y, coupling, U, r=4)
    print(f"  objective: {hist[0]:.4f} -> {hist[-1]:.4f} ({len(hist)} iters)")
    for c in range(3):
        print(f"  S{c + 1}: score={s.S_c[c]:.4f} relevance={s.R_c[c]:.4f}")
        tops = np.argsort(-s.R_i_c[c])[:3]
        print("     top features: " + ", ".join(f"{names[i]} ({s.R_i_c[c][i]:.4f})" for i in tops))
    print(f"  captured: {s.captured:.3f}  residual relevance: {s.residual:.4f}")

    fig_subspaces(
        os.path.join(FIGDIR, "fig_03_abalone_uwax.png"),
        s.S_c, s.R_c, s.R_i_c, names, s.residual,
    )

    print("\nclustering baseline ([6]-analog):")
    R_c, R_i_c = clustering_baseline(X, Y, coupling.gamma, k=3, seed=args.seed)
    for c in range(3):
        tops = np.argsort(-R_i_c[c])[:3]
        print(f"  cluster {c + 1}: local shift {R_c[c]:.4f} | top: "
              + ", ".join(f"{names[i]} ({R_i_c[c][i]:.4f})" for i in tops))


if __name__ == "__main__":
    main()
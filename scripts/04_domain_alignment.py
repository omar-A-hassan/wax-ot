"""Script 04 - Section V analog: identifying robust features for domain alignment.

Uses a synthetic two-domain dataset with spurious (domain-shifted) features.
Features are scored by the paper's domain-alignment rule R_i * sigma_i^lambda
(equation (7)); pruning the most relevant directions is evaluated by 1-NN
accuracy on the target domain, against Coupling- and MeanShift-based pruning.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from wax import attribute, exact
from wax.baselines import CouplingBaseline, MeanShift
from wax.datasets import synthetic_domains

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs")
os.makedirs(FIGDIR, exist_ok=True)


def pooled_std(X, Y):
    return np.sqrt(
        (np.var(X, axis=0) + np.var(Y, axis=0)) / 2.0
    )


def knn_accuracy(Xtr, ytr, Xte, yte, cols):
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(Xtr[:, cols], ytr)
    return float(knn.score(Xte[:, cols], yte))


def removal_pcts(n_feats):
    return np.arange(0, n_feats) / n_feats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--d", type=int, default=15)
    ap.add_argument("--informative", type=int, default=5)
    ap.add_argument("--spurious", type=int, default=4)
    ap.add_argument("--lambda", dest="lam", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    Xs, ys, Xt, yt, meta = synthetic_domains(
        n=args.n, d=args.d, n_informative=args.informative,
        n_spurious=args.spurious, seed=args.seed,
    )
    coupling = exact(Xs, Xt, 2, 2)
    R_wa = attribute(Xs, Xt, coupling, 2, 2).R_i
    R_cp = CouplingBaseline(exact)(Xs, Xt, 2, 2)
    R_ms = MeanShift()(Xs, Xt, 2, 2)
    sigma = pooled_std(Xs, Xt)

    baseline = knn_accuracy(Xs, ys, Xt, yt, np.arange(args.d))
    print(f"\n=== domain alignment (d={args.d}), baseline 1-NN accuracy = {baseline:.3f} ===")
    spurious = np.nonzero(meta["roles"] == "spurious")[0]
    print("spurious features: " + ", ".join(str(i) for i in spurious))

    curves = {}
    pcts = removal_pcts(args.d)
    for lam in args.lam:
        score = R_wa * sigma ** lam
        order = np.argsort(-score)
        curves[f"WaX lambda={lam}"] = [
            knn_accuracy(Xs, ys, Xt, yt, order[int(p) :]) for p in pcts * args.d
        ]
        if lam == args.lam[0]:
            sp = [int(i) for i in np.nonzero(meta["roles"] == "spurious")[0]]
            ranks = {int(f): int(np.nonzero(order == f)[0][0]) for f in sp}
            print(f"lambda={lam}: spurious feature ranks (lower = pruned first): {ranks}")

    order = np.argsort(-R_cp)
    curves["Coupling"] = [knn_accuracy(Xs, ys, Xt, yt, order[int(p) :]) for p in pcts * args.d]
    order = np.argsort(-R_ms)
    curves["MeanShift"] = [knn_accuracy(Xs, ys, Xt, yt, order[int(p) :]) for p in pcts * args.d]

    print("\naccuracy after removing the most relevant features:")
    header = f"{'removed':>8}" + "".join(f"{k:>16}" for k in curves)
    print(header)
    for j, pct in enumerate(pcts):
        print(f"{pct * 100:>6.0f}%" + "".join(f"{curves[k][j]:>16.3f}" for k in curves))

    from wax.plotting import fig_removal_curves

    fig_removal_curves(
        os.path.join(FIGDIR, "fig_04_domain_alignment.png"),
        pcts, curves, baseline,
    )


if __name__ == "__main__":
    main()
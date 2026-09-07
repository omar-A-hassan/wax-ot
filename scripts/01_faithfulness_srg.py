"""Script 01 - Section IV.A: explanation faithfulness via the SRG metric (Table I).

Compares WaX against the MeanShift, Occlusion and Coupling baselines on the
Symmetric Relevance Gain (equation (5)), over the five Wasserstein models of
Table I and three of its datasets.

The preprocessing and the bootstrap protocol follow Supplementary Notes E and
F of the arXiv version (arXiv:2505.06123), which the IEEE article omits. The
rebuilt datasets then match the N/M/d of Table I. Wisconsin (332/163/30) and
Musk1 (133/167/166) match exactly, and Wine has two rows more (3783 against
3781 / 1086 / 12). The numbers below are therefore comparable to the paper.

The one deviation is the bootstrap budget. Note F averages 100 trials, which
needs about 18 hours for Wine on an 8 GB laptop, so ``--reps`` defaults to 3.
Use ``--reps 100`` for the protocol of the paper.

Standardization is necessary here. The SRG compares Wasserstein distances
across feature subsets and is not scale invariant. On raw units one feature of
large magnitude dominates every method, and all four give the same ranking and
the same score.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from wax import attribute, exact, srg
from wax.baselines import CouplingBaseline, MeanShift, Occlusion
from wax.datasets import load_musk1, load_wine_quality, load_wisconsin, preprocess_tabular

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figs")
RESULTDIR = os.path.join(ROOT, "results")

DATASETS = {"wisconsin": load_wisconsin, "musk1": load_musk1, "wine-quality": load_wine_quality}

#: The (p, q) Wasserstein models of Table I.
MODELS = [(1, 2), (2, 1), (2, 2), (2, np.inf), (10, 2)]

METHODS = ["MeanShift", "Occlusion", "Coupling", "WaX"]

#: Table I of the paper for the datasets we cover, keyed by (dataset, p, q),
#: in METHODS order.
PAPER = {
    ("wisconsin", 1, 2): (1.82, 1.81, 1.82, 1.82),
    ("wisconsin", 2, 1): (7.06, 7.07, 7.04, 7.08),
    ("wisconsin", 2, 2): (1.82, 1.84, 1.84, 1.85),
    ("wisconsin", 2, "inf"): (0.50, 0.24, 0.50, 0.54),
    ("wisconsin", 10, 2): (1.99, 2.33, 2.10, 2.34),
    ("musk1", 1, 2): (0.94, 2.19, 2.19, 2.19),
    ("musk1", 2, 1): (6.24, 12.85, 10.65, 12.84),
    ("musk1", 2, 2): (1.02, 2.07, 2.04, 2.07),
    ("musk1", 2, "inf"): (0.31, 0.89, 0.57, 0.97),
    ("musk1", 10, 2): (0.92, 2.14, 1.72, 2.30),
    ("wine-quality", 1, 2): (1.08, 1.08, 1.09, 1.09),
    ("wine-quality", 2, 1): (2.58, 2.55, 2.56, 2.58),
    ("wine-quality", 2, 2): (1.01, 1.03, 1.03, 1.04),
    ("wine-quality", 2, "inf"): (0.47, 0.50, 0.45, 0.50),
    ("wine-quality", 10, 2): (0.89, 0.96, 0.96, 0.98),
}


def attributions(X, Y, p, q):
    """The four per-feature relevance vectors, in METHODS order."""
    return [
        MeanShift()(X, Y, p, q),
        Occlusion(exact)(X, Y, p, q),
        CouplingBaseline(exact)(X, Y, p, q),
        attribute(X, Y, exact(X, Y, p, q), p, q).R_i,
    ]


def run(X, Y, p, q, reps, max_n, seed):
    """Bootstrap SRG scores, following Supplementary Note F.

    Each trial resamples the source and the target with replacement, back up to
    N and M. It then adds N(0, 1e-8) noise. The note prescribes this noise to
    break the ties that the duplicate rows make in the transport problem.
    """
    rng = np.random.default_rng(seed)
    n = len(X) if max_n <= 0 else min(len(X), max_n)
    m = len(Y) if max_n <= 0 else min(len(Y), max_n)
    scores, ranks_equal = [], []
    for _ in range(reps):
        x = X[rng.choice(len(X), size=n, replace=True)]
        y = Y[rng.choice(len(Y), size=m, replace=True)]
        x = x + rng.normal(0.0, 1e-8, x.shape)
        y = y + rng.normal(0.0, 1e-8, y.shape)
        R = attributions(x, y, p, q)
        scores.append([srg(x, y, p, q, r, exact) for r in R])
        # how many of the four methods share WaX's exact feature ranking
        wax_order = tuple(np.argsort(-R[-1]))
        ranks_equal.append(sum(tuple(np.argsort(-r)) == wax_order for r in R))
    arr = np.asarray(scores)
    return {
        "mean": arr.mean(axis=0).tolist(),
        "std": arr.std(axis=0).tolist(),
        "reps": reps,
        "n": int(n),
        "m": int(m),
        "d": int(X.shape[1]),
        "methods_sharing_wax_ranking": float(np.mean(ranks_equal)),
    }


def write_markdown(results, path):
    lines = [
        "### Table I - Symmetric Relevance Gain (eq. 5), higher is better",
        "",
        "`paper` / **`ours`** for each method, best per row in bold. Datasets are "
        "rebuilt with the Supplementary Note E preprocessing and match Table I's "
        "N/M/d. In the paper WaX is best or tied in 26 of Table I's 30 rows, with "
        "Occlusion taking most of the rest.",
        "",
        "| dataset | N / M / d | model | " + " | ".join(METHODS) + " | WaX best? |",
        "|---|---|---|" + "---|" * (len(METHODS) + 1),
    ]
    for (name, p, q), row in results.items():
        best = int(np.argmax(row["mean"]))
        paper = PAPER[(name, p, q)]
        cells = [
            f"{paper[i]:.2f} / " + (f"**{v:.2f}**" if i == best else f"{v:.2f}")
            for i, v in enumerate(row["mean"])
        ]
        won = "yes" if best == len(METHODS) - 1 else (
            "tie" if abs(row["mean"][-1] - row["mean"][best]) < 5e-3 else "no"
        )
        lines.append(
            f"| {name} | {row['n']}/{row['m']}/{row['d']} | p={p}, q={q} | "
            + " | ".join(cells)
            + f" | {won} |"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--reps", type=int, default=3, help="Note F uses 100")
    ap.add_argument("--max-n", type=int, default=0, help="0 = full N/M, as the paper")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(RESULTDIR, exist_ok=True)
    print(f"\n=== SRG faithfulness, {args.reps} reps, <= {args.max_n} rows per side ===")
    print(f"{'dataset':<14}{'model':<14}" + "".join(f"{m:>16}" for m in METHODS) + "  ties")

    results = {}
    for name in args.datasets:
        X, Y = preprocess_tabular(*DATASETS[name]()[:2])  # Note E
        for p, q in MODELS:
            row = run(X, Y, p, q, args.reps, args.max_n, args.seed)
            key = (name, p, "inf" if np.isinf(q) else q)
            results[key] = row
            print(
                f"{name:<14}p={p}, q={str(q):<7}"
                + "".join(
                    f"{m:>11.4f}+-{s:<.3f}" for m, s in zip(row["mean"], row["std"], strict=False)
                )
                + f"  {row['methods_sharing_wax_ranking']:.1f}/4"
            )

    with open(os.path.join(RESULTDIR, "01_table1.json"), "w") as f:
        json.dump(
            {
                "meta": {"methods": METHODS, "reps": args.reps, "max_n": args.max_n,
                         "protocol": "arXiv:2505.06123 Supplementary Notes E and F"},
                "paper": {f"{k[0]}|p={k[1]}|q={k[2]}": v for k, v in PAPER.items()},
                "ours": {f"{k[0]}|p={k[1]}|q={k[2]}": v for k, v in results.items()},
            },
            f,
            indent=2,
        )
    write_markdown(results, os.path.join(RESULTDIR, "01_table1.md"))
    print(f"\nwrote {RESULTDIR}/01_table1.json and 01_table1.md")

    from matplotlib import pyplot as plt

    names = sorted({k[0] for k in results})
    fig, axes = plt.subplots(1, len(names), figsize=(4.4 * len(names), 4.0), squeeze=False)
    width = 0.2
    for ax, name in zip(axes[0], names, strict=False):
        keys = [k for k in results if k[0] == name]
        x = np.arange(len(keys))
        for j, m in enumerate(METHODS):
            ax.bar(x + j * width, [results[k]["mean"][j] for k in keys], width, label=m)
        ax.set_xticks(x + width * 1.5, [f"p={k[1]}\nq={k[2]}" for k in keys], fontsize=7)
        ax.set_title(f"{name} (d={results[keys[0]]['d']})", fontsize=9)
        ax.set_ylabel("SRG")
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_01_srg.png"), dpi=140)


if __name__ == "__main__":
    main()

"""Script 02 - Section IV.B: characterizing transport phenomena (Table III).

Reproduces Table III of Naumann et al.: mean cosine similarity between the
ground-truth per-feature transport relevance (equation (6)) and the relevance
computed by WaX and four baselines, on three periodic UCI series at delays of
one to six hours.

The protocol follows Supplementary Notes E and G of the arXiv version
(arXiv:2505.06123), which the IEEE article omits:

* channels with too many missing values are dropped, series are standardized to
  zero mean and unit variance, and samples that are missing a value or deviate
  by more than three standard deviations are removed *together with their
  coupled match*, so the ground-truth coupling stays complete (Note E);
* scores are averaged over every source hour ``t`` in ``{0, ..., 23}`` (Note G);
* crucially, the source and target subsets are drawn from **disjoint** sets of
  days - ``D~_S = {x_(t+kT)}_(k in K1)`` and ``D~_T = {x_(t+dt+kT)}_(k in K2)``
  with ``K1`` and ``K2`` a random partition.  This is what stops the method from
  seeing the ground-truth coupling; reusing the same days inflates WaX from
  ~0.95 to ~0.98 on Appliances by leaking the pairing (Note G);
* only cases with at least 50 source and target instances are kept (Note G).

The ground truth of (6) is still computed over the full coupled set - it is the
*methods* that are denied the pairing, not the experimenter.

The PLISM histopathology rows of Table III need three pathology foundation
models and are out of scope for this reproduction.

"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from wax import attribute, cosine_similarity, exact, uniform
from wax.baselines import LogisticBaseline, MeanShift, Uniform
from wax.datasets import (
    SAMPLES_PER_HOUR,
    fetch_air_quality,
    fetch_appliances,
    fetch_electricity,
    preprocess_series,
    ts_shift,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figs")
RESULTDIR = os.path.join(ROOT, "results")

#: name -> (loader, samples per period)
DATASETS = {
    "air-quality": (fetch_air_quality, 24),
    "electricity": (fetch_electricity, 96),
    "appliances": (fetch_appliances, 144),
}

METHODS = ["Uniform", "MeanShift", "Logistic[L2]", "Logistic[GI]", "WaX y*", "WaX y*reg"]

#: Table III of the paper, keyed by dataset then delay in hours, in METHODS order.
#: The trailing entry of each row is the median subset size N~ the paper reports.
PAPER = {
    "air-quality": {
        1: (0.83, 0.82, 0.65, 0.47, 0.85, 0.80, 363),
        2: (0.84, 0.87, 0.68, 0.40, 0.91, 0.83, 354),
        3: (0.85, 0.88, 0.67, 0.40, 0.93, 0.87, 353),
        4: (0.86, 0.87, 0.67, 0.40, 0.94, 0.89, 349),
        5: (0.87, 0.92, 0.67, 0.43, 0.95, 0.90, 351),
        6: (0.88, 0.91, 0.67, 0.42, 0.95, 0.91, 352),
    },
    "electricity": {
        1: (0.84, 0.61, 0.47, 0.43, 0.83, 0.88, 1213),
        2: (0.90, 0.68, 0.54, 0.42, 0.86, 0.94, 1201),
        3: (0.93, 0.71, 0.55, 0.35, 0.86, 0.96, 1192),
        4: (0.94, 0.74, 0.60, 0.42, 0.87, 0.97, 1191),
        5: (0.94, 0.75, 0.60, 0.37, 0.88, 0.98, 1221),
        6: (0.94, 0.77, 0.61, 0.37, 0.88, 0.98, 1234),
    },
    "appliances": {
        1: (0.38, 0.28, 0.06, 0.10, 0.51, 0.41, 124),
        2: (0.47, 0.35, 0.10, 0.14, 0.62, 0.50, 121),
        3: (0.53, 0.40, 0.15, 0.19, 0.69, 0.56, 120),
        4: (0.58, 0.43, 0.20, 0.25, 0.74, 0.62, 119),
        5: (0.62, 0.48, 0.23, 0.29, 0.78, 0.66, 119),
        6: (0.65, 0.54, 0.26, 0.34, 0.81, 0.69, 119),
    },
}


def score_subset(Xs: np.ndarray, Xt: np.ndarray, truth: np.ndarray) -> list[float]:
    """Cosine similarity to the ground truth (6) for each method, in METHODS order."""
    wax_exact = attribute(Xs, Xt, exact(Xs, Xt, 2, 2), 2, 2).R_i
    wax_unif = attribute(Xs, Xt, uniform(Xs, Xt, 2, 2), 2, 2).R_i
    return [
        cosine_similarity(Uniform()(Xs, Xt, 2, 2), truth),
        cosine_similarity(MeanShift()(Xs, Xt, 2, 2), truth),
        cosine_similarity(LogisticBaseline("gradient")(Xs, Xt, 2, 2), truth),
        cosine_similarity(LogisticBaseline("gi")(Xs, Xt, 2, 2), truth),
        cosine_similarity(wax_exact, truth),
        cosine_similarity(wax_unif, truth),
    ]


def source_hours(dt_hours: int, max_subsets: int) -> np.ndarray:
    """Source hours t in {0, ..., 23} with t + dt <= 23 (Note G)."""
    hours = np.arange(0, 24 - dt_hours)
    if max_subsets and len(hours) > max_subsets:
        hours = hours[np.linspace(0, len(hours) - 1, max_subsets).astype(int)]
    return hours


def run_dataset(
    name: str, X: np.ndarray, valid: np.ndarray, period: int, delays,
    max_subsets: int, max_n: int, min_n: int, seed: int,
):
    sph = SAMPLES_PER_HOUR[name]
    rng = np.random.default_rng(seed)
    rows = {}
    for dt_hours in delays:
        dt = dt_hours * sph
        scores, sizes = [], []
        for t in source_hours(dt_hours, max_subsets):
            Xs, Xt, truth = ts_shift(
                X, period=period, t=int(t) * sph, dt=dt, valid=valid
            )
            # ground truth (6) over the full coupled set, before the partition
            if len(Xs) < 2 * min_n or not np.any(truth > 0):
                continue
            k = rng.permutation(len(Xs))
            half = len(k) // 2
            K1, K2 = k[:half], k[half : 2 * half]  # disjoint, Note G
            Xs_sub, Xt_sub = Xs[K1][:max_n], Xt[K2][:max_n]
            if min(len(Xs_sub), len(Xt_sub)) < min_n:
                continue
            scores.append(score_subset(Xs_sub, Xt_sub, truth))
            sizes.append(len(Xs))
        arr = np.asarray(scores)
        rows[dt_hours] = {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "n_subsets": int(len(arr)),
            "n_median": int(np.median(sizes)),
        }
        print(
            f"  {name:<12} dt={dt_hours}h  N~={rows[dt_hours]['n_median']:<5}"
            + "".join(f"{v:>9.2f}" for v in rows[dt_hours]["mean"])
        )
    return rows


def write_markdown(results: dict, path: str) -> None:
    """Render a paper-vs-ours comparison table."""
    lines = [
        "### Table III - mean cosine similarity to the ground truth of eq. (6)",
        "",
        "`paper` / **`ours`** for each method. Best per row in bold.",
        "",
        "| dataset | dt | N~ paper / ours | " + " | ".join(METHODS) + " |",
        "|---|---|---|" + "---|" * len(METHODS),
    ]
    for name, rows in results.items():
        for dt_hours, row in sorted(rows.items(), key=lambda kv: int(kv[0])):
            paper = PAPER[name][int(dt_hours)]
            best = int(np.argmax(row["mean"]))
            cells = []
            for i, v in enumerate(row["mean"]):
                ours = f"**{v:.2f}**" if i == best else f"{v:.2f}"
                cells.append(f"{paper[i]:.2f} / {ours}")
            lines.append(
                f"| {name} | {dt_hours}h | {paper[6]} / {row['n_median']} | "
                + " | ".join(cells)
                + " |"
            )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--delays", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--max-subsets", type=int, default=0, help="0 = every source hour")
    ap.add_argument("--max-n", type=int, default=1500)
    ap.add_argument("--min-n", type=int, default=50, help="Note G minimum subset size")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(RESULTDIR, exist_ok=True)
    print("\n=== Table III: cosine similarity to ground-truth transport (6) ===")
    print(f"  {'dataset':<12} {'delay':<10}{'N~':<5}" + "".join(f"{m:>9}" for m in METHODS))

    results = {}
    for name in args.datasets:
        loader, period = DATASETS[name]
        try:
            X = loader()
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:<12} skipped: {exc}")
            continue
        X, valid = preprocess_series(X)  # Note E
        results[name] = run_dataset(
            name, X, valid, period, args.delays, args.max_subsets,
            args.max_n, args.min_n, args.seed,
        )

    meta = {
        "methods": METHODS,
        "max_subsets": args.max_subsets,
        "max_n": args.max_n,
        "min_n": args.min_n,
        "protocol": "arXiv:2505.06123 Supplementary Notes E and G",
        "note": "PLISM rows of Table III are out of scope (pathology foundation models).",
    }
    with open(os.path.join(RESULTDIR, "02_table3.json"), "w") as f:
        json.dump({"meta": meta, "paper": PAPER, "ours": results}, f, indent=2)
    write_markdown(results, os.path.join(RESULTDIR, "02_table3.md"))
    print(f"\nwrote {RESULTDIR}/02_table3.json and 02_table3.md")

    from matplotlib import pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(4.6 * len(results), 4.0), squeeze=False)
    for ax, (name, rows) in zip(axes[0], results.items(), strict=False):
        dts = sorted(int(k) for k in rows)
        for j, m in enumerate(METHODS):
            ax.plot(dts, [rows[k]["mean"][j] for k in dts], "-o", ms=4, label=m)
            ax.plot(
                dts, [PAPER[name][k][j] for k in dts], "--", lw=1, alpha=0.45,
                color=ax.lines[-1].get_color(),
            )
        ax.set_title(f"{name}\n(solid = ours, dashed = paper)", fontsize=9)
        ax.set_xlabel("delay $\\Delta t$ [h]")
        ax.set_ylabel("cosine similarity to (6)")
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig_02_characterize.png"), dpi=140)


if __name__ == "__main__":
    main()

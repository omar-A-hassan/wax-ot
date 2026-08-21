# wax-ot

WaX: explainable Wasserstein distances.

An independent, unofficial implementation of

> **Wasserstein Distances Made Explainable: Insights Into Dataset Shifts and
> Transport Phenomena.** P. Naumann, J. Kauffmann, G. Montavon.
> IEEE TPAMI 48(6), 2026. doi:10.1109/TPAMI.2026.3656947.
> Preprint with the supplementary notes: [arXiv:2505.06123](https://arxiv.org/abs/2505.06123).

This project is not affiliated with the authors. At the time of writing there is
no official implementation, so this appears to be the first public one. See
[Verification](#verification) for what has been checked and what has not.

## What it does

A Wasserstein distance tells you *how far apart* two datasets are. It does not
tell you *why*. WaX answers the second question: it rewrites the distance with a
fixed optimal coupling as a two-layer network, then runs a backward pass with
layer-wise relevance propagation rules. The result is an exact decomposition of
the distance onto:

- **instance pairs** `R_kl`, which source and target points drive the distance,
- **input features** `R_i`, which measured quantities drive it,
- **subspaces** (U-WaX), which interpretable directions drive it.

The decomposition conserves the distance, so `sum(R_i) == sum(R_kl) == W_p`
exactly. That is a checkable property, and it is checked in the test suite.

## Install

```
pip install wax-ot
```

The core package is numpy, scipy and POT only. Optional extras:

| Extra | Adds | Needed for |
|---|---|---|
| `wax-ot[viz]` | matplotlib | `wax.plotting` |
| `wax-ot[torch]` | torch | the reference cross-check |
| `wax-ot[repro]` | the above plus scikit-learn and open-clip | the experiment scripts |

## Quick start

```python
import numpy as np
from wax import attribute, exact

X = np.random.randn(300, 8)
Y = np.random.randn(300, 8) + np.array([0.6, 0, 0, 0.2, 0, 0, -0.4, 0])

coupling = exact(X, Y, p=2, q=2)        # or sinkhorn(...) / uniform(...)
a = attribute(X, Y, coupling, p=2, q=2)

a.W                # the Wasserstein distance itself
a.R_i              # per-feature relevance, shape (8,)
a.R_kl             # per-pair relevance, shape (300, 300)
a.conserved        # True when sum(R_i) == sum(R_kl) == W
```

### How to read the output

`R_i` is in the same units as `W`, not a normalised score. A feature with
`R_i = 0.4` when `W = 1.0` is responsible for 40 percent of the distance
between the two datasets. The values sum to `W` by construction, so you can
read them as a budget. `R_kl[k, l]` is the share contributed by moving source
point `k` to target point `l`, which is what makes outlier pairs visible.

Sign conventions matter less than scale: standardise your features first if
they are measured in different units, or the largest-magnitude column will
dominate every attribution.

### Subspaces

```python
from wax import uwax_search, uwax_attribute

U, history = uwax_search(X, Y, coupling, r=4, dims=[1, 1, 1], seed=0)
s = uwax_attribute(X, Y, coupling, U, r=4)

s.R_c              # relevance of each subspace
s.captured         # fraction of W explained by the subspaces found
```

## Modules

| Module | Paper reference |
|---|---|
| `wax.coupling` | couplings: exact OT (eq. 1), Sinkhorn, uniform |
| `wax.forward` | neuralized graph `z_kl -> W_p` (eq. 2) |
| `wax.wax` | LRP rules (eq. 3), alpha and beta heuristic, reference autodiff path |
| `wax.uwax` | subspace search and attribution (eqs. 9 to 12) |
| `wax.metrics` | Symmetric Relevance Gain (eq. 5), cosine similarity (eq. 6) |
| `wax.baselines` | MeanShift, Occlusion, Coupling, Uniform, Logistic |
| `wax.datasets` | UCI loaders and the preprocessing of Supplementary Note E |

## Verification

The supplementary notes are not in the IEEE article. The arXiv preprint has all
of them, including Note D, which publishes the authors' own implementation of
WaX. Everything below follows Notes D, E, F and G.

### The algorithm is verified to machine precision

This is the strongest claim here, and it does not depend on any dataset.

`wax.attribute` is an independent closed form of the LRP rules.
`wax.torch_gradient_attribution` is a transcription of the authors' published
code (Note D, Fig. S3), which uses the "detach trick" twice so that plain
autodiff reproduces the rules for arbitrary `alpha` and `beta`. The two agree to
within **2.4e-15**, across three couplings and eight combinations of
`(p, q, alpha, beta)` including `alpha != p` and `beta != q`:

```
pytest tests/test_wax.py -k matches_authors_reference_implementation
```

Conservation and the U-WaX closed form of Note J are covered by the rest of the
64 tests.

### Table III is reproduced

Mean absolute error against the published table, per method, averaged over the
six delays. Full numbers in [results/02_table3.md](results/02_table3.md).

| dataset | Uniform | MeanShift | Logistic L2 | Logistic GI | WaX | WaX reg | row MAE |
|---|---|---|---|---|---|---|---|
| air-quality | 0.032 | 0.096 | 0.016 | 0.052 | 0.062 | 0.030 | 0.048 |
| electricity | 0.160 | 0.058 | 0.227 | 0.152 | 0.041 | 0.081 | 0.120 |
| appliances | 0.020 | 0.061 | 0.115 | 0.139 | 0.035 | 0.007 | 0.063 |
| **overall** | 0.071 | 0.072 | 0.119 | 0.114 | **0.046** | **0.039** | **0.077** |

WaX's own two columns reproduce best, which is the part that matters most. The
rebuilt datasets also match the published shapes: feature counts `d = 9, 7, 25`
are exact, and the Air Quality subset sizes are within five days of the paper on
every row.

### Table I is reproduced on three of six datasets

Covers Musk1, Wine and Wisconsin. Crime, Mice and Robot are not attempted.
Applying the Note E preprocessing rebuilds the paper's dataset shapes almost
exactly, which is good evidence the chain is right:

| dataset | ours (N/M/d) | paper (N/M/d) |
|---|---|---|
| wisconsin | 332 / 163 / 30 | 332 / 163 / 30, exact |
| musk1 | 133 / 167 / 166 | 133 / 167 / 166, exact |
| wine-quality | 3783 / 1086 / 12 | 3781 / 1086 / 12 |

Mean absolute error against Table I, full numbers in
[results/01_table1.md](results/01_table1.md):

| dataset | MeanShift | Occlusion | Coupling | WaX | MAE |
|---|---|---|---|---|---|
| wine-quality | 0.004 | 0.005 | 0.012 | 0.007 | **0.007** |
| wisconsin | 0.075 | 0.084 | 0.080 | 0.093 | 0.083 |
| musk1 | 0.325 | 0.198 | 0.222 | 0.174 | 0.230 |
| **overall** | 0.135 | 0.095 | 0.105 | 0.091 | **0.107** |

WaX ranks best or tied in 13 of these 15 rows; the paper itself ranks it best or
tied in 14 of the same 15, so the pattern reproduces as well as the values do.
Error tracks sample size, as Note F predicts it should: Wine has 3783 source
rows and reproduces to 0.007, Musk1 has 133 with 166 features and reproduces to
0.230.

### Known deviations

- **Bootstrap budget.** Note F averages 100 trials. That is about 18 hours for
  Wine on an 8 GB laptop, so `--reps` defaults to 3. This is the dominant source
  of error in Table I. Pass `--reps 100` for the paper's protocol.
- **Electricity client selection.** We use the first seven client columns that
  are active over the full span. The notes do not say which seven of the 370 the
  paper uses. This is the weakest row in Table III.
- **Air Quality channel selection.** We keep the nine channels with at least 90
  percent coverage, which gives the paper's `d = 9`, but the threshold is not
  stated and other rules also give nine.
- **Not attempted.** The PLISM rows of Table III need three histopathology
  foundation models. Table S2 confidence intervals are not computed.
- **Not scored.** Scripts 03, 04 and 05 produce figures that are qualitatively
  consistent with the paper but are not compared against a published number.
  Treat them as demonstrations. Script 05 caps both face datasets at 500 images
  to fit in 8 GB.

All results above were produced on an Apple M3 laptop with 8 GB of memory, CPU
only.

## Reproducing the experiments

```
pip install "wax-ot[repro]"
python scripts/01_faithfulness_srg.py     # Table I,   writes results/01_table1.*
python scripts/02_characterize_shift.py   # Table III, writes results/02_table3.*
python scripts/03_uwax_abalone.py         # Section VI,   U-WaX subspaces
python scripts/04_domain_alignment.py     # Section V,    domain alignment
python scripts/05_celeba_lfw.py           # Section VIII, CelebA against LFW
```

Scripts 02 and 05 download datasets on first use, about 260 MB for the
Electricity load diagrams and 1.3 GB for CelebA. They are cached in
`~/.cache/wax`, or in `$WAX_CACHE` if you set it.

## Development

```
uv sync --all-extras
uv run pytest
uv run ruff check
```

## Data and licensing

The code is MIT licensed. The method and the text of the paper belong to the
authors; the paper itself is not redistributed here, only linked. The bundled
data files are redistributed under their original licences: UCI abalone
(Nash et al., 1995) and the google-10000-english word list. Datasets fetched at
run time are not redistributed at all.

## Citation

Please cite the original paper first. If you also want to cite this
implementation, see [CITATION.cff](CITATION.cff).

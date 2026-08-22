# wax-ot

WaX: an explanation method for Wasserstein distances.

This package is an independent implementation of the method in this article:

> **Wasserstein Distances Made Explainable: Insights Into Dataset Shifts and
> Transport Phenomena.** P. Naumann, J. Kauffmann, G. Montavon.
> IEEE Transactions on Pattern Analysis and Machine Intelligence,
> volume 48, number 6, pages 6393 to 6406, 2026.
> doi:[10.1109/TPAMI.2026.3656947](https://doi.org/10.1109/TPAMI.2026.3656947).
> Preprint with the supplementary notes:
> [arXiv:2505.06123](https://arxiv.org/abs/2505.06123).

The authors of the article did not write this package. They are not related to
it. On the date of this release, no official implementation is public.

Section [Verification](#verification) gives the tests and the results.

## Description

A Wasserstein distance gives the quantity of the difference between two
datasets. It does not give the cause of the difference.

WaX gives the cause. The method writes the distance as a network of two layers
and keeps the coupling constant. The method then does a backward pass with
layer-wise relevance propagation rules. The result is a division of the
distance into relevance values for:

- instance pairs, `R_kl`: the source points and the target points that cause
  the distance;
- input features, `R_i`: the measured quantities that cause the distance;
- subspaces (U-WaX): the directions in the input space that cause the distance.

The division keeps the total. Thus `sum(R_i)` and `sum(R_kl)` are both equal to
the distance `W_p`. The test suite examines this property.

## Installation

```
pip install wax-ot
```

The core package needs numpy, scipy and POT only. Four optional extras are
available:

| Extra | Adds | Necessary for |
|---|---|---|
| `wax-ot[viz]` | matplotlib | the `wax.plotting` module |
| `wax-ot[torch]` | torch | the comparison with the reference implementation |
| `wax-ot[repro]` | the two extras above, scikit-learn and open-clip | the experiment scripts |

## Use

```python
import numpy as np
import wax

X = np.random.randn(300, 8)
Y = np.random.randn(300, 8) + np.array([0.6, 0, 0, 0.2, 0, 0, -0.4, 0])

a = wax.explain(X, Y)

a.W                # the Wasserstein distance
a.R_i              # relevance of each feature, shape (8,)
a.R_kl             # relevance of each instance pair, shape (300, 300)
a.conserved        # True if sum(R_i) and sum(R_kl) are equal to W
```

### Other Wasserstein models

The defaults of `wax.explain` are the model of the main experiments in the
article: the exact coupling with p = q = 2. To use a different model, build the
coupling and then explain it. The coupling keeps a record of its `p` and `q`,
and `wax.attribute` reads them. Therefore the transport problem and the
explanation of it cannot disagree by accident.

```python
coupling = wax.exact(X, Y, p=3, q=2)   # also wax.sinkhorn or wax.uniform
a = wax.attribute(X, Y, coupling)      # p and q come from the coupling
```

Give `p` and `q` to `wax.attribute` only to explain a model other than the one
the coupling solves. The article does this for the regularized coupling of
Section IV-B.


### How to read the relevance values

The relevance values use the same units as the distance `W`. They are not
percentages. If `W` is 1.0, and one feature has an `R_i` of 0.4, then that
feature causes 40 percent of the distance. The sum of the values is always
equal to `W`.

The value `R_kl[k, l]` is the part of the distance that comes from the movement
of source point `k` to target point `l`. Large values show unusual pairs.

Make the scale of all features equal before you calculate the relevance values.
If you do not do this, the feature with the largest values controls the result.
The function `wax.datasets.standardize` does this operation.

### Many features

With `beta = 2`, which the parameter heuristic selects whenever `q = 2`, the
feature relevance has a closed form that needs no difference tensor. Data with
thousands of features is therefore possible: 1000 by 1000 points with 18000
features takes about half a second for the relevance step. The tensor form of
the same calculation would need 144 GB.

Other values of `beta` use the direct sum, which builds a tensor of shape
`(N, M, d)` in blocks of `chunk_rows` rows. Reduce `chunk_rows` if the memory
is too large.

### Subspaces

```python
U, history = wax.uwax_search(X, Y, coupling, r=4, dims=[1, 1, 1], seed=0)
s = wax.uwax_attribute(X, Y, coupling, U, r=4)

s.R_c              # relevance of each subspace
s.captured         # the part of W that the subspaces explain
```

## Modules

| Module | Part of the article |
|---|---|
| `wax.coupling` | the couplings: exact transport (eq. 1), Sinkhorn, uniform |
| `wax.forward` | the network `z_kl` to `W_p` (eq. 2) |
| `wax.wax` | the relevance rules (eq. 3) and the reference implementation |
| `wax.uwax` | the subspace search and its relevance values (eqs. 9 to 12) |
| `wax.metrics` | Symmetric Relevance Gain (eq. 5), cosine similarity (eq. 6) |
| `wax.baselines` | MeanShift, Occlusion, Coupling, Uniform, Logistic |
| `wax.datasets` | the UCI loaders and the preprocessing of Supplementary Note E |

The `wax` namespace holds the method. The modules `wax.baselines`,
`wax.datasets`, `wax.plotting` and `wax.words` support the experiments. They
are not part of the interface that the version number promises to keep.

## Verification

The IEEE article does not contain the supplementary notes. The arXiv preprint
contains all of them. Supplementary Note D contains the implementation of WaX
by the authors. This package follows Notes D, E, F and G.

### Comparison with the implementation by the authors

The function `wax.attribute` is an independent closed form of the relevance
rules. The function `wax.torch_gradient_attribution` is a copy of the code in
Note D, Figure S3. That code uses the detach operation two times. Therefore
automatic differentiation gives the same result as the rules for all values of
`alpha` and `beta`.

The two functions agree to 2.4e-15. The test uses three couplings and eight
sets of the parameters `(p, q, alpha, beta)`. Four of the eight sets have an
`alpha` that is different from `p`, or a `beta` that is different from `q`. To
run this test:

```
pytest tests/test_wax.py -k matches_authors_reference_implementation
```

The test suite has 64 tests. This comparison is 24 of them. The other 40 tests
examine the conservation property and the closed form of Supplementary Note J.

### Results for Table III

The table gives the mean absolute error against the published table. Each value
is a mean over the six delays. File
[results/02_table3.md](results/02_table3.md) gives all the numbers.

| Dataset | Uniform | MeanShift | Logistic L2 | Logistic GI | WaX | WaX reg | Mean |
|---|---|---|---|---|---|---|---|
| air-quality | 0.032 | 0.096 | 0.016 | 0.052 | 0.062 | 0.030 | 0.048 |
| electricity | 0.160 | 0.058 | 0.227 | 0.152 | 0.041 | 0.081 | 0.120 |
| appliances | 0.020 | 0.061 | 0.115 | 0.139 | 0.035 | 0.007 | 0.063 |
| **all** | 0.071 | 0.072 | 0.119 | 0.114 | **0.046** | **0.039** | **0.077** |

The two WaX columns have the smallest error. The datasets also have the
dimensions of the published table. The feature counts 9, 7 and 25 are correct.
The subset sizes for air-quality are within five days of the published values
in each row.

### Results for Table I

The scripts use three of the six datasets in Table I. They do not use Crime,
Mice or Robot.

The preprocessing of Supplementary Note E gives almost the same dataset
dimensions as the article:

| Dataset | This package (N/M/d) | Article (N/M/d) |
|---|---|---|
| wisconsin | 332 / 163 / 30 | 332 / 163 / 30 |
| musk1 | 133 / 167 / 166 | 133 / 167 / 166 |
| wine-quality | 3783 / 1086 / 12 | 3781 / 1086 / 12 |

The next table gives the mean absolute error against Table I. File
[results/01_table1.md](results/01_table1.md) gives all the numbers.

| Dataset | MeanShift | Occlusion | Coupling | WaX | Mean |
|---|---|---|---|---|---|
| wine-quality | 0.004 | 0.005 | 0.012 | 0.007 | **0.007** |
| wisconsin | 0.075 | 0.084 | 0.080 | 0.093 | 0.083 |
| musk1 | 0.325 | 0.198 | 0.222 | 0.174 | 0.230 |
| **all** | 0.135 | 0.095 | 0.105 | 0.091 | **0.107** |

WaX has the best value, or an equal value, in 13 of these 15 rows. In the
article, WaX has the best value, or an equal value, in 14 of the same 15 rows.

The error increases when the number of samples decreases. Supplementary Note F
gives the same relation. Wine has 3783 source rows and an error of 0.007. Musk1
has 133 source rows and 166 features, and an error of 0.230.

### Known differences from the article

- **Number of bootstrap trials.** Note F uses 100 trials. On a laptop with 8 GB
  of memory, 100 trials of the Wine dataset need approximately 18 hours. Thus
  the default value of `--reps` is 3. This is the largest cause of error in
  Table I. Use `--reps 100` for the protocol of the article.
- **Selection of the electricity clients.** The scripts use the first seven
  client columns with data for the full period. The notes do not say which
  seven of the 370 clients the article uses. This dataset has the largest error
  in Table III.
- **Selection of the air-quality channels.** The scripts keep the nine channels
  with a minimum of 90 percent of the values present. This gives the nine
  channels of the article. The notes do not give the threshold, and other
  thresholds also give nine channels.
- **Tests that are not done.** The PLISM rows of Table III need three
  histopathology models. The confidence intervals of Table S2 are not
  calculated.
- **Results that are not compared.** Scripts 03, 04 and 05 make figures. The
  figures agree with the article, but no script compares them with a published
  number. Script 05 uses a maximum of 500 images from each face dataset,
  because of the memory limit.

A laptop with an Apple M3 processor and 8 GB of memory made all the results
above. The scripts used the processor only.

## How to run the experiments

```
pip install "wax-ot[repro]"
python scripts/01_faithfulness_srg.py     # Table I. Writes results/01_table1.*
python scripts/02_characterize_shift.py   # Table III. Writes results/02_table3.*
python scripts/03_uwax_abalone.py         # Section VI. U-WaX subspaces
python scripts/04_domain_alignment.py     # Section V. Domain alignment
python scripts/05_celeba_lfw.py           # Section VIII. CelebA and LFW
```

Scripts 02 and 05 download datasets at the first use. The electricity dataset
is approximately 260 MB. The CelebA dataset is approximately 1.3 GB. The
scripts keep the files in `~/.cache/wax`. To use a different directory, set the
environment variable `WAX_CACHE`.

## Development

```
uv sync --all-extras
uv run pytest
uv run ruff check
```

## Data and licenses

The code has the MIT license. The method and the text of the article belong to
the authors. This repository does not contain the article. It gives a link to
the article.

Two data files are in the package. They keep their original licenses: the UCI
abalone data (Nash et al., 1995) and the google-10000-english word list. The
scripts download the other datasets at run time. This repository does not
contain them.

## Citation

Cite the article first. This package is an implementation of the method in the
article. It is not new research.

```bibtex
@article{naumann2026wasserstein,
  author  = {Naumann, Philip and Kauffmann, Jacob and Montavon, Gr{\'e}goire},
  title   = {Wasserstein Distances Made Explainable: Insights Into Dataset
             Shifts and Transport Phenomena},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume  = {48},
  number  = {6},
  pages   = {6393--6406},
  year    = {2026},
  doi     = {10.1109/TPAMI.2026.3656947}
}
```

To cite this package in addition to the article, use the metadata in
[CITATION.cff](CITATION.cff).

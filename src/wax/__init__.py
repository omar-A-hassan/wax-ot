"""WaX: an explanation method for Wasserstein distances.

An independent implementation of Naumann, Kauffmann and Montavon, "Wasserstein
Distances Made Explainable", IEEE TPAMI 48(6), 2026.  The method divides a
Wasserstein distance into relevance values for instance pairs, input features
and subspaces, by layer-wise relevance propagation on the neuralized distance
graph.

This namespace holds the method itself.  The support code for the experiments
stays in its own modules, because it is not part of the interface that the
version number promises to keep:

- ``wax.baselines``: the baseline methods that the paper compares against;
- ``wax.datasets``: the dataset loaders and the preprocessing of Note E;
- ``wax.plotting``: the figures (needs the ``viz`` extra);
- ``wax.words``: the word list for the subspace descriptions.
"""

from importlib.metadata import PackageNotFoundError, version

from .coupling import Coupling, cost_matrix, exact, sinkhorn, uniform
from .forward import wasserstein
from .metrics import cosine_similarity, srg
from .uwax import UwaxResult, eigen_subspace, uwax_attribute, uwax_search
from .wax import (
    Attribution,
    attribute,
    explain,
    recommend_parameters,
    torch_gradient_attribution,
)

try:
    __version__ = version("wax-ot")
except PackageNotFoundError:  # running from a source tree with no install
    __version__ = "0.0.0.dev0"

__all__ = [
    # one-call entry point
    "explain",
    # transport couplings
    "Coupling",
    "exact",
    "sinkhorn",
    "uniform",
    "cost_matrix",
    "wasserstein",
    # attribution
    "Attribution",
    "attribute",
    "recommend_parameters",
    "torch_gradient_attribution",
    # subspaces
    "UwaxResult",
    "eigen_subspace",
    "uwax_search",
    "uwax_attribute",
    # evaluation
    "srg",
    "cosine_similarity",
]

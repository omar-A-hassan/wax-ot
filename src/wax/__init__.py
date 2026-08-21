"""WaX - Wasserstein distances made explainable.

Reimplementation of Naumann, Kauffmann and Montavon, "Wasserstein Distances Made
Explainable", IEEE TPAMI 48(6), 2026.  Attributes a Wasserstein distance between
two empirical distributions to instance pairs, input features and subspaces via
layer-wise relevance propagation on the neuralized distance graph.
"""

from importlib.metadata import PackageNotFoundError, version

from .baselines import CouplingBaseline, LogisticBaseline, MeanShift, Occlusion, Uniform
from .coupling import Coupling, cost_matrix, exact, sinkhorn, uniform
from .datasets import (
    ABALONE_FEATURES,
    abalone_aging_split,
    fetch_air_quality,
    fetch_appliances,
    fetch_electricity,
    load_abalone,
    synthetic_domains,
    synthetic_ts,
    ts_shift,
)
from .forward import pairwise_diffs, pairwise_distance, wasserstein
from .metrics import cosine_similarity, srg
from .uwax import eigen_subspace, uwax_attribute, uwax_search
from .wax import (
    attribute,
    feature_relevance,
    instance_relevance,
    recommend_parameters,
    torch_gradient_attribution,
)
from .words import load_words

try:
    __version__ = version("wax-ot")
except PackageNotFoundError:  # running from a source tree with no install
    __version__ = "0.0.0.dev0"

__all__ = [
    "Coupling",
    "exact",
    "sinkhorn",
    "uniform",
    "cost_matrix",
    "pairwise_distance",
    "pairwise_diffs",
    "wasserstein",
    "instance_relevance",
    "feature_relevance",
    "recommend_parameters",
    "attribute",
    "torch_gradient_attribution",
    "MeanShift",
    "Occlusion",
    "CouplingBaseline",
    "Uniform",
    "LogisticBaseline",
    "srg",
    "cosine_similarity",
    "eigen_subspace",
    "uwax_search",
    "uwax_attribute",
    "load_abalone",
    "abalone_aging_split",
    "fetch_air_quality",
    "fetch_appliances",
    "fetch_electricity",
    "ts_shift",
    "synthetic_ts",
    "synthetic_domains",
    "ABALONE_FEATURES",
    "load_words",
]
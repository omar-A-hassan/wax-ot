"""Tests for the reusable WaX package."""

import numpy as np
import pytest

import wax
from wax import (
    MeanShift,
    attribute,
    cosine_similarity,
    eigen_subspace,
    exact,
    sinkhorn,
    srg,
    torch_gradient_attribution,
    uniform,
    uwax_attribute,
    uwax_search,
    wasserstein,
)
from wax.baselines import LogisticBaseline
from wax.datasets import (
    abalone_aging_split,
    standardize,
    synthetic_domains,
    synthetic_ts,
)


def make_data(n=60, d=5, seed=0, shift=None):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    if shift is None:
        rng2 = np.random.default_rng(seed + 1)
        shift = rng2.normal(size=d) * 0.6
    Y = rng.normal(size=(n, d)) + shift
    return X, Y


def factories():
    return [exact, sinkhorn, uniform]


@pytest.mark.parametrize("factory", factories())
@pytest.mark.parametrize("p,q", [(1, 1), (2, 2), (3, 2), (2, 3)])
def test_conservation(factory, p, q):
    X, Y = make_data()
    c = factory(X, Y, p, q)
    a = attribute(X, Y, c, p, q)
    assert a.conserved
    assert np.isclose(a.R_kl.sum(), a.W, rtol=1e-6)
    assert np.isclose(a.R_i.sum(), a.W, rtol=1e-6)


@pytest.mark.parametrize("factory", factories())
@pytest.mark.parametrize("p,q", [(2, 2), (3, 2), (5, 3)])
def test_gradient_identity(factory, p, q):
    pytest.importorskip("torch")
    X, Y = make_data()
    c = factory(X, Y, p, q)
    a_np = attribute(X, Y, c, p, q)
    a_t = torch_gradient_attribution(X, Y, c, p, q)
    assert np.allclose(a_np.R_kl, a_t.R_kl, rtol=1e-5, atol=1e-6)
    assert np.allclose(a_np.R_i, a_t.R_i, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("factory", factories())
@pytest.mark.parametrize(
    "p,q,alpha,beta",
    [(1, 2, 1, 2), (2, 2, 2, 2), (2, 1, 2, 1), (3, 2, 3, 2),
     (2, 2, 4, 3), (10, 2, 10, 2), (2, 3, 2, 4), (1, 1, 1, 3)],
)
def test_matches_authors_reference_implementation(factory, p, q, alpha, beta):
    """Our closed form must equal the published code of Supplementary Note D.

    ``torch_gradient_attribution`` is a transcription of the authors' Fig. S3,
    which uses the double "detach trick" and is valid for arbitrary alpha and
    beta - not only the gradient-identical case of Propositions 1 and 2.  This
    is the load-bearing faithfulness check for the whole package, so it is held
    to machine precision rather than a loose tolerance.
    """
    pytest.importorskip("torch")
    X, Y = make_data(n=40, d=5, seed=3)
    c = factory(X, Y, p, q)
    ours = attribute(X, Y, c, p, q, alpha=alpha, beta=beta)
    ref = torch_gradient_attribution(X, Y, c, p, q, alpha=alpha, beta=beta)
    assert np.abs(ours.R_i - ref.R_i).max() < 1e-12
    assert np.abs(ours.R_kl - ref.R_kl).max() < 1e-12
    assert np.isclose(ours.W, ref.W, rtol=1e-12)


def test_general_alpha_gradient_scaling():
    X, Y = make_data()
    c = exact(X, Y, 2, 2)
    a = attribute(X, Y, c, 2, 2, alpha=3.0, beta=min(4.0, 2.0))
    assert a.conserved


def test_recommend_parameters():
    assert wax.recommend_parameters(2.0, 2.0) == (2.0, 2.0)
    assert wax.recommend_parameters(4.0, 3.0) == (4.0, 3.0)
    assert wax.recommend_parameters(5.0, 4.0) == (5.0, 4.0)


def test_exact_coupling_is_permutation_for_balanced_masses():
    X, Y = make_data(n=40, d=3, seed=2)
    c = exact(X, Y, 2, 2)
    assert c.kind == "exact"
    row = c.gamma.sum(axis=1)
    col = c.gamma.sum(axis=0)
    assert np.allclose(row, np.full(40, 1.0 / 40))
    assert np.allclose(col, np.full(40, 1.0 / 40))
    active = c.gamma > 1e-12
    assert np.all(active.sum(axis=1) == 1)
    assert np.all(active.sum(axis=0) == 1)


def test_sinkhorn_marginals():
    X, Y = make_data(n=30, d=3, seed=3)
    c = sinkhorn(X, Y, 2, 2, reg=0.5)
    assert c.check_marginals(X, Y)
    assert np.all(c.gamma >= 0.0)


def test_uniform_coupling():
    X, Y = make_data(n=20, d=3, seed=4)
    c = uniform(X, Y, 2, 2)
    assert np.allclose(c.gamma, 1.0 / 400)


def test_wasserstein_value():
    X, Y = make_data(n=30, d=2, seed=5)
    c = exact(X, Y, 2, 2)
    W = wasserstein(X, Y, c, 2, 2)
    assert W > 0.0
    assert np.isclose(W**2, np.sum(c.gamma * np.linalg.norm(X[:, None] - Y[None], axis=2) ** 2))


def test_eigen_subspace_closed_form():
    X, Y = make_data(n=50, d=5, seed=6)
    c = exact(X, Y, 2, 2)
    U = eigen_subspace(X, Y, c, [1])
    assert np.allclose(U.T @ U, np.eye(1), atol=1e-8)
    D = X[:, None, :] - Y[None, :, :]
    A = np.einsum("kli,kl,klj->ij", D, c.gamma, D)
    A = 0.5 * (A + A.T)
    from numpy.linalg import eigh

    top = eigh(A)[1][:, -1]
    assert np.isclose(abs(U[:, 0] @ top), 1.0, atol=1e-8)


def test_uwax_search_converges_and_stays_orthonormal():
    X, Y = make_data(n=50, d=5, seed=7)
    c = exact(X, Y, 2, 2)
    U, history = uwax_search(X, Y, c, r=4, dims=[1, 1, 1, 1], seed=0)
    assert np.allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-8)
    assert history[-1] >= history[0] - 1e-9


def test_uwax_attribute_conservation_and_capture():
    X, Y = make_data(n=50, d=5, seed=8)
    c = exact(X, Y, 2, 2)
    full = wasserstein(X, Y, c, 2, 2)
    U, _ = uwax_search(X, Y, c, r=4, dims=[1, 1, 1], seed=0)
    s = uwax_attribute(X, Y, c, U, 4)
    assert np.isclose(s.W2, full, rtol=1e-10)
    assert s.conserved
    assert s.captured <= 1.0 + 1e-9
    assert s.residual >= -1e-9
    for ic in range(len(s.dims)):
        assert np.isclose(s.R_kl_c[ic].sum(), s.R_c[ic], rtol=1e-5)
        assert np.isclose(s.R_i_c[ic].sum(), s.R_c[ic], rtol=1e-5)


def test_full_cover_captures_all():
    X, Y = make_data(n=40, d=4, seed=9)
    c = exact(X, Y, 2, 2)
    U, _ = uwax_search(X, Y, c, r=2, dims=[1, 1, 1, 1], seed=0)
    s = uwax_attribute(X, Y, c, U, 2)
    assert np.isclose(s.captured, 1.0, atol=1e-8)
    assert np.isclose(s.R_i.sum(), s.W2, rtol=1e-6)


def test_srg_wax_beats_meanshift_on_directional_shift():
    # A shift along a single feature with large variance in another: WaX
    # focuses on the transported direction, MeanShift ties both.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    X[:, 0] *= 4.0
    Y = X.copy()
    Y[:, 1] += 2.0
    Y = Y + rng.normal(size=Y.shape) * 0.05
    R_wax = attribute(X, Y, exact(X, Y, 2, 2), 2, 2).R_i
    R_ms = MeanShift()(X, Y, 2, 2)
    s_wax = srg(X, Y, 2, 2, R_wax, exact)
    s_ms = srg(X, Y, 2, 2, R_ms, exact)
    assert s_wax > s_ms


def test_cosine_similarity():
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([0.0, 1.0])
    assert np.isclose(cosine_similarity(a, b), 1.0)
    assert np.isclose(cosine_similarity(a, c), 0.0)


def test_datasets_smoke():
    Xs, ys, Xt, yt, meta = synthetic_domains(60, 12, 4, 3, seed=1)
    assert len(meta["roles"]) == 12
    assert set(meta["roles"]).issubset({"informative", "spurious", "noise"})
    A, B, truth = synthetic_ts(n_days=30, d=4, seed=2)
    assert A.shape == B.shape == (30, 4)
    assert truth.shape == (4,)


def test_abalone_aging_split_smoke():
    X, Y, names = abalone_aging_split(n=200, seed=0)
    assert X.shape == (200, 7) == Y.shape
    assert names == wax.ABALONE_FEATURES
    c = exact(X, Y, 2, 2)
    a = attribute(X, Y, c, 2, 2)
    assert a.conserved


def test_abalone_uwax_aging():
    X, Y, _ = abalone_aging_split(n=180, seed=1)
    c = exact(X, Y, 2, 2)
    a = attribute(X, Y, c, 2, 2)
    assert a.conserved
    U, hist = uwax_search(X, Y, c, r=4, dims=[1, 1, 1], seed=0)
    s = uwax_attribute(X, Y, c, U, 4)
    assert s.conserved
    assert len(hist) > 1

def test_synthetic_ts_depends_on_the_delay():
    """Regression guard: ``dt`` was accepted but never used, so every delay in
    the script 02 sweep silently produced byte-identical numbers."""
    truths = []
    for dt in (1, 2, 4):
        Xs, Xt, truth = synthetic_ts(n_days=120, period=24, d=8, t=7, dt=dt, seed=0)
        assert Xs.shape == Xt.shape == (120, 8)
        truths.append(truth)
    assert not np.allclose(truths[0], truths[1])
    assert not np.allclose(truths[1], truths[2])


def test_wax_recovers_synthetic_transport():
    """Sanity check on the synthetic generator, not a benchmark.

    Source and target are different samples of one periodic process, so the
    coupling is not the identity and the recovered relevance is not trivially
    equal to the ground truth - but WaX should still beat the baselines that
    only see a mean shift or a decision boundary.
    """
    pytest.importorskip("sklearn")
    Xs, Xt, truth = synthetic_ts(n_days=240, period=24, d=8, t=7, dt=2, seed=0)
    R = attribute(Xs, Xt, exact(Xs, Xt, 2, 2), 2, 2).R_i
    c_wax = cosine_similarity(R, truth)
    assert c_wax > 0.7
    assert c_wax > cosine_similarity(MeanShift()(Xs, Xt, 2, 2), truth)
    assert c_wax > cosine_similarity(LogisticBaseline("gradient")(Xs, Xt, 2, 2), truth)


def test_standardize_is_scale_invariant():
    X, Y = make_data(n=40, d=4, seed=11)
    Xa, Ya = standardize(X, Y)
    scaled = np.array([1.0, 1000.0, 0.001, 1.0])
    Xb, Yb = standardize(X * scaled, Y * scaled)
    assert np.allclose(Xa, Xb) and np.allclose(Ya, Yb)


def test_standardize_leaves_constant_columns_alone():
    X = np.ones((5, 2))
    Y = np.ones((5, 2))
    Xs, Ys = standardize(X, Y)
    assert np.all(np.isfinite(Xs)) and np.all(np.isfinite(Ys))

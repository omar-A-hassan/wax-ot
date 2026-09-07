"""Tests for the reusable WaX package."""

import numpy as np
import pytest

import wax
from wax import (
    Coupling,
    attribute,
    cosine_similarity,
    eigen_subspace,
    exact,
    explain,
    sinkhorn,
    srg,
    torch_gradient_attribution,
    uniform,
    uwax_attribute,
    uwax_search,
    wasserstein,
)
from wax.baselines import LogisticBaseline, MeanShift
from wax.datasets import (
    ABALONE_FEATURES,
    abalone_aging_split,
    standardize,
    synthetic_domains,
    synthetic_ts,
)
from wax.forward import pairwise_distance
from wax.wax import (
    _chunk_rows,
    _feature_relevance_beta2,
    _feature_relevance_loop,
    instance_relevance,
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
    assert names == ABALONE_FEATURES
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


def test_coupling_records_its_wasserstein_model():
    X, Y = make_data(n=30, d=3, seed=12)
    for factory in factories():
        c = factory(X, Y, 3, 2)
        assert (c.p, c.q) == (3.0, 2.0)
    assert (Coupling(gamma=np.ones((2, 2)) / 4).p, Coupling(gamma=np.ones((2, 2)) / 4).q) == (
        None,
        None,
    )


def test_attribute_takes_p_and_q_from_the_coupling():
    """The exponents cannot disagree by accident any more.

    Passing them explicitly is still allowed, because the paper explains a
    different model than the coupling solves for the regularized variant.
    """
    X, Y = make_data(n=40, d=4, seed=13)
    c = exact(X, Y, 3, 2)
    assert np.isclose(attribute(X, Y, c).W, attribute(X, Y, c, 3, 2).W, rtol=1e-12)
    # an explicit override still works and still conserves
    other = attribute(X, Y, c, 2, 2)
    assert other.conserved
    assert not np.isclose(other.W, attribute(X, Y, c).W)


def test_attribute_reports_unknown_exponents_for_a_hand_made_coupling():
    X, Y = make_data(n=20, d=3, seed=14)
    c = Coupling(gamma=np.full((20, 20), 1.0 / 400))
    with pytest.raises(ValueError, match="p and q are unknown"):
        attribute(X, Y, c)
    assert attribute(X, Y, c, 2, 2).conserved


def test_explain_equals_the_two_step_sequence():
    X, Y = make_data(n=40, d=4, seed=15)
    short = explain(X, Y)
    long = attribute(X, Y, exact(X, Y, 2, 2), 2, 2)
    assert np.isclose(short.W, long.W, rtol=1e-12)
    assert np.allclose(short.R_i, long.R_i)
    assert short.conserved
    for kind, factory in (("sinkhorn", sinkhorn), ("uniform", uniform)):
        assert np.isclose(
            explain(X, Y, coupling=kind).W, attribute(X, Y, factory(X, Y, 2, 2), 2, 2).W
        )
    with pytest.raises(ValueError, match="unknown coupling"):
        explain(X, Y, coupling="nope")


@pytest.mark.parametrize("q", [1, 2, 3, np.inf])
@pytest.mark.parametrize("offset", [0.0, 1e3, 1e6, 1e9])
def test_beta2_fast_path_equals_the_reference_loop(q, offset):
    """The closed form must equal the direct sum, and must keep conservation.

    Both parameters matter. The beta = 2 denominator is the Euclidean norm for
    every q, so reusing the q-norm here would be wrong for q != 2. And the
    expansion behind the closed form loses all its digits on data with a large
    offset unless the inputs are centred first, which breaks the conservation
    property that the whole method rests on.
    """
    rng = np.random.default_rng(int(offset) % 97 + 1)
    X = offset + rng.normal(0.0, 1.0, size=(50, 30))
    Y = offset + rng.normal(0.0, 1.0, size=(50, 30)) + 0.4
    c = exact(X, Y, 2, q)
    W = wasserstein(X, Y, c, 2, q)
    R_kl = instance_relevance(pairwise_distance(X, Y, q), c.gamma, 2.0, W)

    ref = _feature_relevance_loop(X, Y, R_kl, 2.0)
    fast = _feature_relevance_beta2(X, Y, R_kl)
    scale = max(np.abs(ref).max(), 1e-30)
    assert np.abs(ref - fast).max() / scale < 1e-10
    assert np.isclose(fast.sum(), W, rtol=1e-8)


def test_feature_relevance_dispatches_on_beta():
    X, Y = make_data(n=30, d=8, seed=21)
    c = exact(X, Y, 2, 2)
    W = wasserstein(X, Y, c, 2, 2)
    R_kl = instance_relevance(pairwise_distance(X, Y, 2), c.gamma, 2.0, W)
    # beta = 2 takes the closed form
    assert np.allclose(
        wax.wax.feature_relevance(X, Y, R_kl, 2.0), _feature_relevance_beta2(X, Y, R_kl)
    )
    # every other beta keeps the direct sum
    for beta in (1.0, 3.0, 4.0):
        assert np.allclose(
            wax.wax.feature_relevance(X, Y, R_kl, beta),
            _feature_relevance_loop(X, Y, R_kl, beta),
        )


def test_beta2_fast_path_handles_many_features():
    """Many features end to end, at a shape the direct sum handles badly.

    The difference tensor for these shapes is 200 * 200 * 4000 * 8 = 1.28 GB,
    and the loop holds several of them at once. The closed form needs none.
    """
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 4000))
    Y = rng.normal(size=(200, 4000)) + 0.3
    c = exact(X, Y, 2, 2)
    a = attribute(X, Y, c)
    assert a.conserved
    assert a.R_i.shape == (4000,)


def test_reference_path_resolves_p_and_q_like_attribute():
    """Both entry points must read the model off the coupling.

    If only one of them does, cross-checking a non-default model compares two
    different Wasserstein distances and reports a mismatch that is not there.
    """
    pytest.importorskip("torch")
    X, Y = make_data(n=40, d=5, seed=31)
    c = exact(X, Y, 3, 2)
    a = attribute(X, Y, c)
    r = torch_gradient_attribution(X, Y, c)
    assert np.isclose(a.W, r.W, rtol=1e-12)
    assert np.abs(a.R_i - r.R_i).max() < 1e-12
    with pytest.raises(ValueError, match="p and q are unknown"):
        torch_gradient_attribution(X, Y, Coupling(gamma=c.gamma))


def test_uwax_attribute_honours_multi_dimensional_blocks():
    """Equation (8) allows blocks of different sizes, so dims must come from the
    caller. The old code inferred it by testing consecutive columns of U for
    orthogonality, which holds for every orthonormal U and always gave ones."""
    X, Y = make_data(n=60, d=6, seed=41)
    c = exact(X, Y, 2, 2)
    U, _ = uwax_search(X, Y, c, r=4, dims=[2, 2], seed=0)

    blocks = uwax_attribute(X, Y, c, U, 4, dims=[2, 2])
    singles = uwax_attribute(X, Y, c, U, 4)

    assert blocks.dims == (2, 2)
    assert singles.dims == (1, 1, 1, 1)
    assert blocks.conserved and singles.conserved
    # sum of S_c^2 is invariant to how the span is subdivided, which is why the
    # conservation check alone cannot catch a wrong partition
    assert np.isclose(blocks.R_c.sum(), singles.R_c.sum(), rtol=1e-10)
    assert not np.isclose(blocks.R_c[0], singles.R_c[0])


def test_uwax_attribute_rejects_dims_that_do_not_match_u():
    X, Y = make_data(n=40, d=5, seed=42)
    c = exact(X, Y, 2, 2)
    U, _ = uwax_search(X, Y, c, r=4, dims=[1, 1], seed=0)
    with pytest.raises(ValueError, match="dims sum to"):
        uwax_attribute(X, Y, c, U, 4, dims=[1, 1, 1])


def test_uwax_survives_identical_inputs():
    """W2 is zero, so R_c = S_c^2 / W2 is 0/0 and the tailedness gradient raises
    on Q^(1-r). Both follow attribute() and return zeros."""
    rng = np.random.default_rng(43)
    Z = rng.normal(size=(40, 5))
    c = exact(Z, Z, 2, 2)
    assert wasserstein(Z, Z, c, 2, 2) == 0.0
    U, _ = uwax_search(Z, Z, c, r=4, dims=[1, 1], seed=0)
    assert np.allclose(U.T @ U, np.eye(2), atol=1e-8)
    s = uwax_attribute(Z, Z, c, U, 4, dims=[1, 1])
    assert np.all(np.isfinite(s.R_c)) and np.allclose(s.R_c, 0.0)
    assert np.all(np.isfinite(s.R_i))
    assert attribute(Z, Z, c).conserved


@pytest.mark.parametrize("beta", [1.0, 3.0, 4.0])
def test_block_size_does_not_change_the_direct_sum(beta):
    X, Y = make_data(n=48, d=40, seed=51)
    c = exact(X, Y, 2, 3)
    W = wasserstein(X, Y, c, 2, 3)
    R_kl = instance_relevance(pairwise_distance(X, Y, 3), c.gamma, 2.0, W)
    ref = _feature_relevance_loop(X, Y, R_kl, beta, chunk_rows=len(X))
    for chunk in (None, 1, 7, 1000):
        got = _feature_relevance_loop(X, Y, R_kl, beta, chunk_rows=chunk)
        assert np.abs(ref - got).max() < 1e-12


def test_chunk_rows_only_splits_when_the_block_is_large():
    assert _chunk_rows(60, 60, 5) == 60          # small: one block, as before
    assert _chunk_rows(300, 300, 30) == 300
    assert 0 < _chunk_rows(617, 617, 2326) < 617  # large: split
    assert _chunk_rows(10, 10**6, 10**6) == 1     # never below one row

"""Tests for the graph construction modules (correlation, threshold, build, metrics).

The threshold tests target the property the permutation null exists to have:
it must destroy cross-asset dependence while leaving each asset's marginal
untouched. Both halves are checked directly on data with a known, strong
correlation structure, because a null that quietly preserves some dependence
would still produce a plausible-looking tau -- just a badly inflated one.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from cryptognn.config import Config, GraphConfig, ThresholdConfig
from cryptognn.graph.build import (
    apply_threshold,
    edge_density,
    mantegna_distance,
    mantegna_weights,
    normalized_adjacency,
    validate_adjacency,
    validate_weights,
)
from cryptognn.graph.correlation import correlation_from_windows, rolling_correlation
from cryptognn.graph.metrics import (
    algebraic_connectivity,
    algebraic_connectivity_combinatorial,
    eigs_outside_mp,
    graph_density,
    market_mode_share,
    mean_correlation,
    mst_length,
    spectral_entropy,
)
from cryptognn.graph.threshold import calibrate_tau, check_tau_plausible, permutation_null

N_ASSETS = 6
N_PAIRS = N_ASSETS * (N_ASSETS - 1) // 2


@pytest.fixture
def correlated_returns() -> pd.DataFrame:
    """A (400, 6) return panel with a strong common factor, so every pair is
    correlated at roughly 0.8 -- far enough from zero that a null which failed
    to break the dependence is unmistakable.
    """
    rng = np.random.default_rng(123)
    n_obs = 400
    market = rng.standard_normal((n_obs, 1))
    idiosyncratic = rng.standard_normal((n_obs, N_ASSETS))
    values = 0.02 * (2.0 * market + idiosyncratic)
    index = pd.date_range("2021-01-01", periods=n_obs, freq="D", tz="UTC")
    return pd.DataFrame(values, index=index, columns=[f"A{i}" for i in range(N_ASSETS)])


def _config(**threshold_overrides) -> Config:
    """A minimal Config carrying only what calibrate_tau() reads, so the tests
    do not depend on config/default.yaml staying fixed.
    """
    threshold = ThresholdConfig(
        **{
            "method": "permutation",
            "alpha": 0.05,
            "n_permutations": 200,
            "n_calibration_windows": 5,
            "statistic": "pooled",
            "tau_fixed": 0.30,
            **threshold_overrides,
        }
    )
    graph = GraphConfig(window=60, weight="mantegna", self_loops=True, threshold=threshold)
    return Config(
        data=None, graph=graph, features=None, walkforward=None, model=None, backtest=None, seed=42
    )


def test_correlation_from_windows_matches_numpy(correlated_returns):
    """The shared batched kernel must agree with numpy's reference implementation."""
    values = correlated_returns.to_numpy()
    windows = np.stack([values[0:60].T, values[100:160].T])  # (2, N, W)

    result = correlation_from_windows(windows)

    assert result.shape == (2, N_ASSETS, N_ASSETS)
    np.testing.assert_allclose(result[0], np.corrcoef(values[0:60].T), atol=1e-12)
    np.testing.assert_allclose(result[1], np.corrcoef(values[100:160].T), atol=1e-12)


def test_rolling_correlation_uses_shared_kernel(correlated_returns):
    """Regression guard on the extraction of correlation_from_windows():
    rolling_correlation() must still match numpy window by window.
    """
    corr, corr_index = rolling_correlation(correlated_returns, window=60)
    values = correlated_returns.to_numpy()

    assert corr.shape == (len(correlated_returns) - 59, N_ASSETS, N_ASSETS)
    assert corr_index[0] == correlated_returns.index[59]
    np.testing.assert_allclose(corr[0], np.corrcoef(values[0:60].T), atol=1e-6)
    np.testing.assert_allclose(corr[-1], np.corrcoef(values[-60:].T), atol=1e-6)


def test_permutation_null_shapes(correlated_returns):
    pooled, max_per_permutation = permutation_null(
        correlated_returns.iloc[:60], n_permutations=50, rng=np.random.default_rng(0)
    )

    assert pooled.shape == (50 * N_PAIRS,)
    assert max_per_permutation.shape == (50,)
    assert max_per_permutation.max() <= pooled.max()


def test_permutation_null_preserves_marginals(correlated_returns):
    """Each shuffled column must be a permutation of the original column -- that
    is what keeps the heavy tails of the marginals in the null (thesis Sec. 4.2).

    Checked through the diagonal of the replica correlation matrices: shuffling
    a column cannot change its variance, so the variance implied by the null is
    the observed one. A null that resampled or refitted the marginals instead
    would not have this property.
    """
    window = correlated_returns.iloc[:60]
    rng = np.random.default_rng(0)
    values = window.to_numpy()

    replicas = np.broadcast_to(values, (20, 60, N_ASSETS))
    shuffled = rng.permuted(replicas, axis=1)

    for replica in shuffled:
        np.testing.assert_allclose(np.sort(replica, axis=0), np.sort(values, axis=0), atol=1e-12)


def test_permutation_null_destroys_cross_dependence(correlated_returns):
    """On a panel whose true pairwise correlation is ~0.8, the null must be
    centered at zero. If columns were permuted jointly instead of independently,
    the null would sit near the observed correlation and tau would be inflated.
    """
    window = correlated_returns.iloc[:60]
    observed = np.corrcoef(window.to_numpy().T)[np.triu_indices(N_ASSETS, k=1)]

    pooled, _ = permutation_null(window, n_permutations=500, rng=np.random.default_rng(0))

    assert observed.mean() > 0.5, "fixture should be strongly correlated"
    assert abs(pooled.mean()) < 0.02
    # Sampling sd of rho under the null is ~1/sqrt(T_w - 1) = 0.13 for T_w = 60.
    assert 0.08 < pooled.std() < 0.20


def test_permutation_null_does_not_mutate_input(correlated_returns):
    window = correlated_returns.iloc[:60]
    before = window.to_numpy().copy()

    permutation_null(window, n_permutations=20, rng=np.random.default_rng(0))

    np.testing.assert_array_equal(window.to_numpy(), before)


def test_permutation_null_rejects_bad_input(correlated_returns):
    rng = np.random.default_rng(0)

    with pytest.raises(ValueError, match="2-D"):
        permutation_null(np.zeros(60), n_permutations=10, rng=rng)
    with pytest.raises(ValueError, match="n_permutations"):
        permutation_null(correlated_returns.iloc[:60], n_permutations=0, rng=rng)
    with pytest.raises(ValueError, match="at least 2 assets"):
        permutation_null(correlated_returns.iloc[:60, :1], n_permutations=10, rng=rng)


def test_calibrate_tau_is_deterministic(correlated_returns):
    """Same config, same seed, same tau -- reproducibility of the study threshold."""
    config = _config()

    first = calibrate_tau(correlated_returns, config)
    second = calibrate_tau(correlated_returns, config)

    assert first.tau == second.tau
    assert first.tau_fwer == second.tau_fwer
    assert first.per_window_tau == second.per_window_tau


def test_calibrate_tau_record_is_complete(correlated_returns):
    config = _config()

    calibration = calibrate_tau(correlated_returns, config)

    assert calibration.n_calibration_windows == 5
    assert len(calibration.per_window_tau) == 5
    assert len(calibration.window_end_dates) == 5
    assert calibration.n_pairs == N_PAIRS
    assert calibration.window == 60
    assert calibration.seed == 42
    # Windows must span the whole sample, not cluster at its start.
    assert calibration.window_end_dates[0] == str(correlated_returns.index[59].date())
    assert calibration.window_end_dates[-1] == str(correlated_returns.index[-1].date())
    # JSON-serializable, since it is written to results/metrics/tau_calibration.json.
    json.loads(json.dumps(calibration.to_dict()))


def test_calibrate_tau_ordering_of_thresholds(correlated_returns):
    """tau controls the per-pair rate, tau_fwer the family-wise rate over all
    pairs at once, so the latter is necessarily the more conservative bound.
    """
    calibration = calibrate_tau(correlated_returns, _config())

    assert 0.0 < calibration.tau < calibration.tau_fwer


def test_calibrate_tau_independent_of_observed_correlation(correlated_returns):
    """tau must depend on the null only. Doubling the common factor -- which
    raises every observed correlation -- must leave tau essentially unchanged,
    since the null is built by destroying exactly that structure.
    """
    config = _config()
    rng = np.random.default_rng(7)
    stronger = correlated_returns + 0.05 * rng.standard_normal((len(correlated_returns), 1))

    baseline = calibrate_tau(correlated_returns, config)
    inflated = calibrate_tau(stronger, config)

    assert np.corrcoef(stronger.to_numpy().T)[0, 1] > np.corrcoef(correlated_returns.to_numpy().T)[0, 1]
    assert abs(baseline.tau - inflated.tau) < 0.03


def test_calibrate_tau_rejects_unsupported_settings(correlated_returns):
    with pytest.raises(ValueError, match="threshold method"):
        calibrate_tau(correlated_returns, _config(method="fisher"))
    with pytest.raises(ValueError, match="threshold statistic"):
        calibrate_tau(correlated_returns, _config(statistic="max"))


def test_check_tau_plausible_bounds(correlated_returns):
    calibration = calibrate_tau(correlated_returns, _config())
    check_tau_plausible(calibration)

    # A null that failed to break the dependence would land near the observed
    # correlation; a collapsed one, near zero. Both must be caught.
    with pytest.raises(ValueError, match="plausible range"):
        check_tau_plausible(dataclasses.replace(calibration, tau=0.85))
    with pytest.raises(ValueError, match="plausible range"):
        check_tau_plausible(dataclasses.replace(calibration, tau=0.001))


# --------------------------------------------------------------------------
# Graph construction (S2.2)
# --------------------------------------------------------------------------

TAU = 0.2145  # the value calibrated on the study data, used here as a realistic threshold


@pytest.fixture
def sample_corr() -> np.ndarray:
    """A 4x4 correlation matrix spanning the cases the threshold must separate:
    one strongly anticorrelated pair (-0.5), one weakly positive pair (0.1) that
    still falls below tau, and three pairs comfortably above it.
    """
    return np.array(
        [
            [1.0, 0.8, 0.3, -0.5],
            [0.8, 1.0, 0.6, 0.1],
            [0.3, 0.6, 1.0, 0.9],
            [-0.5, 0.1, 0.9, 1.0],
        ]
    )


def test_mantegna_distance_known_values():
    """d = sqrt(2(1-rho)): 0 at perfect correlation, sqrt(2) at independence,
    2 at perfect anticorrelation.
    """
    corr = np.array([[1.0, 0.0, -1.0]])

    distance = mantegna_distance(corr)

    np.testing.assert_allclose(distance, [[0.0, np.sqrt(2.0), 2.0]], atol=1e-12)


def test_mantegna_distance_clips_out_of_range():
    """float32 storage lets rho drift a hair outside [-1, 1] (validate_correlation
    tolerates 1e-6). Without the clip the square root would take a negative
    argument and return NaN instead of 0.
    """
    corr = np.array([[1.0 + 1e-7, -1.0 - 1e-7]])

    distance = mantegna_distance(corr)

    assert not np.isnan(distance).any()
    np.testing.assert_allclose(distance, [[0.0, 2.0]], atol=1e-12)


def test_mantegna_weights_range_and_zero_diagonal(sample_corr):
    weights = mantegna_weights(sample_corr)

    assert ((weights >= 0.0) & (weights <= 1.0)).all()
    np.testing.assert_allclose(np.diagonal(weights), 0.0, atol=1e-12)
    np.testing.assert_allclose(weights, weights.T, atol=1e-12)
    # 1 - sqrt(2(1-rho))/2 at rho = 0.8 and at rho = -0.5.
    assert weights[0, 1] == pytest.approx(1.0 - np.sqrt(2 * 0.2) / 2)
    assert weights[0, 3] == pytest.approx(1.0 - np.sqrt(2 * 1.5) / 2)


def test_mantegna_weights_monotone_in_correlation():
    """A higher correlation must never produce a lower weight -- otherwise the
    threshold would not be selecting the strongest relationships.
    """
    corr = np.linspace(-1.0, 1.0, 21)
    weights = 1.0 - mantegna_distance(corr) / 2.0

    assert (np.diff(weights) > 0).all()
    assert weights[0] == pytest.approx(0.0)
    assert weights[-1] == pytest.approx(1.0)


def test_build_functions_are_shape_agnostic(sample_corr):
    """The pipeline runs batched over ~1947 windows; a single matrix and the same
    matrix stacked must take the identical code path and agree exactly.
    """
    batched_corr = np.stack([sample_corr, sample_corr * 0.5])

    single_weights = mantegna_weights(sample_corr)
    batched_weights = mantegna_weights(batched_corr)
    np.testing.assert_allclose(batched_weights[0], single_weights, atol=1e-12)

    single_thresh = apply_threshold(sample_corr, single_weights, TAU)
    batched_thresh = apply_threshold(batched_corr, batched_weights, TAU)
    np.testing.assert_allclose(batched_thresh[0], single_thresh, atol=1e-12)

    single_adjacency = normalized_adjacency(single_thresh)
    batched_adjacency = normalized_adjacency(batched_thresh)
    np.testing.assert_allclose(batched_adjacency[0], single_adjacency, atol=1e-12)

    assert edge_density(batched_weights).shape == (2,)
    assert np.isscalar(edge_density(single_weights)) or edge_density(single_weights).ndim == 0


def test_apply_threshold_removes_anticorrelated_pair(sample_corr):
    """Thresholding is on signed rho, so a strongly anticorrelated pair is cut --
    and it is a real edge being cut, not a near-zero one: its Mantegna weight is
    0.134, well above zero.
    """
    weights = mantegna_weights(sample_corr)
    assert weights[0, 3] > 0.13, "the anticorrelated pair should carry real weight"

    thresholded = apply_threshold(sample_corr, weights, TAU)

    assert thresholded[0, 3] == 0.0  # rho = -0.5
    assert thresholded[1, 3] == 0.0  # rho = 0.1, positive but below tau
    assert thresholded[0, 1] > 0.0   # rho = 0.8
    assert thresholded[2, 3] > 0.0   # rho = 0.9


def test_apply_threshold_keeps_original_weight(sample_corr):
    """Surviving edges keep their Mantegna weight unrescaled: the threshold
    decides membership, it does not reweight.
    """
    weights = mantegna_weights(sample_corr)

    thresholded = apply_threshold(sample_corr, weights, TAU)

    survivors = thresholded != 0.0
    np.testing.assert_allclose(thresholded[survivors], weights[survivors], atol=1e-12)
    np.testing.assert_allclose(thresholded, thresholded.T, atol=1e-12)


def test_apply_threshold_boundary_is_exclusive():
    """rho exactly at tau is not evidence against the null, so it is cut; the
    next representable step above it survives.
    """
    corr = np.array([[1.0, TAU], [TAU, 1.0]])
    weights = mantegna_weights(corr)

    assert apply_threshold(corr, weights, TAU)[0, 1] == 0.0

    corr_above = np.array([[1.0, TAU + 1e-9], [TAU + 1e-9, 1.0]])
    assert apply_threshold(corr_above, mantegna_weights(corr_above), TAU)[0, 1] > 0.0


def test_normalized_adjacency_matches_explicit_formula():
    """Check eq:renormalization term by term on a graph small enough to verify by hand."""
    weights = np.array([[0.0, 0.5], [0.5, 0.0]])

    adjacency = normalized_adjacency(weights, self_loops=True)

    a_tilde = weights + np.eye(2)
    degree = np.diag(a_tilde.sum(axis=1) ** -0.5)
    np.testing.assert_allclose(adjacency, degree @ a_tilde @ degree, atol=1e-12)
    np.testing.assert_allclose(adjacency, np.array([[1.0, 0.5], [0.5, 1.0]]) / 1.5, atol=1e-12)


def test_normalized_adjacency_symmetric_with_bounded_spectrum(sample_corr):
    """The property the renormalization trick exists to provide: eigenvalues
    confined to [-1, 1], so stacked layers neither explode nor vanish.
    """
    thresholded = apply_threshold(sample_corr, mantegna_weights(sample_corr), TAU)

    adjacency = normalized_adjacency(thresholded, self_loops=True)

    np.testing.assert_allclose(adjacency, adjacency.T, atol=1e-12)
    eigenvalues = np.linalg.eigvalsh(adjacency)
    assert eigenvalues.min() >= -1.0 - 1e-9
    assert eigenvalues.max() <= 1.0 + 1e-9


def test_normalized_adjacency_self_loops_rescue_isolated_node():
    """62 of the study's 1947 windows contain an isolated node at the calibrated
    tau, so this is the real-data case, not a corner one. With self-loops the
    node's row is a clean basis vector; without them the degree is 0 and the
    normalization must refuse rather than emit inf.
    """
    weights = np.array(
        [
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 0.0],  # isolated
        ]
    )

    adjacency = normalized_adjacency(weights, self_loops=True)

    assert np.isfinite(adjacency).all()
    np.testing.assert_allclose(adjacency[2], [0.0, 0.0, 1.0], atol=1e-12)

    with pytest.raises(ValueError, match="zero degree"):
        normalized_adjacency(weights, self_loops=False)


def test_normalized_adjacency_without_self_loops_is_reachable():
    """self_loops=False is a real branch, not dead code: on a graph with no
    isolated node it must produce the normalization without the identity term.
    """
    weights = np.array([[0.0, 0.5], [0.5, 0.0]])

    adjacency = normalized_adjacency(weights, self_loops=False)

    np.testing.assert_allclose(adjacency, np.array([[0.0, 1.0], [1.0, 0.0]]), atol=1e-12)


def test_edge_density_counts_upper_triangle_only():
    """Density is over the N(N-1)/2 pairs, so the zeroed diagonal must not count
    -- for the study's 15 assets the denominator is 105, not 120.
    """
    weights = np.array(
        [
            [0.0, 0.5, 0.0, 0.0],
            [0.5, 0.0, 0.7, 0.0],
            [0.0, 0.7, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )

    assert edge_density(weights) == pytest.approx(2 / 6)
    assert edge_density(np.zeros((4, 4))) == pytest.approx(0.0)
    complete = np.ones((4, 4)) - np.eye(4)
    assert edge_density(complete) == pytest.approx(1.0)


def test_validate_weights_raises_on_violations(sample_corr):
    weights = mantegna_weights(sample_corr)
    validate_weights(weights)

    asymmetric = weights.copy()
    asymmetric[0, 1] += 0.2
    with pytest.raises(ValueError, match="not symmetric"):
        validate_weights(asymmetric)

    out_of_range = weights.copy()
    out_of_range[0, 1] = out_of_range[1, 0] = 1.5
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        validate_weights(out_of_range)

    with_self_loops = weights.copy()
    np.fill_diagonal(with_self_loops, 1.0)
    with pytest.raises(ValueError, match="diagonal"):
        validate_weights(with_self_loops)

    with_nan = weights.copy()
    with_nan[0, 1] = with_nan[1, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_weights(with_nan)


def test_validate_adjacency_raises_on_violations(sample_corr):
    thresholded = apply_threshold(sample_corr, mantegna_weights(sample_corr), TAU)
    adjacency = normalized_adjacency(thresholded)
    validate_adjacency(adjacency)

    asymmetric = adjacency.copy()
    asymmetric[0, 1] += 0.3
    with pytest.raises(ValueError, match="not symmetric"):
        validate_adjacency(asymmetric)

    non_finite = adjacency.copy()
    non_finite[0, 0] = np.inf
    with pytest.raises(ValueError, match="Non-finite"):
        validate_adjacency(non_finite)

    # Un-normalized weights: symmetric and finite, but the spectrum escapes [-1, 1].
    with pytest.raises(ValueError, match="spectrum outside"):
        validate_adjacency(np.full((4, 4), 0.9))


def test_full_construction_chain_on_rolling_windows(correlated_returns):
    """End-to-end over the real batched shape: returns -> rolling correlation ->
    weights -> threshold -> adjacency, with every invariant asserted, which is
    exactly what scripts/02_build_graphs.py does.
    """
    corr, _ = rolling_correlation(correlated_returns, window=60)

    w_full = mantegna_weights(corr)
    validate_weights(w_full)

    w_thresh = apply_threshold(corr, w_full, TAU)
    validate_weights(w_thresh)

    a_hat = normalized_adjacency(w_thresh, self_loops=True)
    validate_adjacency(a_hat)

    assert a_hat.shape == corr.shape
    # Thresholding can only remove edges, never add them.
    assert (edge_density(w_thresh) <= edge_density(w_full)).all()
    # The fixture is strongly correlated, so almost every edge survives.
    assert edge_density(w_thresh).mean() > 0.9


# --------------------------------------------------------------------------
# Topological metrics (S2.3)
# --------------------------------------------------------------------------


@pytest.fixture
def complete_graph_weights() -> np.ndarray:
    """K_3 with unit weights: both Laplacians have closed-form eigenvalues."""
    return np.ones((3, 3)) - np.eye(3)


def test_mean_correlation_excludes_diagonal(sample_corr):
    """The 1s on the diagonal must not inflate the average co-movement."""
    expected = (0.8 + 0.3 - 0.5 + 0.6 + 0.1 + 0.9) / 6

    assert mean_correlation(sample_corr) == pytest.approx(expected)
    assert mean_correlation(np.eye(4)) == pytest.approx(0.0)


def test_graph_density_delegates_to_edge_density(sample_corr):
    """Density has one implementation in the study: the diagnostics in
    tau_calibration.json and the series in Section 6.6 must be the same number.
    """
    w_thresh = apply_threshold(sample_corr, mantegna_weights(sample_corr), TAU)

    assert graph_density(w_thresh) == edge_density(w_thresh)
    assert graph_density(w_thresh) == pytest.approx(4 / 6)  # 2 of 6 pairs cut


def test_algebraic_connectivity_on_complete_graph(complete_graph_weights):
    """For K_n with unit weights, lambda_2 of L_sym is exactly n/(n-1)."""
    assert algebraic_connectivity(complete_graph_weights) == pytest.approx(3 / 2)


def test_algebraic_connectivity_first_eigenvalue_is_zero(complete_graph_weights):
    """lambda_1 of any Laplacian is 0, which is what makes index 1 the right
    place to read lambda_2 -- but only on a connected graph.
    """
    laplacian = np.eye(3) - normalized_adjacency(complete_graph_weights, self_loops=False)

    assert np.linalg.eigvalsh(laplacian)[0] == pytest.approx(0.0, abs=1e-12)


def test_algebraic_connectivity_zero_when_disconnected():
    """Two components -> lambda_2 = 0. This is precisely why PLANNING computes
    connectivity on W_full and not on the thresholded graph, where 62 of the
    study's windows have an isolated node and this would report a collapse in
    connectivity that is really an artifact of the threshold.
    """
    weights = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )

    assert algebraic_connectivity(weights) == pytest.approx(0.0, abs=1e-12)
    assert algebraic_connectivity_combinatorial(weights) == pytest.approx(0.0, abs=1e-12)


def test_combinatorial_connectivity_on_complete_graph(complete_graph_weights):
    """For K_n with unit weights, lambda_2 of L = D - W is exactly n."""
    assert algebraic_connectivity_combinatorial(complete_graph_weights) == pytest.approx(3.0)


def test_two_connectivities_differ_in_scale_invariance(complete_graph_weights):
    """The reason both are reported: L_sym is invariant to W -> cW, so on a
    complete graph a market-wide rise in correlation is invisible to it, while
    the combinatorial Laplacian scales with it. On this study's data that is the
    difference between cv 0.021 and cv 0.178.
    """
    doubled = complete_graph_weights * 2.0

    assert algebraic_connectivity(doubled) == pytest.approx(algebraic_connectivity(complete_graph_weights))
    assert algebraic_connectivity_combinatorial(doubled) == pytest.approx(
        2 * algebraic_connectivity_combinatorial(complete_graph_weights)
    )


def test_mst_length_on_known_tree():
    """A path 0-1-2-3 of distances 1, 2, 3 is the cheapest spanning tree; the
    direct 5s must be left out. Length is normalized by N-1, i.e. a mean edge.
    """
    distance = np.array(
        [
            [0.0, 1.0, 5.0, 5.0],
            [1.0, 0.0, 2.0, 5.0],
            [5.0, 2.0, 0.0, 3.0],
            [5.0, 5.0, 3.0, 0.0],
        ]
    )

    assert mst_length(distance) == pytest.approx((1.0 + 2.0 + 3.0) / 3)


def test_mst_length_shortens_as_correlation_rises():
    """The metric of onnela2003dynamics must move the way the thesis reads it:
    stronger correlation -> shorter distances -> more compact tree.
    """
    weak = np.eye(4) * 0.5 + 0.5 * np.ones((4, 4)) * 0.2
    np.fill_diagonal(weak, 1.0)
    strong = np.full((4, 4), 0.9)
    np.fill_diagonal(strong, 1.0)

    assert mst_length(mantegna_distance(strong)) < mst_length(mantegna_distance(weak))


def test_spectral_entropy_bounds():
    """Maximal at log(N) when C = I (variance spread evenly), 0 when a single
    mode explains everything. The rank-1 case also exercises the x log x -> 0
    guard, since three eigenvalues are exactly zero.
    """
    assert spectral_entropy(np.eye(4)) == pytest.approx(np.log(4))

    rank_one = np.ones((4, 4))
    entropy = spectral_entropy(rank_one)
    assert np.isfinite(entropy)
    assert entropy == pytest.approx(0.0, abs=1e-9)


def test_market_mode_share_bounds():
    """1/N when no common structure, 1 when every asset moves identically."""
    assert market_mode_share(np.eye(4)) == pytest.approx(0.25)
    assert market_mode_share(np.ones((4, 4))) == pytest.approx(1.0)


def test_eigs_outside_mp_counts_only_beyond_the_edge():
    """With q = 0.25 the bulk ends at (1 + 0.5)^2 = 2.25: an identity matrix has
    every eigenvalue at 1 and nothing escapes; a single dominant mode does.
    """
    assert eigs_outside_mp(np.eye(4), q=0.25) == 0
    assert eigs_outside_mp(np.ones((4, 4)), q=0.25) == 1  # eigenvalues [0, 0, 0, 4]


def test_eigs_outside_mp_edge_depends_on_q():
    """q is the caller's to supply: the same matrix crosses the edge or not
    depending on the window that produced it.
    """
    corr = np.ones((4, 4))

    assert eigs_outside_mp(corr, q=0.25) == 1  # edge 2.25 < 4
    assert eigs_outside_mp(corr, q=1.0) == 0   # edge 4.00, not exceeded


def test_metrics_are_shape_agnostic(sample_corr, complete_graph_weights):
    """Same code path for one matrix and for a stack of windows."""
    batched_corr = np.stack([sample_corr, np.eye(4)])
    batched_weights = np.stack([complete_graph_weights, complete_graph_weights * 2])

    np.testing.assert_allclose(mean_correlation(batched_corr)[0], mean_correlation(sample_corr))
    np.testing.assert_allclose(spectral_entropy(batched_corr)[1], np.log(4))
    np.testing.assert_allclose(market_mode_share(batched_corr)[1], 0.25)
    np.testing.assert_allclose(eigs_outside_mp(batched_corr, q=0.25), [0, 0])
    np.testing.assert_allclose(
        algebraic_connectivity(batched_weights), [1.5, 1.5]
    )  # scale-invariant, so both are n/(n-1)
    np.testing.assert_allclose(algebraic_connectivity_combinatorial(batched_weights), [3.0, 6.0])
    np.testing.assert_allclose(
        mst_length(mantegna_distance(batched_corr))[0], mst_length(mantegna_distance(sample_corr))
    )


def test_all_metrics_on_rolling_windows(correlated_returns):
    """The full S2.3 computation over the real batched shape, with every
    theoretical bound asserted -- what scripts/03_topology_analysis.py does.
    """
    window = 60
    corr, corr_index = rolling_correlation(correlated_returns, window)
    w_full = mantegna_weights(corr)
    w_thresh = apply_threshold(corr, w_full, TAU)
    q = N_ASSETS / window

    metrics = {
        "mean_correlation": mean_correlation(corr),
        "graph_density": graph_density(w_thresh),
        "algebraic_connectivity": algebraic_connectivity(w_full),
        "algebraic_connectivity_combinatorial": algebraic_connectivity_combinatorial(w_full),
        "mst_length": mst_length(mantegna_distance(corr)),
        "spectral_entropy": spectral_entropy(corr),
        "market_mode_share": market_mode_share(corr),
        "eigs_outside_mp": eigs_outside_mp(corr, q),
    }

    for name, series in metrics.items():
        assert len(series) == len(corr_index), name
        assert np.isfinite(series).all(), name

    assert ((metrics["graph_density"] >= 0) & (metrics["graph_density"] <= 1)).all()
    assert ((metrics["spectral_entropy"] >= 0) & (metrics["spectral_entropy"] <= np.log(N_ASSETS))).all()
    assert ((metrics["market_mode_share"] >= 1 / N_ASSETS) & (metrics["market_mode_share"] <= 1)).all()
    assert (metrics["algebraic_connectivity"] >= 0).all()
    assert (metrics["mst_length"] > 0).all()
    assert (metrics["eigs_outside_mp"] >= 0).all()

    # The fixture is driven by one common factor, so the market mode must
    # dominate and the spectrum must be far from uniform.
    assert metrics["market_mode_share"].mean() > 0.5
    assert metrics["spectral_entropy"].mean() < np.log(N_ASSETS)

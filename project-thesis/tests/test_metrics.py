"""Tests for cryptognn.graph.metrics.

Each metric is pinned to a value with a closed form -- lambda_2 of K_n, the
entropy of the identity, the market mode share of a rank-one matrix -- rather
than to whatever the implementation currently returns. A topological metric
that silently drifts produces a plot that still looks like a result.
"""
from __future__ import annotations

import numpy as np
import pytest
from conftest import N_ASSETS, TAU

from cryptognn.graph.build import (
    apply_threshold,
    edge_density,
    mantegna_distance,
    mantegna_weights,
    normalized_adjacency,
)
from cryptognn.graph.correlation import rolling_correlation
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


@pytest.fixture
def complete_graph_weights() -> np.ndarray:
    """K_3 with unit weights: both Laplacians have closed-form eigenvalues."""
    return np.ones((3, 3)) - np.eye(3)


class TestMeanCorrelation:
    def test_excludes_diagonal(self, sample_corr):
        """The 1s on the diagonal must not inflate the average co-movement."""
        expected = (0.8 + 0.3 - 0.5 + 0.6 + 0.1 + 0.9) / 6

        assert mean_correlation(sample_corr) == pytest.approx(expected)
        assert mean_correlation(np.eye(4)) == pytest.approx(0.0)


class TestGraphDensity:
    def test_delegates_to_edge_density(self, sample_corr):
        """Density has one implementation in the study: the diagnostics in
        tau_calibration.json and the series in Section 6.6 must be the same number.
        """
        w_thresh = apply_threshold(sample_corr, mantegna_weights(sample_corr), TAU)

        assert graph_density(w_thresh) == edge_density(w_thresh)
        assert graph_density(w_thresh) == pytest.approx(4 / 6)  # 2 of 6 pairs cut


class TestAlgebraicConnectivity:
    def test_complete_graph(self, complete_graph_weights):
        """For K_n with unit weights, lambda_2 of L_sym is exactly n/(n-1)."""
        assert algebraic_connectivity(complete_graph_weights) == pytest.approx(3 / 2)

    def test_first_eigenvalue_is_zero(self, complete_graph_weights):
        """lambda_1 of any Laplacian is 0, which is what makes index 1 the right
        place to read lambda_2 -- but only on a connected graph.
        """
        laplacian = np.eye(3) - normalized_adjacency(complete_graph_weights, self_loops=False)

        assert np.linalg.eigvalsh(laplacian)[0] == pytest.approx(0.0, abs=1e-12)

    def test_zero_when_disconnected(self):
        """Two components -> lambda_2 = 0. This is precisely why connectivity is
        computed on the complete graph and not on the thresholded one, where 62
        of the study's windows have an isolated node and this would report a
        collapse in connectivity that is really an artifact of the threshold.
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

    def test_combinatorial_on_complete_graph(self, complete_graph_weights):
        """For K_n with unit weights, lambda_2 of L = D - W is exactly n."""
        assert algebraic_connectivity_combinatorial(complete_graph_weights) == pytest.approx(3.0)

    def test_variants_differ_in_scale_invariance(self, complete_graph_weights):
        """The reason both are reported: L_sym is invariant to W -> cW, so on a
        complete graph a market-wide rise in correlation is invisible to it,
        while the combinatorial Laplacian scales with it. On this study's data
        that is the difference between cv 0.021 and cv 0.178.
        """
        doubled = complete_graph_weights * 2.0

        assert algebraic_connectivity(doubled) == pytest.approx(
            algebraic_connectivity(complete_graph_weights)
        )
        assert algebraic_connectivity_combinatorial(doubled) == pytest.approx(
            2 * algebraic_connectivity_combinatorial(complete_graph_weights)
        )


class TestMstLength:
    def test_known_tree(self):
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

    def test_shortens_as_correlation_rises(self):
        """The metric of onnela2003dynamics must move the way the thesis reads
        it: stronger correlation -> shorter distances -> more compact tree.
        """
        weak = np.eye(4) * 0.5 + 0.5 * np.ones((4, 4)) * 0.2
        np.fill_diagonal(weak, 1.0)
        strong = np.full((4, 4), 0.9)
        np.fill_diagonal(strong, 1.0)

        assert mst_length(mantegna_distance(strong)) < mst_length(mantegna_distance(weak))


class TestSpectralEntropy:
    def test_bounds(self):
        """Maximal at log(N) when C = I (variance spread evenly), 0 when a single
        mode explains everything. The rank-1 case also exercises the x log x -> 0
        guard, since three eigenvalues are exactly zero.
        """
        assert spectral_entropy(np.eye(4)) == pytest.approx(np.log(4))

        rank_one = np.ones((4, 4))
        entropy = spectral_entropy(rank_one)
        assert np.isfinite(entropy)
        assert entropy == pytest.approx(0.0, abs=1e-9)


class TestMarketModeShare:
    def test_bounds(self):
        """1/N when no common structure, 1 when every asset moves identically."""
        assert market_mode_share(np.eye(4)) == pytest.approx(0.25)
        assert market_mode_share(np.ones((4, 4))) == pytest.approx(1.0)


class TestEigsOutsideMp:
    def test_counts_only_beyond_the_edge(self):
        """With q = 0.25 the bulk ends at (1 + 0.5)^2 = 2.25: an identity matrix
        has every eigenvalue at 1 and nothing escapes; a dominant mode does.
        """
        assert eigs_outside_mp(np.eye(4), q=0.25) == 0
        assert eigs_outside_mp(np.ones((4, 4)), q=0.25) == 1  # eigenvalues [0, 0, 0, 4]

    def test_edge_depends_on_q(self):
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
    assert (
        (metrics["spectral_entropy"] >= 0) & (metrics["spectral_entropy"] <= np.log(N_ASSETS))
    ).all()
    assert (
        (metrics["market_mode_share"] >= 1 / N_ASSETS) & (metrics["market_mode_share"] <= 1)
    ).all()
    assert (metrics["algebraic_connectivity"] >= 0).all()
    assert (metrics["mst_length"] > 0).all()
    assert (metrics["eigs_outside_mp"] >= 0).all()

    # The fixture is driven by one common factor, so the market mode must
    # dominate and the spectrum must be far from uniform.
    assert metrics["market_mode_share"].mean() > 0.5
    assert metrics["spectral_entropy"].mean() < np.log(N_ASSETS)

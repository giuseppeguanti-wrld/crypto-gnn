"""Tests for cryptognn.graph.build.

Each transform in the chain C -> distance -> weights -> threshold -> adjacency
is checked against a value computable by hand, not merely for plausible shape:
these are the definitions the whole study rests on, and an error here would
propagate into every metric and every figure without ever raising.
"""
from __future__ import annotations

import numpy as np
import pytest
from synthetic import TAU

from cryptognn.graph.build import (
    apply_threshold,
    edge_density,
    mantegna_distance,
    mantegna_weights,
    normalized_adjacency,
    validate_adjacency,
    validate_weights,
)
from cryptognn.graph.correlation import rolling_correlation


class TestMantegnaDistance:
    def test_known_values(self):
        """d = sqrt(2(1-rho)): 0 at perfect correlation, sqrt(2) at independence,
        2 at perfect anticorrelation.
        """
        corr = np.array([[1.0, 0.0, -1.0]])

        distance = mantegna_distance(corr)

        np.testing.assert_allclose(distance, [[0.0, np.sqrt(2.0), 2.0]], atol=1e-12)

    def test_clips_out_of_range(self):
        """float32 storage lets rho drift a hair outside [-1, 1]
        (validate_correlation tolerates 1e-6). Without the clip the square root
        would take a negative argument and return NaN instead of 0.
        """
        corr = np.array([[1.0 + 1e-7, -1.0 - 1e-7]])

        distance = mantegna_distance(corr)

        assert not np.isnan(distance).any()
        np.testing.assert_allclose(distance, [[0.0, 2.0]], atol=1e-12)


class TestMantegnaWeights:
    def test_range_and_zero_diagonal(self, sample_corr):
        weights = mantegna_weights(sample_corr)

        assert ((weights >= 0.0) & (weights <= 1.0)).all()
        np.testing.assert_allclose(np.diagonal(weights), 0.0, atol=1e-12)
        np.testing.assert_allclose(weights, weights.T, atol=1e-12)
        # 1 - sqrt(2(1-rho))/2 at rho = 0.8 and at rho = -0.5.
        assert weights[0, 1] == pytest.approx(1.0 - np.sqrt(2 * 0.2) / 2)
        assert weights[0, 3] == pytest.approx(1.0 - np.sqrt(2 * 1.5) / 2)

    def test_monotone_in_correlation(self):
        """A higher correlation must never produce a lower weight -- otherwise
        the threshold would not be selecting the strongest relationships.
        """
        corr = np.linspace(-1.0, 1.0, 21)
        weights = 1.0 - mantegna_distance(corr) / 2.0

        assert (np.diff(weights) > 0).all()
        assert weights[0] == pytest.approx(0.0)
        assert weights[-1] == pytest.approx(1.0)


class TestApplyThreshold:
    def test_removes_anticorrelated_pair(self, sample_corr):
        """Thresholding is on signed rho, so a strongly anticorrelated pair is
        cut -- and it is a real edge being cut, not a near-zero one: its
        Mantegna weight is 0.134, well above zero.
        """
        weights = mantegna_weights(sample_corr)
        assert weights[0, 3] > 0.13, "the anticorrelated pair should carry real weight"

        thresholded = apply_threshold(sample_corr, weights, TAU)

        assert thresholded[0, 3] == 0.0  # rho = -0.5
        assert thresholded[1, 3] == 0.0  # rho = 0.1, positive but below tau
        assert thresholded[0, 1] > 0.0   # rho = 0.8
        assert thresholded[2, 3] > 0.0   # rho = 0.9

    def test_keeps_original_weight(self, sample_corr):
        """Surviving edges keep their Mantegna weight unrescaled: the threshold
        decides membership, it does not reweight.
        """
        weights = mantegna_weights(sample_corr)

        thresholded = apply_threshold(sample_corr, weights, TAU)

        survivors = thresholded != 0.0
        np.testing.assert_allclose(thresholded[survivors], weights[survivors], atol=1e-12)
        np.testing.assert_allclose(thresholded, thresholded.T, atol=1e-12)

    def test_boundary_is_exclusive(self):
        """rho exactly at tau is not evidence against the null, so it is cut; the
        next representable step above it survives.
        """
        corr = np.array([[1.0, TAU], [TAU, 1.0]])
        weights = mantegna_weights(corr)

        assert apply_threshold(corr, weights, TAU)[0, 1] == 0.0

        corr_above = np.array([[1.0, TAU + 1e-9], [TAU + 1e-9, 1.0]])
        assert apply_threshold(corr_above, mantegna_weights(corr_above), TAU)[0, 1] > 0.0


class TestNormalizedAdjacency:
    def test_matches_explicit_formula(self):
        """Check eq:renormalization term by term on a graph small enough to
        verify by hand.
        """
        weights = np.array([[0.0, 0.5], [0.5, 0.0]])

        adjacency = normalized_adjacency(weights, self_loops=True)

        a_tilde = weights + np.eye(2)
        degree = np.diag(a_tilde.sum(axis=1) ** -0.5)
        np.testing.assert_allclose(adjacency, degree @ a_tilde @ degree, atol=1e-12)
        np.testing.assert_allclose(adjacency, np.array([[1.0, 0.5], [0.5, 1.0]]) / 1.5, atol=1e-12)

    def test_symmetric_with_bounded_spectrum(self, sample_corr):
        """The property the renormalization trick exists to provide: eigenvalues
        confined to [-1, 1], so stacked layers neither explode nor vanish.
        """
        thresholded = apply_threshold(sample_corr, mantegna_weights(sample_corr), TAU)

        adjacency = normalized_adjacency(thresholded, self_loops=True)

        np.testing.assert_allclose(adjacency, adjacency.T, atol=1e-12)
        eigenvalues = np.linalg.eigvalsh(adjacency)
        assert eigenvalues.min() >= -1.0 - 1e-9
        assert eigenvalues.max() <= 1.0 + 1e-9

    def test_self_loops_rescue_isolated_node(self):
        """62 of the study's 1947 windows contain an isolated node at the
        calibrated tau, so this is the real-data case, not a corner one. With
        self-loops the node's row is a clean basis vector; without them the
        degree is 0 and the normalization must refuse rather than emit inf.
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

    def test_without_self_loops_is_reachable(self):
        """self_loops=False is a real branch, not dead code: on a graph with no
        isolated node it must produce the normalization without the identity term.
        """
        weights = np.array([[0.0, 0.5], [0.5, 0.0]])

        adjacency = normalized_adjacency(weights, self_loops=False)

        np.testing.assert_allclose(adjacency, np.array([[0.0, 1.0], [1.0, 0.0]]), atol=1e-12)


class TestEdgeDensity:
    def test_counts_upper_triangle_only(self):
        """Density is over the N(N-1)/2 pairs, so the zeroed diagonal must not
        count -- for the study's 15 assets the denominator is 105, not 120.
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


class TestValidateWeights:
    def test_raises_on_violations(self, sample_corr):
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


class TestValidateAdjacency:
    def test_raises_on_violations(self, sample_corr):
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


def test_functions_are_shape_agnostic(sample_corr):
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


def test_full_construction_chain(correlated_returns):
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

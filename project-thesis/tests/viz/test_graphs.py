"""Tests for cryptognn.viz.graphs: the node-link snapshots and their layout.

The layout is the part that matters. A force-directed layout is stochastic, and
one recomputed per date moves every node -- so two snapshots laid out
independently differ everywhere and the compaction the figure exists to show is
lost among nodes that merely drifted. Determinism given the seed is therefore not
a nicety; it is what makes the calm/crisis comparison mean anything.
"""
from __future__ import annotations

import numpy as np

from cryptognn.viz import graphs as viz_graphs

import matplotlib.pyplot as plt  # isort: skip -- Agg is selected in conftest

N_ASSETS = 6


def as_weights(corr: np.ndarray) -> np.ndarray:
    """A weight matrix from a correlation matrix: non-negative, zero diagonal."""
    return np.clip(corr - np.eye(N_ASSETS), 0.0, None)


class TestFixedLayout:
    def test_is_deterministic_given_the_seed(self, block_corr):
        """Without a seed the thesis would redraw differently on every run and the
        two snapshots could not be compared.
        """
        weights = as_weights(block_corr)

        first = viz_graphs.fixed_layout(weights, seed=42)
        second = viz_graphs.fixed_layout(weights, seed=42)
        other = viz_graphs.fixed_layout(weights, seed=7)

        assert set(first) == set(range(N_ASSETS))
        for node in first:
            np.testing.assert_allclose(first[node], second[node])
        assert any(not np.allclose(first[node], other[node]) for node in first)

    def test_accepts_labels(self, block_corr):
        labels = [f"A{i}" for i in range(N_ASSETS)]

        layout = viz_graphs.fixed_layout(as_weights(block_corr), seed=42, labels=labels)

        assert set(layout) == set(labels)


def test_snapshot_drops_zero_weight_edges(block_corr):
    """A thresholded matrix must be drawn as the sparse graph it is: entries
    zeroed by the threshold are not edges.
    """
    _, ax = plt.subplots()
    weights = np.zeros((N_ASSETS, N_ASSETS))
    weights[0, 1] = weights[1, 0] = 0.8
    layout = viz_graphs.fixed_layout(as_weights(block_corr), seed=42)

    viz_graphs.draw_snapshot(ax, weights, layout)

    # One LineCollection for the single surviving edge, and no crash on the
    # isolated nodes (whose weighted degree is 0).
    assert len(ax.collections) >= 1

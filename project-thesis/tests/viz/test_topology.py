"""Tests for cryptognn.viz.topology: orderings, heatmaps and the MP density.

These do not assert on how a panel looks -- that is what reading the PDF is for.
They assert on the properties that make the panels *comparable* to one another:
one asset ordering reused everywhere, a colour scale pinned regardless of the
data, and a Marchenko-Pastur curve that is the closed form it claims to be.

Each of those is load-bearing and each fails invisibly. A heatmap rescaled per
date still renders beautifully and no longer distinguishes a calm market from a
crisis, which is the only thing those panels exist to do.
"""
from __future__ import annotations

import numpy as np
import pytest

from cryptognn.events import Event
from cryptognn.viz import topology as viz_topology

import matplotlib.pyplot as plt  # isort: skip -- Agg is selected in conftest
import pandas as pd  # isort: skip

N_ASSETS = 6


class TestHierarchicalOrder:
    def test_is_a_valid_permutation(self, block_corr):
        order = viz_topology.hierarchical_order(block_corr)

        assert sorted(order.tolist()) == list(range(N_ASSETS))

    def test_groups_blocks(self, block_corr):
        """The ordering must put correlated assets adjacent -- otherwise the block
        structure of the market never appears on the diagonal of the heatmap.
        """
        order = viz_topology.hierarchical_order(block_corr).tolist()

        first_block_positions = sorted(order.index(i) for i in (0, 1, 2))
        assert first_block_positions in ([0, 1, 2], [3, 4, 5]), order

    def test_is_deterministic(self, block_corr):
        """Reused across all three heatmap panels, so it must not vary per call."""
        assert np.array_equal(
            viz_topology.hierarchical_order(block_corr), viz_topology.hierarchical_order(block_corr)
        )


class TestHeatmap:
    def test_color_scale_is_pinned(self, block_corr):
        """Fixed at [-1, 1] regardless of the data: a per-date rescaling would make
        a calm market and a crisis look identical, which is the one thing these
        panels exist to distinguish.
        """
        _, (ax_a, ax_b) = plt.subplots(1, 2)
        order = np.arange(N_ASSETS)

        image_a = viz_topology.draw_heatmap(ax_a, block_corr, order)
        image_b = viz_topology.draw_heatmap(ax_b, block_corr * 0.2, order)

        assert image_a.get_clim() == (-1.0, 1.0)
        assert image_b.get_clim() == (-1.0, 1.0)

    def test_applies_the_order(self, block_corr):
        """The reordering must permute both axes symmetrically, or a cell would no
        longer name the pair it shows.
        """
        _, ax = plt.subplots()
        order = np.array([5, 4, 3, 2, 1, 0])

        image = viz_topology.draw_heatmap(ax, block_corr, order)

        np.testing.assert_allclose(image.get_array(), block_corr[np.ix_(order, order)])


class TestMarchenkoPasturDensity:
    @pytest.mark.parametrize("q", [0.25, 0.5, 1.0])
    def test_integrates_to_one(self, q):
        """Closed-form check of the formula: a probability density over its support."""
        lower, upper = (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2
        grid = np.linspace(lower, upper, 200_001)

        density = viz_topology.marchenko_pastur_density(grid, q)

        assert np.trapezoid(density, grid) == pytest.approx(1.0, abs=5e-3)

    def test_support_is_exactly_the_bulk(self):
        """Zero outside [(1-sqrt q)^2, (1+sqrt q)^2], positive inside, never NaN."""
        q = 0.25
        lower, upper = (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2
        outside = np.array([0.0, lower - 0.01, upper + 0.01, 10.0])
        inside = np.linspace(lower + 1e-3, upper - 1e-3, 50)

        assert np.all(viz_topology.marchenko_pastur_density(outside, q) == 0.0)
        density_inside = viz_topology.marchenko_pastur_density(inside, q)
        assert np.all(density_inside > 0.0)
        assert np.isfinite(density_inside).all()


def test_metric_series_with_events_and_multiple_metrics(topology_frame):
    """The density panel of the central figure: two series plus event markers."""
    _, ax = plt.subplots()
    events = [
        Event(key="e1", date=pd.Timestamp("2021-03-01").date(), label="Primo"),
        Event(key="e2", date=pd.Timestamp("2021-08-01").date(), label="Secondo"),
    ]

    viz_topology.draw_metric_series(
        ax,
        topology_frame,
        ["graph_density", "graph_density_fwer"],
        events=events,
        labels=["tau", "tau FWER"],
        ylabel="Densità",
    )

    assert len(ax.lines) == 2 + len(events)  # two series plus one axvline each
    assert ax.get_ylabel() == "Densità"
    assert ax.get_legend() is not None

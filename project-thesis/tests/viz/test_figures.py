"""Tests for figure composition (cryptognn.viz.figures).

These verify the arrangement properties the figures depend on for their meaning
-- a shared node layout, a shared colour scale, a shared time axis. Each is
load-bearing and each fails silently: a snapshot pair laid out independently
still renders a handsome picture, it just no longer shows what it claims to.

None of this was reachable before the composition functions moved out of
scripts/06_make_figures.py, whose name cannot be imported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptognn.events import Event
from cryptognn.viz import figures
from cryptognn.viz.style import FIGURE_WIDTH
from cryptognn.viz.topology import hierarchical_order

import matplotlib.pyplot as plt  # isort: skip -- Agg is selected in conftest

N_ASSETS = 5
N_WINDOWS = 400
SYMBOLS = [f"A{i}" for i in range(N_ASSETS)]


@pytest.fixture
def corr_index() -> pd.DatetimeIndex:
    return pd.date_range("2021-01-01", periods=N_WINDOWS, freq="D", name="date")


@pytest.fixture
def corr(corr_index) -> np.ndarray:
    """Symmetric correlation tensor with a unit diagonal and varying strength."""
    rng = np.random.default_rng(0)
    base = rng.uniform(0.1, 0.9, (N_WINDOWS, N_ASSETS, N_ASSETS))
    tensor = (base + base.transpose(0, 2, 1)) / 2
    np.einsum("kii->ki", tensor)[:] = 1.0
    return tensor


@pytest.fixture
def weights(corr) -> np.ndarray:
    result = 1.0 - np.sqrt(2 * (1 - corr)) / 2
    np.einsum("kii->ki", result)[:] = 0.0
    return result


@pytest.fixture
def topology(corr_index) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "mean_correlation": rng.uniform(0.3, 0.9, N_WINDOWS),
            "graph_density": rng.uniform(0.9, 1.0, N_WINDOWS),
            "graph_density_fwer": rng.uniform(0.4, 1.0, N_WINDOWS),
            "algebraic_connectivity_combinatorial": rng.uniform(4, 10, N_WINDOWS),
            "mst_length": rng.uniform(0.3, 0.9, N_WINDOWS),
        },
        index=corr_index,
    )


@pytest.fixture
def events() -> list[Event]:
    return [
        Event(key="china_crackdown", date=pd.Timestamp("2021-05-19").date(), label="Stretta cinese"),
        Event(key="terra_luna", date=pd.Timestamp("2021-08-09").date(), label="Terra/Luna"),
        Event(key="ftx", date=pd.Timestamp("2021-11-08").date(), label="FTX"),
    ]


# --------------------------------------------------------------------------
# Reference dates
# --------------------------------------------------------------------------


class TestSelectReferenceDates:
    def test_calm_is_the_least_correlated_window(self, topology, events):
        dates = figures.select_reference_dates(topology, events, ("ftx",))

        assert dates["Calmo"] == topology["mean_correlation"].idxmin()

    def test_crisis_is_the_first_fully_post_event_window(self, topology, events):
        """Offset +60: at the event date the 60-day window is 59/60 pre-event,
        so reading it there would understate the shock.
        """
        dates = figures.select_reference_dates(topology, events, ("ftx",))

        expected = pd.Timestamp("2021-11-08") + pd.Timedelta(days=figures.POST_EVENT_OFFSET)
        assert dates["FTX +60g"] == expected

    def test_order_and_labels(self, topology, events):
        dates = figures.select_reference_dates(topology, events, ("terra_luna", "ftx"))

        assert list(dates) == ["Calmo", "Terra/Luna +60g", "FTX +60g"]

    def test_unknown_key_lists_the_available_ones(self, topology, events):
        """The keys come from config/events.yaml; renaming one there must produce
        a message that says what to use, not a bare KeyError.
        """
        with pytest.raises(ValueError) as error:
            figures.select_reference_dates(topology, events, ("luna_terra",))

        message = str(error.value)
        assert "luna_terra" in message
        assert "ftx" in message and "china_crackdown" in message

    def test_offset_is_configurable(self, topology, events):
        dates = figures.select_reference_dates(topology, events, ("ftx",), offset=30)

        assert dates["FTX +30g"] == pd.Timestamp("2021-11-08") + pd.Timedelta(days=30)


# --------------------------------------------------------------------------
# The four compositions
# --------------------------------------------------------------------------


def _all_figures(corr, corr_index, weights, topology, events):
    order = hierarchical_order(corr.mean(axis=0))
    dates = figures.select_reference_dates(topology, events, ("terra_luna", "ftx"))
    pair = figures.select_reference_dates(topology, events, ("china_crackdown",))
    return {
        "timeseries": figures.figure_topology_timeseries(topology, events),
        "heatmaps": figures.figure_correlation_heatmaps(corr, corr_index, dates, SYMBOLS, order),
        "snapshots": figures.figure_graph_snapshots(
            weights, weights, corr_index, pair, SYMBOLS, seed=42
        ),
        "spectrum": figures.figure_mp_spectrum(corr, topology, q=N_ASSETS / 60),
    }


def test_every_composition_returns_a_figure(corr, corr_index, weights, topology, events):
    for name, fig in _all_figures(corr, corr_index, weights, topology, events).items():
        assert isinstance(fig, plt.Figure), name
        assert fig.get_figwidth() == pytest.approx(FIGURE_WIDTH), name


def test_compositions_write_nothing(corr, corr_index, weights, topology, events, tmp_path, monkeypatch):
    """Composition returns a Figure; saving belongs to the script. Checked by
    running every composition with the working directory inside an empty tmp_path
    and asserting it stays empty -- the behavioural counterpart to the AST guard
    in test_viz.py.
    """
    monkeypatch.chdir(tmp_path)

    _all_figures(corr, corr_index, weights, topology, events)

    assert list(tmp_path.iterdir()) == []


class TestTopologyTimeseries:
    def test_has_four_panels_sharing_the_x_axis(self, topology, events):
        """The shared axis is the point of the figure: a vertical line at a
        crisis date must be readable across all four metrics at once.
        """
        fig = figures.figure_topology_timeseries(topology, events)

        axes = fig.get_axes()
        assert len(axes) == 4
        for ax in axes[1:]:
            assert ax.get_shared_x_axes().joined(ax, axes[0])

    def test_marks_every_event(self, topology, events):
        fig = figures.figure_topology_timeseries(topology, events)

        top_panel = fig.get_axes()[0]
        # axvline keeps the Timestamp it was given rather than a matplotlib
        # date number, so the marker positions read back directly.
        marked = {
            pd.Timestamp(line.get_xdata()[0]).date()
            for line in top_panel.lines
            if len(set(line.get_xdata())) == 1  # a vertical rule
        }
        assert {event.date for event in events} <= marked

    def test_density_panel_shows_both_thresholds(self, topology, events):
        """The saturation of the calibrated threshold is only visible next to
        the FWER one; a single flat line would look like a broken metric.
        """
        fig = figures.figure_topology_timeseries(topology, events)

        density_panel = fig.get_axes()[1]
        series = [line for line in density_panel.lines if len(set(line.get_xdata())) > 1]
        assert len(series) == 2
        assert density_panel.get_legend() is not None


class TestCorrelationHeatmaps:
    def test_panels_share_the_colour_scale(self, corr, corr_index, topology, events):
        """A per-date rescaling would give a calm market and a crisis the same
        saturated colours, which is exactly the comparison the panels exist for.
        """
        order = hierarchical_order(corr.mean(axis=0))
        dates = figures.select_reference_dates(topology, events, ("terra_luna", "ftx"))

        fig = figures.figure_correlation_heatmaps(corr, corr_index, dates, SYMBOLS, order)

        images = [image for ax in fig.get_axes() for image in ax.images]
        assert len(images) == 3
        assert all(image.get_clim() == (-1.0, 1.0) for image in images)

    def test_panels_share_the_asset_ordering(self, corr, corr_index, topology, events):
        """Same cell, same pair, in every panel."""
        order = hierarchical_order(corr.mean(axis=0))
        dates = figures.select_reference_dates(topology, events, ("terra_luna", "ftx"))

        fig = figures.figure_correlation_heatmaps(corr, corr_index, dates, SYMBOLS, order)

        panels = [ax for ax in fig.get_axes() if ax.images]
        labels = [[t.get_text() for t in ax.get_xticklabels()] for ax in panels]
        assert labels[0] == labels[1] == labels[2]
        assert labels[0] == [SYMBOLS[i] for i in order]

    def test_every_panel_names_both_axes(self, corr, corr_index, topology, events):
        """The panels of a grid sit apart from one another, so each must be
        readable on its own: a panel whose rows are named only on its neighbour
        cannot be interpreted without counting cells across a gap.
        """
        order = hierarchical_order(corr.mean(axis=0))
        dates = figures.select_reference_dates(topology, events, ("terra_luna", "ftx"))

        fig = figures.figure_correlation_heatmaps(corr, corr_index, dates, SYMBOLS, order)

        expected = [SYMBOLS[i] for i in order]
        for panel in [ax for ax in fig.get_axes() if ax.images]:
            assert [t.get_text() for t in panel.get_yticklabels()] == expected
            assert [t.get_text() for t in panel.get_xticklabels()] == expected

    def test_spare_cell_holds_the_colourbar(self, corr, corr_index, topology, events):
        """Three panels in a 2x2 grid leave one cell free; the colour bar takes
        it, so the grid stays square instead of losing width to a bar squeezed
        against the figure edge.
        """
        order = hierarchical_order(corr.mean(axis=0))
        dates = figures.select_reference_dates(topology, events, ("terra_luna", "ftx"))

        fig = figures.figure_correlation_heatmaps(corr, corr_index, dates, SYMBOLS, order)

        panels = [ax for ax in fig.get_axes() if ax.images]
        assert len(panels) == 3
        # The spare grid cell is turned off, and a colour bar axes was added.
        assert any(not ax.axison for ax in fig.get_axes())


class TestGraphSnapshots:
    def test_both_panels_use_identical_node_positions(self, weights, corr_index, topology, events):
        """The property the whole figure rests on.

        A force-directed layout recomputed per date moves every node, so two
        independently laid out snapshots differ everywhere and the compaction
        drowns among nodes that merely shifted. The result still looks like a
        proper figure, which is why this needs a test rather than an eye.
        """
        dates = figures.select_reference_dates(topology, events, ("china_crackdown",))

        fig = figures.figure_graph_snapshots(
            weights, weights, corr_index, dates, SYMBOLS, seed=42
        )

        panels = fig.get_axes()
        assert len(panels) == 2
        positions = [
            next(c.get_offsets() for c in ax.collections if len(c.get_offsets()) == N_ASSETS)
            for ax in panels
        ]
        np.testing.assert_allclose(positions[0], positions[1])

    def test_layout_follows_the_seed(self, weights, corr_index, topology, events):
        """Reproducibility: the same seed must redraw the thesis identically."""
        dates = figures.select_reference_dates(topology, events, ("china_crackdown",))

        def node_positions(seed):
            fig = figures.figure_graph_snapshots(
                weights, weights, corr_index, dates, SYMBOLS, seed=seed
            )
            ax = fig.get_axes()[0]
            return next(c.get_offsets() for c in ax.collections if len(c.get_offsets()) == N_ASSETS)

        np.testing.assert_allclose(node_positions(42), node_positions(42))
        assert not np.allclose(node_positions(42), node_positions(7))


class TestMpSpectrum:
    def test_zooms_past_the_market_mode(self, corr, topology):
        """The market mode sits far right and would squeeze the bulk comparison
        into a sliver; the view is clipped while the histogram keeps the full
        range, so the plotted density stays correct.
        """
        q = N_ASSETS / 60
        upper_edge = (1 + np.sqrt(q)) ** 2

        fig = figures.figure_mp_spectrum(corr, topology, q=q)

        ax = fig.get_axes()[0]
        assert ax.get_xlim()[0] == pytest.approx(0.0)
        assert ax.get_xlim()[1] == pytest.approx(1.15 * upper_edge)

    def test_shows_both_regimes_and_the_theoretical_curve(self, corr, topology):
        fig = figures.figure_mp_spectrum(corr, topology, q=N_ASSETS / 60)

        labels = [text.get_text() for text in fig.get_axes()[0].get_legend().get_texts()]
        assert any("Calmo" in label for label in labels)
        assert any("Crisi" in label for label in labels)
        assert any("Marchenko" in label for label in labels)

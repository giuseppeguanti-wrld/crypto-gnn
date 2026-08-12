"""Tests for figure composition (cryptognn.viz.figures).

These verify the arrangement properties the figures depend on for their meaning
-- a shared node layout, a shared colour scale, a shared time axis. Each is
load-bearing and each fails silently: a snapshot pair laid out independently
still renders a handsome picture, it just no longer shows what it claims to.

None of this was reachable before the composition functions moved out of
scripts/07_make_figures.py, whose name cannot be imported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptognn.evaluation.walkforward import make_folds
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


MODELS = ["zero", "ar", "var-p5", "gcn", "gcn-nograph"]
COSTS = (0.0, 10.0)


@pytest.fixture
def folds() -> list:
    return make_folds(N_WINDOWS, train=100, val=20, test=20, step=20)


@pytest.fixture
def by_fold(folds) -> pd.DataFrame:
    """One row per (model, fold), as summarize_predictions(by="fold") emits."""
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        {
            "model": np.repeat(MODELS, len(folds)),
            "fold": np.tile(np.arange(len(folds)), len(MODELS)),
            "skill_score": rng.normal(0.0, 0.05, len(MODELS) * len(folds)),
        }
    )
    # zero is the reference: its skill is identically 0 by construction, and one
    # baseline runs far below the rest, as var-p5 does on the real data.
    frame.loc[frame["model"] == "zero", "skill_score"] = 0.0
    frame.loc[frame["model"] == "var-p5", "skill_score"] -= 1.0
    return frame


@pytest.fixture
def curves(corr_index) -> pd.DataFrame:
    rng = np.random.default_rng(8)
    blocks = []
    for cost in COSTS:
        for model in [*MODELS, "buy-and-hold"]:
            returns = np.zeros(N_WINDOWS) if model == "zero" else rng.normal(0.0, 0.01, N_WINDOWS)
            blocks.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "cost_bps": cost,
                        "date": corr_index,
                        "equity": np.cumprod(1.0 + returns),
                    }
                )
            )
    return pd.concat(blocks, ignore_index=True)


def _all_figures(corr, corr_index, weights, topology, events, folds, by_fold, curves):
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
        "scheme": figures.figure_walkforward_scheme(folds, corr_index, events),
        "by_fold": figures.figure_results_by_fold(by_fold),
        "equity": figures.figure_equity_curves(curves),
        "density": figures.figure_density_vs_error(topology, by_fold, folds, corr_index),
    }


def test_every_composition_returns_a_figure(corr, corr_index, weights, topology, events, folds, by_fold, curves):
    produced = _all_figures(corr, corr_index, weights, topology, events, folds, by_fold, curves)

    for name, fig in produced.items():
        assert isinstance(fig, plt.Figure), name
        assert fig.get_figwidth() == pytest.approx(FIGURE_WIDTH), name


def test_compositions_write_nothing(
    corr, corr_index, weights, topology, events, folds, by_fold, curves, tmp_path, monkeypatch
):
    """Composition returns a Figure; saving belongs to the script. Checked by
    running every composition with the working directory inside an empty tmp_path
    and asserting it stays empty -- the behavioural counterpart to the AST guard
    in test_contract.py.
    """
    monkeypatch.chdir(tmp_path)

    _all_figures(corr, corr_index, weights, topology, events, folds, by_fold, curves)

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


class TestWalkforwardScheme:
    def test_draws_three_blocks_for_every_fold(self, corr_index, events, folds):
        fig = figures.figure_walkforward_scheme(folds, corr_index, events)

        assert len(fig.get_axes()[0].collections) == 3 * len(folds)

    def test_height_grows_with_the_fold_count(self, corr_index, events):
        """Twenty-four rows in a fixed height would merge into a block."""
        few = figures.figure_walkforward_scheme(
            make_folds(N_WINDOWS, train=100, val=20, test=20, step=60), corr_index, events
        )
        many = figures.figure_walkforward_scheme(
            make_folds(N_WINDOWS, train=100, val=20, test=20, step=20), corr_index, events
        )

        assert many.get_figheight() > few.get_figheight()


class TestResultsByFold:
    def test_the_baseline_is_the_reference_line_not_a_series(self, by_fold):
        """zero has skill 0 in every fold by construction; plotting a constant as
        one of five curves would spend a palette slot on an axis.
        """
        fig = figures.figure_results_by_fold(by_fold)

        labels = [line.get_label() for line in fig.get_axes()[0].lines]
        assert "zero" not in labels

    def test_the_scale_follows_the_emphasized_models(self, by_fold):
        """var-p5 runs an order of magnitude below the GCN arms; a shared scale
        would flatten the comparison the figure exists for.
        """
        fig = figures.figure_results_by_fold(by_fold)

        bottom, top = fig.get_axes()[0].get_ylim()
        accent = by_fold[by_fold["model"].isin(["gcn", "gcn-nograph"])]["skill_score"]
        assert bottom < accent.min() and top > accent.max()
        assert bottom > by_fold["skill_score"].min(), "the off-scale baseline must be off scale"

    def test_says_how_far_the_off_scale_baseline_reaches(self, by_fold):
        fig = figures.figure_results_by_fold(by_fold)

        notes = [text.get_text() for text in fig.get_axes()[0].texts]
        assert any("fuori scala" in note for note in notes)


class TestEquityCurves:
    def test_one_panel_per_cost_level_sharing_the_y_axis(self, curves):
        """The pair is the argument: the same curve twice, read against one scale.
        Independent scales would hide the cost, which is the point."""
        fig = figures.figure_equity_curves(curves)

        axes = fig.get_axes()
        assert len(axes) == len(COSTS)
        assert axes[1].get_shared_y_axes().joined(axes[0], axes[1])

    def test_every_panel_names_its_cost(self, curves):
        fig = figures.figure_equity_curves(curves)

        titles = [ax.get_title() for ax in fig.get_axes()]
        assert titles == [f"{cost:g} bps" for cost in COSTS]


class TestDensityVsError:
    def test_one_panel_per_threshold_sharing_the_y_axis(self, topology, by_fold, folds, corr_index):
        fig = figures.figure_density_vs_error(topology, by_fold, folds, corr_index)

        axes = fig.get_axes()
        assert len(axes) == 2
        assert axes[1].get_shared_y_axes().joined(axes[0], axes[1])

    def test_each_panel_scatters_one_point_per_fold(self, topology, by_fold, folds, corr_index):
        fig = figures.figure_density_vs_error(topology, by_fold, folds, corr_index)

        for ax in fig.get_axes():
            assert ax.collections[0].get_offsets().shape == (len(folds), 2)

    def test_the_panels_read_different_threshold_columns(self, topology, by_fold, folds, corr_index):
        """Two panels of the same column would be a duplicated figure that looks
        like a robustness check.
        """
        fig = figures.figure_density_vs_error(topology, by_fold, folds, corr_index)

        left, right = (ax.collections[0].get_offsets()[:, 0] for ax in fig.get_axes())
        assert not np.allclose(left, right)

    def test_an_unknown_model_names_itself(self, topology, by_fold, folds, corr_index):
        with pytest.raises(ValueError, match="lstm"):
            figures.figure_density_vs_error(topology, by_fold, folds, corr_index, model="lstm")


class TestFoldTestMeans:
    def test_averages_the_metric_over_each_test_block(self, topology, folds, corr_index):
        means = figures.fold_test_means(topology, folds, corr_index, "graph_density")

        expected = [topology["graph_density"].iloc[fold.test].mean() for fold in folds]
        np.testing.assert_allclose(means, expected)

    def test_reconciles_a_timezone_aware_index(self, topology, folds, corr_index):
        """The return panel is UTC-aware and topology.parquet is naive; without
        the reconciliation the reindex is silently all-NaN.
        """
        aware = corr_index.tz_localize("UTC")

        np.testing.assert_allclose(
            figures.fold_test_means(topology, folds, aware, "graph_density"),
            figures.fold_test_means(topology, folds, corr_index, "graph_density"),
        )

    def test_an_unknown_column_lists_the_available_ones(self, topology, folds, corr_index):
        with pytest.raises(ValueError, match="mst_length"):
            figures.fold_test_means(topology, folds, corr_index, "graph_denisty")

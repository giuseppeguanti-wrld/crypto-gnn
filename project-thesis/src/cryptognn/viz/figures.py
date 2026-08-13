"""Composition of the study's figures: arrangement, not drawing, not saving.

Three layers keep the thesis figures and the Streamlit explorer showing the same
thing:

  - **Drawing** -- viz.topology, viz.graphs: functions take an existing `ax` and
    render one thing on it. They never create a Figure.
  - **Composition** -- this module: functions create a Figure, lay out its axes,
    call the drawing functions, and **return** it. They never save.
  - **Saving** -- scripts/07_make_figures.py, the only place savefig appears.

Returning the Figure rather than writing it is what makes these functions
testable: a test can assert that the two graph snapshots really share node
positions, or that the three heatmaps really share a colour scale, without
rendering a PDF and looking at it. Those properties are load-bearing -- a
snapshot pair laid out independently produces a plausible-looking figure that
means nothing -- and until this module existed no test could reach them, because
a file named 07_make_figures.py cannot be imported.

Exports:
  - select_reference_dates(): the calm and post-event dates the figures compare
  - figure_topology_timeseries(), figure_correlation_heatmaps(),
    figure_graph_snapshots(), figure_mp_spectrum(): Section 6.6
  - figure_walkforward_scheme(), figure_results_by_fold(),
    figure_equity_curves(), figure_density_vs_error(): Section 6.5
  - fold_test_means(): a topological metric averaged over each fold's test block

Integration: called by scripts/07_make_figures.py, which saves what they return;
  available to app/streamlit_app.py for its PDF export.
Why this module imports from evaluation: figure_density_vs_error() needs the
  Spearman test it annotates, and computing it here rather than in the script
  keeps it testable -- a file named 07_make_figures.py cannot be imported.
  evaluation.metrics carries only numpy, pandas and scipy, all of which viz
  already requires; the Protocols that pull statsmodels and torch live in
  evaluation.protocols, which stays out.
"""
from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cryptognn.evaluation.metrics import rank_association
from cryptognn.viz.graphs import draw_snapshot, fixed_layout
from cryptognn.viz.results import draw_fold_scheme, draw_model_series, draw_scatter_fit
from cryptognn.viz.style import (
    FIGURE_WIDTH,
    REFERENCE_COLOR,
    STACK_PANEL_HEIGHT,
    WIDE_PANEL_HEIGHT,
    emphasis_colors,
)
from cryptognn.viz.topology import draw_heatmap, draw_metric_series, draw_mp_spectrum

# The first window fully clear of an event: the metric at date t is computed on
# the returns of [t-59, t], so only at +60 does the window contain no pre-event
# data at all. Same choice as the event study, for the same reason.
POST_EVENT_OFFSET = 60

# The zero forecast is these figures' reference line rather than one of their
# curves: its skill score is identically 0 and its equity identically 1.
BASELINE = "zero"

# The four panels of fig_topology_timeseries.pdf: metric column(s), their
# in-panel labels (None for a single untitled series), and the shared y-label.
# Named here rather than inlined in figure_topology_timeseries() because
# app/streamlit_app.py draws the same four panels around its date slider and
# must not carry a second copy of this list -- exactly the risk FIGURE_NAMES
# above is named here to avoid.
TOPOLOGY_TIMESERIES_PANELS: tuple[tuple[list[str], list[str] | None, str], ...] = (
    (["mean_correlation"], None, r"$\bar\rho$"),
    (["graph_density", "graph_density_fwer"], [r"$\tau$", r"$\tau_{\mathrm{FWER}}$"], "Densità"),
    (["algebraic_connectivity_combinatorial"], None, r"$\lambda_2$"),
    (["mst_length"], None, "Lung. MST"),
)

# Every figure the study publishes, in the order the chapter introduces them.
# Named here rather than in the script that draws them because script 08 has to
# copy exactly this set into the thesis, and a list kept in two places is a list
# that will one day be copied incomplete.
FIGURE_NAMES = (
    "fig_topology_timeseries",
    "fig_correlation_heatmaps",
    "fig_graph_snapshots",
    "fig_mp_spectrum",
    "fig_walkforward_scheme",
    "fig_results_by_fold",
    "fig_equity_curves",
    "fig_density_vs_error",
)


def select_reference_dates(
    topology: pd.DataFrame,
    events: list,
    keys: tuple[str, ...],
    offset: int = POST_EVENT_OFFSET,
) -> dict[str, pd.Timestamp]:
    """The dates the comparison figures are drawn at, derived from the data.

    Returns a mapping from panel label to date, in order: first the calm
    reference, then one post-event date per requested key.

    The calm reference is the window with the lowest mean correlation -- chosen
    by the data rather than picked by eye. Each crisis date is its event plus
    `offset` days, the first window containing no pre-event observations; taking
    the event date itself would read a window that is 59/60 pre-event and so
    understate the shock.

    Raises a ValueError listing the available keys when one is unknown, instead
    of the bare KeyError a dict lookup would give: the keys come from
    config/events.yaml, and renaming one there should produce a message that
    says what to use.
    """
    by_key = {event.key: event for event in events}
    unknown = [key for key in keys if key not in by_key]
    if unknown:
        raise ValueError(
            f"Unknown event key(s) {unknown}; config/events.yaml declares {sorted(by_key)}"
        )

    dates = {"Calmo": topology["mean_correlation"].idxmin()}
    for key in keys:
        event = by_key[key]
        dates[f"{event.label} +{offset}g"] = pd.Timestamp(event.date) + pd.Timedelta(days=offset)
    return dates


def figure_topology_timeseries(topology: pd.DataFrame, events: list) -> plt.Figure:
    """The central figure of Section 6.6: four metrics over time, shared x axis.

    The shared axis is the point: a vertical line at a crisis date can be read
    across all four panels at once, which splitting them would destroy.

    The density panel carries both thresholds on one axis. At the calibrated tau
    the series saturates near 1 (cv 0.054) because the crypto universe is almost
    fully connected at that level; showing the FWER threshold beside it makes
    that a visible property of the threshold rather than an unexplained flat
    line. lambda_2 is the combinatorial Fiedler value: the normalized Laplacian
    is invariant to a uniform rescaling of the weights and is therefore nearly
    flat here (cv 0.021 against 0.178).
    """
    panels = TOPOLOGY_TIMESERIES_PANELS

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(FIGURE_WIDTH, STACK_PANEL_HEIGHT * len(panels)),
        sharex=True,
    )
    for ax, (metrics, labels, ylabel) in zip(axes, panels, strict=True):
        draw_metric_series(
            ax,
            topology,
            metrics,
            events=events,
            labels=labels,
            ylabel=ylabel,
            label_events=ax is axes[0],
        )
    axes[-1].set_xlabel("Data")
    fig.align_ylabels(axes)
    fig.tight_layout()
    return fig


def figure_correlation_heatmaps(
    corr: np.ndarray,
    corr_index: pd.DatetimeIndex,
    dates: dict[str, pd.Timestamp],
    symbols: list[str],
    order: np.ndarray,
) -> plt.Figure:
    """Correlation matrices in a grid, on one shared colour scale.

    Laid out two per row rather than in a single strip: three 15x15 matrices side
    by side across \\textwidth leave each cell about 3mm, too small to read. A
    2x2 grid gives each panel roughly 7cm instead of 4.3cm -- half again as wide,
    two and a half times the area -- while keeping all three on one page, which a
    comparison figure needs: the reader cannot hold a field of red squares in
    memory across a page turn.

    `order` is computed once on the period-average matrix and passed in, so the
    same cell means the same pair in every panel; the scale is pinned to [-1, 1]
    inside draw_heatmap for the same reason. Recomputing either per date would
    make the panels incomparable while still looking fine.
    """
    n_columns = 2
    n_rows = int(np.ceil(len(dates) / n_columns))
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(FIGURE_WIDTH, FIGURE_WIDTH * n_rows / n_columns * 1.06),
    )
    flat = axes.ravel()

    image = None
    for panel, (label, date) in enumerate(dates.items()):
        position = corr_index.get_indexer([date], method="nearest")[0]
        image = draw_heatmap(
            flat[panel],
            corr[position],
            order,
            labels=symbols,
            title=f"{label} — {corr_index[position].date()}",
            label_size=7.0,
        )

    # The colour bar takes the cell the panels leave empty, so the grid stays
    # square and no space is spent on a bar squeezed against the figure edge.
    for spare in flat[len(dates) :]:
        spare.set_axis_off()
    if len(dates) < len(flat):
        cax = flat[len(dates)].inset_axes([0.15, 0.45, 0.7, 0.05])
        colorbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    else:
        colorbar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    colorbar.set_label(r"$\rho$")
    colorbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    return fig


def figure_graph_snapshots(
    w_thresh: np.ndarray,
    w_full: np.ndarray,
    corr_index: pd.DatetimeIndex,
    dates: dict[str, pd.Timestamp],
    symbols: list[str],
    seed: int,
) -> plt.Figure:
    """Calm against crisis, at node positions computed once for both panels.

    The single layout is what makes the figure mean anything: a force-directed
    layout recomputed per date moves every node, so two independently laid out
    snapshots differ everywhere and the compaction is lost among nodes that
    merely moved.

    The layout comes from the average **complete** graph, not the thresholded
    one: every pair contributes its true proximity, so the positions reflect the
    market's structure rather than which edges happened to survive tau. The
    edges drawn are the thresholded ones.
    """
    layout = fixed_layout(w_full.mean(axis=0), seed=seed, labels=symbols)

    # Two panels across \textwidth make each one about 2.95in wide, and a
    # force-directed layout fills a roughly square box, so the height is set to
    # match rather than to a fixed ratio: any extra would be white space, any
    # less would compress the graph into a band.
    panel_width = FIGURE_WIDTH / len(dates)
    fig, axes = plt.subplots(1, len(dates), figsize=(FIGURE_WIDTH, panel_width * 1.16))
    for ax, (label, date) in zip(axes, dates.items(), strict=True):
        position = corr_index.get_indexer([date], method="nearest")[0]
        draw_snapshot(
            ax,
            w_thresh[position],
            layout,
            labels=symbols,
            title=f"{label} — {corr_index[position].date()}",
            label_size=6.5,
        )
    fig.tight_layout()
    return fig


def figure_mp_spectrum(corr: np.ndarray, topology: pd.DataFrame, q: float) -> plt.Figure:
    """Pooled eigenvalues by regime against the theoretical Marchenko-Pastur density.

    Eigenvalues are pooled over the calmest and most correlated deciles of
    windows: a single 15x15 matrix offers only 15 eigenvalues, too few to
    histogram. The empirical spectrum does not match the MP bulk, and that is
    the finding, not a failure of the figure -- with the market mode absorbing
    ~71% of the trace, the remaining eigenvalues share far less variance than MP
    assumes and pile up below its lower edge.
    """
    eigenvalues = np.linalg.eigvalsh(corr)
    mean_corr = topology["mean_correlation"].to_numpy()
    calm = mean_corr <= np.quantile(mean_corr, 0.1)
    crisis = mean_corr >= np.quantile(mean_corr, 0.9)

    pooled = {
        "Calmo (decile inferiore)": eigenvalues[calm].ravel(),
        "Crisi (decile superiore)": eigenvalues[crisis].ravel(),
    }
    upper_edge = (1.0 + np.sqrt(q)) ** 2

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, WIDE_PANEL_HEIGHT))
    # Zoom on the bulk: the market mode reaches ~13 and would otherwise squeeze
    # the whole comparison into the leftmost fifth of the axes. The histogram is
    # still built over the full range, so the density shown stays correct.
    draw_mp_spectrum(ax, pooled, q=q, xlim=(0.0, 1.15 * upper_edge))

    ax.annotate(
        "modo di mercato fuori scala: "
        + ", ".join(f"$\\lambda_{{\\max}}={values.max():.1f}$" for values in pooled.values()),
        xy=(0.99, 0.42),
        xycoords="axes fraction",
        ha="right",
        fontsize=6.5,
        color="#444444",
    )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Section 6.5 -- the predictive comparison
# --------------------------------------------------------------------------


def figure_walkforward_scheme(folds: Sequence, dates: pd.DatetimeIndex, events: list) -> plt.Figure:
    """The evaluation protocol drawn to scale, in calendar time.

    The figure a reader checks the anti-look-ahead claim against. Three things
    are meant to be legible from it and from nothing else in the chapter: that
    every test block lies strictly after its own validation and training blocks,
    that consecutive test blocks tile the period without overlapping, and that
    both crises fall inside test blocks rather than inside training data.

    Height scales with the fold count so 24 rows stay separable; at 0.13in per
    fold the bars are about 2mm apart on the page.
    """
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, 1.15 + 0.13 * len(folds)))
    draw_fold_scheme(ax, folds, dates, events=events)
    ax.set_xlabel("Data")
    fig.tight_layout()
    return fig


def figure_results_by_fold(
    by_fold: pd.DataFrame,
    highlight: Sequence[str] = ("gcn", "gcn-nograph"),
) -> plt.Figure:
    """Skill score against the zero forecast, fold by fold.

    The overall table of Section 6.4 gives one number per model, which cannot
    distinguish a model that is uniformly slightly worse from one that is far
    better in some regimes and far worse in others. On 24 folds spanning two
    crises that distinction is the whole question, and it only exists here.

    The zero line is the reference by construction: skill is measured against
    that forecast, so a point above it is a fold the model won.

    The y axis is scaled to the emphasized models and the baselines are allowed
    off it, with their worst value annotated. var-p5 reaches -1.27 in one fold --
    fifteen times the whole range the GCN arms move in -- and a shared scale
    would flatten the comparison the figure exists for into a line. Scaling to
    the subject and stating what falls outside is the honest form of that
    choice; the number itself stays visible, and the full range is in
    summary_all_by_fold.parquet.
    """
    series = by_fold[by_fold["model"] != BASELINE]
    models = list(dict.fromkeys(series["model"]))
    colors = emphasis_colors(models, highlight)

    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, WIDE_PANEL_HEIGHT))
    draw_model_series(
        ax,
        series,
        x="fold",
        y="skill_score",
        colors=colors,
        reference=0.0,
        marker="o",
        muted_label="altre baseline",
    )

    accent = series[series["model"].isin(highlight)]["skill_score"]
    margin = 0.55 * (accent.max() - accent.min())
    ax.set_ylim(accent.min() - margin, accent.max() + margin)

    worst = series["skill_score"].min()
    if worst < ax.get_ylim()[0]:
        ax.annotate(
            f"baseline fuori scala fino a ${worst:+.2f}$",
            xy=(0.5, 0.02),
            xycoords="axes fraction",
            fontsize=6.5,
            color=REFERENCE_COLOR,
            ha="center",
            va="bottom",
        )

    ax.set_xlabel("Fold")
    ax.set_ylabel("Skill score")
    fig.tight_layout()
    return fig


def figure_equity_curves(
    curves: pd.DataFrame,
    highlight: Sequence[str] = ("gcn", "gcn-nograph", "buy-and-hold"),
) -> plt.Figure:
    """The sign strategies' equity, gross of costs and net of them.

    Two panels on one shared y axis, which is the point: the same curve appears
    twice and the reader compares its two shapes directly. A single panel with
    both cost levels would double the traces; two panels on independent scales
    would hide the very difference the pair exists to show.

    The costs are what the figure argues about, so they are in the panel titles
    rather than only in the caption -- a figure lifted into a slide keeps them.
    """
    plotted = curves[curves["model"] != BASELINE]
    models = list(dict.fromkeys(plotted["model"]))
    costs = sorted(plotted["cost_bps"].unique())
    colors = emphasis_colors(models, highlight)

    fig, axes = plt.subplots(
        1,
        len(costs),
        figsize=(FIGURE_WIDTH, WIDE_PANEL_HEIGHT * 1.15),
        sharey=True,
    )
    for ax, cost in zip(np.atleast_1d(axes), costs, strict=True):
        draw_model_series(
            ax,
            plotted[plotted["cost_bps"] == cost],
            x="date",
            y="equity",
            colors=colors,
            reference=1.0,
            muted_label="altre baseline",
        )
        ax.set_title(f"{cost:g} bps")
        ax.set_xlabel("Data")
        ax.tick_params(axis="x", labelrotation=30)
    np.atleast_1d(axes)[0].set_ylabel("Capitale (1 = iniziale)")
    fig.tight_layout()
    return fig


def fold_test_means(topology: pd.DataFrame, folds: Sequence, dates: pd.DatetimeIndex, column: str) -> np.ndarray:
    """Mean of a topological metric over each fold's test block.

    The two frames disagree about timezones -- the return panel is UTC-aware and
    topology.parquet is naive, because .npy has no timezone concept and the
    correlation index round-trips through it. Stripping the tzinfo here is the
    same reconciliation features.align_graph() documents; reindexing without it
    silently produces all-NaN and a figure of empty axes.

    A caveat the caption carries: the metric at date t is computed on the window
    [t-59, t], so a fold's mean partly reflects the days before its test block
    began. That is a property of a rolling window, not a leak -- nothing here
    feeds a forecast.
    """
    if column not in topology.columns:
        raise ValueError(f"No column {column!r} in topology; available: {sorted(topology.columns)}")

    naive = pd.DatetimeIndex(dates).tz_localize(None) if dates.tz is not None else pd.DatetimeIndex(dates)
    means = []
    for fold in folds:
        window = topology[column].reindex(naive[fold.test])
        if window.isna().any():
            raise ValueError(f"Fold {fold.index}: {int(window.isna().sum())} test dates absent from topology")
        means.append(window.mean())
    return np.asarray(means, dtype=float)


def figure_density_vs_error(
    topology: pd.DataFrame,
    by_fold: pd.DataFrame,
    folds: Sequence,
    dates: pd.DatetimeIndex,
    model: str = "gcn",
    columns: Sequence[tuple[str, str]] = (
        ("graph_density", r"Densità ($\tau$ calibrata)"),
        ("graph_density_fwer", r"Densità ($\tau_{\mathrm{FWER}}$)"),
    ),
) -> plt.Figure:
    """Does the GCN do better where the graph is denser? The two questions meeting.

    This is Section 6.5's closing figure and the thesis's own tension put to a
    test: correlation structure is most pronounced exactly in the regimes where
    it is hardest to profit from, so a model that reads structure might be
    helped and hindered by the same thing. A flat cloud is a real answer to that,
    and the figure is drawn whether or not the slope survives its p-value.

    Skill on the y axis rather than RMSE, deliberately. RMSE is dominated by the
    fold's volatility level, and density rises in crises for the same reason
    volatility does -- so a density/RMSE correlation would largely measure that
    shared driver rather than anything about the model. Skill is already
    normalized against the zero forecast within the fold, which removes the
    level and leaves the comparison.

    Two panels because the calibrated threshold saturates: a third of the folds
    sit at a density of exactly 1, so the left panel shows a column of tied
    points. That is a property of the threshold worth seeing, not one to hide,
    and the FWER panel beside it carries the same test where the measurement has
    room to vary.
    """
    arm = by_fold[by_fold["model"] == model]
    if arm.empty:
        raise ValueError(f"No rows for model {model!r} in the per-fold summary")
    skill = arm.sort_values("fold")["skill_score"].to_numpy()

    fig, axes = plt.subplots(
        1,
        len(columns),
        figsize=(FIGURE_WIDTH, WIDE_PANEL_HEIGHT * 1.1),
        sharey=True,
    )
    for ax, (column, label) in zip(np.atleast_1d(axes), columns, strict=True):
        density = fold_test_means(topology, folds, dates, column)
        draw_scatter_fit(ax, density, skill, rank_association(density, skill), reference=0.0)
        ax.set_xlabel(label)
    np.atleast_1d(axes)[0].set_ylabel(f"Skill score ({model})")
    fig.tight_layout()
    return fig

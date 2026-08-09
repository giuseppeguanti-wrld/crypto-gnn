"""Composition of the study's figures: arrangement, not drawing, not saving.

Three layers keep the thesis figures and the Streamlit explorer showing the same
thing (risk R7):

  - **Drawing** -- viz.topology, viz.graphs: functions take an existing `ax` and
    render one thing on it. They never create a Figure.
  - **Composition** -- this module: functions create a Figure, lay out its axes,
    call the drawing functions, and **return** it. They never save.
  - **Saving** -- scripts/06_make_figures.py, the only place savefig appears.

Returning the Figure rather than writing it is what makes these functions
testable: a test can assert that the two graph snapshots really share node
positions, or that the three heatmaps really share a colour scale, without
rendering a PDF and looking at it. Those properties are load-bearing -- a
snapshot pair laid out independently produces a plausible-looking figure that
means nothing -- and until this module existed no test could reach them, because
a file named 06_make_figures.py cannot be imported.

Exports:
  - select_reference_dates(): the calm and post-event dates the figures compare
  - figure_topology_timeseries(), figure_correlation_heatmaps(),
    figure_graph_snapshots(), figure_mp_spectrum()

Integration: called by scripts/06_make_figures.py, which saves what they return;
  available to app/streamlit_app.py for its PDF export.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cryptognn.viz.graphs import draw_snapshot, fixed_layout
from cryptognn.viz.style import FIGURE_WIDTH, STACK_PANEL_HEIGHT, WIDE_PANEL_HEIGHT
from cryptognn.viz.topology import draw_heatmap, draw_metric_series, draw_mp_spectrum

# The first window fully clear of an event: the metric at date t is computed on
# the returns of [t-59, t], so only at +60 does the window contain no pre-event
# data at all. Same choice as the event study, for the same reason.
POST_EVENT_OFFSET = 60


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
    panels = [
        (["mean_correlation"], None, r"$\bar\rho$"),
        (["graph_density", "graph_density_fwer"], [r"$\tau$", r"$\tau_{\mathrm{FWER}}$"], "Densità"),
        (["algebraic_connectivity_combinatorial"], None, r"$\lambda_2$"),
        (["mst_length"], None, "Lung. MST"),
    ]

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(FIGURE_WIDTH, STACK_PANEL_HEIGHT * len(panels)),
        sharex=True,
    )
    for ax, (metrics, labels, ylabel) in zip(axes, panels):
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
    """Correlation matrices side by side on one shared colour scale.

    `order` is computed once on the period-average matrix and passed in, so the
    same cell means the same pair in every panel; the scale is pinned to [-1, 1]
    inside draw_heatmap for the same reason. Recomputing either per date would
    make the panels incomparable while still looking fine.
    """
    fig, axes = plt.subplots(1, len(dates), figsize=(FIGURE_WIDTH, FIGURE_WIDTH / 2.7))

    image = None
    for panel, (ax, (label, date)) in enumerate(zip(axes, dates.items())):
        position = corr_index.get_indexer([date], method="nearest")[0]
        image = draw_heatmap(
            ax,
            corr[position],
            order,
            labels=symbols,
            title=f"{label}\n{corr_index[position].date()}",
            # The ordering is shared, so naming the rows once is enough.
            show_ylabels=panel == 0,
        )

    colorbar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    colorbar.set_label(r"$\rho$")
    colorbar.ax.tick_params(labelsize=6)
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

    fig, axes = plt.subplots(1, len(dates), figsize=(FIGURE_WIDTH, WIDE_PANEL_HEIGHT * 1.3))
    for ax, (label, date) in zip(axes, dates.items()):
        position = corr_index.get_indexer([date], method="nearest")[0]
        draw_snapshot(
            ax,
            w_thresh[position],
            layout,
            labels=symbols,
            title=f"{label} — {corr_index[position].date()}",
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

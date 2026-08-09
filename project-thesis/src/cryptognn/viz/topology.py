"""Drawing functions for the topology figures of thesis Section 6.6.

Every function here takes an existing `ax` and draws on it. None of them creates
a figure, and none of them calls savefig(): composing and saving is the job of
scripts/06_make_figures.py. That separation is what lets the Streamlit app of
Sprint 6 render the *same* pictures as the thesis by calling the *same*
functions -- the alternative is two drawing code paths that drift apart until
neither can be trusted (risk R7 in PLANNING).

Exports:
  - draw_metric_series(): a topological metric over time, with event markers
  - hierarchical_order(): asset ordering from hierarchical clustering
  - draw_heatmap(): correlation matrix with a fixed [-1, 1] color scale
  - marchenko_pastur_density(): the theoretical MP density
  - draw_mp_spectrum(): eigenvalue histogram against that density

Integration: called by scripts/06_make_figures.py and by app/streamlit_app.py.
Why the color scale is pinned: a heatmap rescaled to each date's own range makes
  a calm market and a crisis look identical, since both saturate the colormap.
  Fixing it to [-1, 1] is what makes two dates comparable at a glance.
"""
from __future__ import annotations

import matplotlib.axes
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from cryptognn.graph.build import mantegna_distance
from cryptognn.viz.style import COLORS, DIVERGING_CMAP, EVENT_COLOR


def draw_metric_series(
    ax: matplotlib.axes.Axes,
    df: pd.DataFrame,
    metric: str | list[str],
    events: list | None = None,
    labels: list[str] | None = None,
    ylabel: str | None = None,
    label_events: bool = True,
) -> matplotlib.axes.Axes:
    """Plot one or more topological metrics over time, marking crisis events.

    `metric` may be a list, in which case the series share the axes -- used for
    the density panel, where the calibrated and FWER thresholds are shown
    together so the saturation of the calibrated one is visible rather than
    merely described.

    `events` is a list of cryptognn.events.Event; each becomes a dashed vertical
    rule in a color deliberately outside the series palette, so it can never be
    read as one of the plotted quantities.
    """
    metrics = [metric] if isinstance(metric, str) else list(metric)
    names = labels if labels is not None else metrics

    for position, (column, name) in enumerate(zip(metrics, names)):
        ax.plot(
            df.index,
            df[column],
            color=COLORS[position % len(COLORS)],
            label=name,
            # A dashed second series reads as the variant of the first, which is
            # exactly the relationship between the two density thresholds.
            linestyle="-" if position == 0 else "--",
        )

    for event in events or []:
        ax.axvline(pd.Timestamp(event.date), color=EVENT_COLOR, linestyle=":", linewidth=0.9)
        if label_events:
            # Above the axes, not inside them: an event label overlapping the
            # series it annotates hides the very movement it points at.
            ax.annotate(
                event.label,
                xy=(pd.Timestamp(event.date), 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, 3),
                textcoords="offset points",
                fontsize=6.5,
                color=EVENT_COLOR,
                ha="center",
                va="bottom",
                annotation_clip=False,
            )

    ax.set_ylabel(ylabel if ylabel is not None else names[0])
    ax.margins(x=0.01)
    if len(metrics) > 1:
        ax.legend(loc="best", fontsize=6.5)
    return ax


def hierarchical_order(corr: np.ndarray) -> np.ndarray:
    """Asset ordering from hierarchical clustering of the correlation matrix.

    Reorders assets so correlated ones sit adjacent, turning the block structure
    of the market into visible blocks on the diagonal of a heatmap. Uses average
    linkage on the Mantegna distance -- the same metric the graph is built from,
    so the picture and the model agree on what "close" means.

    Meant to be computed **once** on the period-average correlation matrix and
    reused for every date: an ordering recomputed per date would reshuffle the
    axes between panels and make them incomparable, which is the same failure
    the fixed graph layout avoids.
    """
    corr = np.asarray(corr, dtype=np.float64)
    distance = mantegna_distance(corr)
    np.fill_diagonal(distance, 0.0)  # squareform demands an exactly zero diagonal
    linkage_matrix = linkage(squareform(distance, checks=False), method="average")
    return leaves_list(linkage_matrix)


def draw_heatmap(
    ax: matplotlib.axes.Axes,
    corr: np.ndarray,
    order: np.ndarray,
    labels: list[str] | None = None,
    title: str | None = None,
    show_ylabels: bool = True,
) -> matplotlib.image.AxesImage:
    """Correlation matrix as a heatmap, reordered and on a fixed [-1, 1] scale.

    The color scale is pinned rather than fitted to the data: comparing two
    dates is the entire point of these panels, and a per-date rescaling would
    give a calm market and a crisis the same saturated colors. Returns the image
    so the caller can attach a shared colorbar.

    `show_ylabels` exists for side-by-side panels: the ordering is shared, so
    repeating the asset names on every panel only crowds them.
    """
    corr = np.asarray(corr, dtype=np.float64)
    order = np.asarray(order)
    reordered = corr[np.ix_(order, order)]

    image = ax.imshow(reordered, cmap=DIVERGING_CMAP, vmin=-1.0, vmax=1.0, interpolation="nearest")

    if labels is not None:
        ordered_labels = [labels[i] for i in order]
        ax.set_xticks(range(len(ordered_labels)))
        ax.set_yticks(range(len(ordered_labels)))
        ax.set_xticklabels(ordered_labels, rotation=90, fontsize=5.5)
        ax.set_yticklabels(ordered_labels if show_ylabels else [], fontsize=5.5)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    if title is not None:
        ax.set_title(title)
    ax.grid(False)
    return image


def marchenko_pastur_density(x: np.ndarray, q: float) -> np.ndarray:
    """Marchenko-Pastur density for aspect ratio q = N / T, unit variance.

    p(x) = sqrt((x_+ - x)(x - x_-)) / (2 pi q x) on [x_-, x_+], zero outside,
    with x_± = (1 ± sqrt(q))^2. This is the eigenvalue distribution a
    correlation matrix would have if the underlying series were independent --
    the noise reference against which genuine structure is judged
    (laloux1999noise, eq:marchenko-pastur in the thesis).
    """
    x = np.asarray(x, dtype=np.float64)
    lower = (1.0 - np.sqrt(q)) ** 2
    upper = (1.0 + np.sqrt(q)) ** 2

    inside = (x > lower) & (x < upper)
    density = np.zeros_like(x)
    safe = np.where(inside, x, 1.0)  # avoid 0/0 warnings outside the support
    density[inside] = (
        np.sqrt((upper - safe[inside]) * (safe[inside] - lower)) / (2.0 * np.pi * q * safe[inside])
    )
    return density


def draw_mp_spectrum(
    ax: matplotlib.axes.Axes,
    eigenvalues: dict[str, np.ndarray],
    q: float,
    bin_width: float = 0.1,
    show_edge: bool = True,
    xlim: tuple[float, float] | None = None,
) -> matplotlib.axes.Axes:
    """Empirical eigenvalue histograms against the theoretical MP density.

    `eigenvalues` maps a regime label (e.g. "calmo", "crisi") to its pooled
    eigenvalues. Pooling across the windows of a regime is necessary: a single
    15x15 window yields only 15 eigenvalues, far too few for a histogram.

    The upper edge (1 + sqrt(q))^2 is marked because it, not the fit, is what
    the study uses MP for: eigenvalues beyond it cannot be explained by
    estimation noise.

    Bins are of fixed width over the **full** range of the data, so the density
    normalization stays correct; `xlim` then zooms the view onto the bulk. The
    market mode sits far to the right (lambda ~ 8-13 here) and would otherwise
    squeeze the whole comparison into the leftmost fifth of the axes. Zooming
    the view rather than dropping those eigenvalues from the histogram keeps the
    plotted density honest.
    """
    all_values = np.concatenate([np.asarray(values).ravel() for values in eigenvalues.values()])
    edges = np.arange(0.0, float(all_values.max()) + bin_width, bin_width)

    for position, (label, values) in enumerate(eigenvalues.items()):
        ax.hist(
            np.asarray(values).ravel(),
            bins=edges,
            density=True,
            histtype="step",
            color=COLORS[position % len(COLORS)],
            label=label,
        )

    grid = np.linspace(0.0, float(all_values.max()) * 1.02, 2000)
    ax.plot(
        grid,
        marchenko_pastur_density(grid, q),
        color="black",
        linewidth=1.0,
        label=f"Marchenko-Pastur ($q={q:g}$)",
    )

    if show_edge:
        upper = (1.0 + np.sqrt(q)) ** 2
        ax.axvline(upper, color=EVENT_COLOR, linestyle=":", linewidth=0.9)
        # Anchored to the bottom: the top of these axes is where the legend and
        # the histogram's peak live.
        ax.annotate(
            f"$(1+\\sqrt{{q}})^2 = {upper:.2f}$",
            xy=(upper, 0.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, 4),
            textcoords="offset points",
            fontsize=6.5,
            color=EVENT_COLOR,
            va="bottom",
        )

    if xlim is not None:
        ax.set_xlim(*xlim)

    ax.set_xlabel("Autovalore")
    ax.set_ylabel("Densità")
    ax.legend(loc="upper right")
    return ax

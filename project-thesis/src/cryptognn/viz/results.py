"""Drawing functions for the result figures of thesis Section 6.5.

Same contract as viz.topology: every function takes an existing `ax` and draws on
it, none creates a Figure, none saves. Composing is viz.figures, saving is
scripts/07_make_figures.py, and the separation is what lets the Streamlit app of
Sprint 6 show the reader exactly the pictures the thesis prints.

What distinguishes these figures from the topology ones is the number of series.
Section 6.6 plots one or two quantities; Section 6.5 compares seven models, and
the palette holds four colours that stay separable in greyscale. Cycling them
would give two models the same colour, which is worse than giving them none, so
these functions draw by **emphasis** instead: the models under discussion take a
palette slot, the rest are drawn in MUTED, behind, under a single legend entry.
The choice of which is which is made once by style.emphasis_colors() and handed
in, so two figures of the same chapter cannot disagree about what colour the GCN
is.

Two consequences worth stating, because both are visible in the figures:

  - **The zero forecast is a reference line, not a series.** Its skill score is
    identically 0 and its equity identically 1, by construction. Plotting a
    constant as one of seven curves spends a colour on an axis.
  - **A scatter carries its test.** draw_scatter_fit() takes a RankAssociation
    rather than computing one, so the rho printed on the figure and the rho
    quoted in the text come from the same call.

Exports:
  - draw_fold_scheme(): the walk-forward protocol as train/val/test bars
  - draw_model_series(): several models on one axes, drawn by emphasis
  - draw_scatter_fit(): fold-level scatter with its regression line and rho

Integration: called by cryptognn.viz.figures, hence by
  scripts/07_make_figures.py and by app/streamlit_app.py.
"""
from __future__ import annotations

from collections.abc import Sequence

import matplotlib.axes
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cryptognn.viz.style import COLORS, EVENT_COLOR, MUTED, REFERENCE_COLOR

# Train, validation, test. Fixed slots, in the order the blocks occur in time, so
# the legend reads in the same direction as the bars.
BLOCK_COLORS = (COLORS[0], COLORS[2], COLORS[1])
BLOCK_LABELS = ("Train", "Validazione", "Test")


def draw_fold_scheme(
    ax: matplotlib.axes.Axes,
    folds: Sequence,
    dates: pd.DatetimeIndex,
    events: list | None = None,
    bar_height: float = 0.62,
) -> matplotlib.axes.Axes:
    """The walk-forward protocol: one row per fold, three blocks per row.

    `folds` is a list of cryptognn.evaluation.walkforward.Fold, whose blocks are
    integer positions into `dates`; the conversion happens here so the x axis is
    calendar time. A reader cannot check an anti-look-ahead argument against
    observation indices, and time is what the argument is about.

    Fold 0 sits at the top and time runs to the right, so the staircase descends
    the way the protocol advances. Events are drawn as vertical rules because of
    what they show on this particular figure: both crises fall inside test
    blocks, which is the reason the study trains on 365 days rather than 504 --
    a longer train window would have pushed Terra/Luna into the training period
    and made Section 6.5 uncrossable with Section 6.6.
    """
    for position, fold in enumerate(folds):
        spans = []
        for block in (fold.train, fold.val, fold.test):
            start = mdates.date2num(dates[block[0]])
            spans.append((start, mdates.date2num(dates[block[-1]]) - start))
        for span, color in zip(spans, BLOCK_COLORS, strict=True):
            ax.broken_barh([span], (position - bar_height / 2, bar_height), facecolors=color, linewidth=0)

    for event in events or []:
        ax.axvline(pd.Timestamp(event.date), color=EVENT_COLOR, linestyle=":", linewidth=0.9, zorder=3)
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

    ax.xaxis_date()
    ax.set_ylim(len(folds) - 0.5, -0.5)
    ax.set_yticks(range(0, len(folds), 4))
    ax.set_ylabel("Fold")
    # The bars are the data; a horizontal grid would only stripe them.
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[Patch(facecolor=color, label=label) for color, label in zip(BLOCK_COLORS, BLOCK_LABELS, strict=True)],
        # Upper right: the staircase descends left to right, so that corner is
        # the one region of the axes no bar ever reaches.
        loc="upper right",
        fontsize=7,
        ncol=3,
    )
    return ax


def draw_model_series(
    ax: matplotlib.axes.Axes,
    frame: pd.DataFrame,
    x: str,
    y: str,
    colors: dict[str, str],
    reference: float | None = None,
    marker: str | None = None,
    muted_label: str | None = None,
) -> matplotlib.axes.Axes:
    """Every model's `y` against `x`, the emphasized ones in colour over the rest.

    One function for both the skill-by-fold panel and the equity curves: they
    differ in what is on the axes and in nothing else, and writing them twice is
    how two panels of one chapter end up disagreeing about which model is blue.

    The muted series are drawn first and share a single legend entry. Listing all
    five of them by name would fill a third of the panel to label curves the
    figure is deliberately not asking the reader to tell apart -- their role is
    to show the band the emphasized models sit in.

    `reference` draws the horizontal line the quantity is read against: 0 for a
    skill score, 1 for an equity curve. It is the zero forecast's own series in
    both cases, which is why that model is not plotted as a curve.
    """
    models = list(dict.fromkeys(frame["model"]))
    missing = [name for name in models if name not in colors]
    if missing:
        raise ValueError(f"No colour assigned to {missing}: build the map with style.emphasis_colors()")

    if reference is not None:
        ax.axhline(reference, color=REFERENCE_COLOR, linewidth=0.8, zorder=1)

    accent = []
    for model in sorted(models, key=lambda name: colors[name] != MUTED):
        series = frame[frame["model"] == model].sort_values(x)
        is_muted = colors[model] == MUTED
        line, = ax.plot(
            series[x],
            series[y],
            color=colors[model],
            linewidth=0.8 if is_muted else 1.3,
            alpha=0.75 if is_muted else 1.0,
            marker=None if is_muted else marker,
            markersize=2.5,
            label=model,
            zorder=2 if is_muted else 4,
        )
        if not is_muted:
            accent.append(line)

    handles = list(accent)
    if muted_label is not None and any(colors[model] == MUTED for model in models):
        handles.append(Line2D([], [], color=MUTED, linewidth=0.8, label=muted_label))
    if handles:
        ax.legend(handles=handles, loc="best", fontsize=6.5)
    ax.margins(x=0.02)
    return ax


def draw_scatter_fit(
    ax: matplotlib.axes.Axes,
    x: np.ndarray,
    y: np.ndarray,
    association,
    color: str = COLORS[0],
    reference: float | None = None,
) -> matplotlib.axes.Axes:
    """A fold-level scatter with its least-squares line and Spearman's rho.

    `association` is a cryptognn.evaluation.metrics.RankAssociation, computed by
    the caller. Taking it rather than computing it keeps one number in one place:
    the rho annotated on the figure and the rho quoted in Section 6.5 are the
    same call, and cannot drift into disagreeing by a decimal.

    The line is drawn even when the association is not significant, which here it
    is not. Omitting it whenever p is large would leave the reader unable to see
    *how* flat the relationship is -- and a visible flat line through a scattered
    cloud is the finding this figure exists to report.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if reference is not None:
        ax.axhline(reference, color=REFERENCE_COLOR, linewidth=0.8, zorder=1)

    ax.scatter(x, y, s=14, color=color, alpha=0.85, zorder=3, edgecolors="none")

    span = np.array([x.min(), x.max()])
    ax.plot(span, association.intercept + association.slope * span, color=REFERENCE_COLOR, linewidth=1.0, zorder=2)

    ax.annotate(
        f"$\\rho = {association.rho:+.2f}$, $p = {association.p_value:.2f}$  ($n = {association.n}$)",
        xy=(0.03, 0.03),
        xycoords="axes fraction",
        fontsize=6.5,
        color=REFERENCE_COLOR,
        ha="left",
        va="bottom",
    )
    ax.margins(x=0.08, y=0.12)
    return ax

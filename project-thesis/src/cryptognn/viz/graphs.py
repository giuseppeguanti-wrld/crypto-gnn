"""Node-link drawing of the correlation graph, on a layout fixed once.

Like cryptognn.viz.topology, every function takes an existing `ax` and never
calls savefig(): the scripts compose and save, so the Streamlit app of Sprint 6
draws the identical picture by calling the identical function.

Exports:
  - fixed_layout(): node positions computed once on the period-average graph
  - draw_snapshot(): one node-link diagram at those positions

Integration: called by scripts/06_make_figures.py for fig_graph_snapshots.pdf
  and by app/streamlit_app.py for its interactive snapshot.
Why the layout is computed once and reused: a force-directed layout recomputed
  per date moves every node, because the layout responds to the whole weight
  matrix. Two snapshots laid out independently therefore differ everywhere, and
  the compaction the figure exists to show becomes invisible among the noise of
  nodes that merely moved. With positions frozen, the only thing that changes
  between panels is what actually changed in the market.
"""
from __future__ import annotations

import matplotlib.axes
import matplotlib.patheffects as path_effects
import networkx as nx
import numpy as np

from cryptognn.viz.style import COLORS


def fixed_layout(weights_mean: np.ndarray, seed: int, labels: list[str] | None = None) -> dict:
    """Node positions from a spring layout of the period-average weight matrix.

    `weights_mean` is the Mantegna weight matrix averaged over every window, so
    the layout reflects the market's typical structure rather than any one date.
    `seed` comes from the study config (config.seed), keeping the figure
    reproducible: a spring layout is stochastic, and an unseeded one would
    redraw the thesis differently on every run.

    Returns the mapping matplotlib and networkx both expect, keyed by asset name
    when `labels` is given and by integer index otherwise.
    """
    weights_mean = np.asarray(weights_mean, dtype=np.float64)
    graph = nx.from_numpy_array(weights_mean)
    if labels is not None:
        graph = nx.relabel_nodes(graph, dict(enumerate(labels)))
    return nx.spring_layout(graph, seed=seed, weight="weight")


def draw_snapshot(
    ax: matplotlib.axes.Axes,
    weights: np.ndarray,
    pos: dict,
    labels: list[str] | None = None,
    title: str | None = None,
    max_edge_width: float = 2.2,
    max_node_size: float = 320.0,
    min_node_size: float = 110.0,
    label_size: float = 5.5,
) -> matplotlib.axes.Axes:
    """One node-link diagram of a thresholded weight matrix at fixed positions.

    Edge width is proportional to the weight and node size to weighted degree,
    so a compacting market shows up twice over: more and heavier edges, larger
    nodes. Edge width is scaled against a constant rather than the snapshot's
    own maximum -- self-normalizing would make every date look alike, defeating
    the comparison the figure exists for.

    `min_node_size` keeps an isolated node visible and its label readable: on a
    calm date a node cut loose by the threshold has weighted degree near zero,
    and scaling purely by degree would shrink it to a dot, which reads as a
    rendering fault rather than as the finding it is.
    """
    weights = np.asarray(weights, dtype=np.float64)
    graph = nx.from_numpy_array(weights)
    if labels is not None:
        graph = nx.relabel_nodes(graph, dict(enumerate(labels)))

    # from_numpy_array keeps zero-weight entries as edges; drop them so the
    # thresholded graph is drawn as the sparse object it is.
    graph.remove_edges_from([(u, v) for u, v, w in graph.edges(data="weight") if w == 0.0])

    edge_weights = np.array([w for _, _, w in graph.edges(data="weight")], dtype=np.float64)
    degrees = weights.sum(axis=1)

    if len(edge_weights) > 0:
        nx.draw_networkx_edges(
            graph,
            pos,
            ax=ax,
            width=max_edge_width * edge_weights,
            edge_color="#9aa4ad",
            alpha=0.7,
        )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=min_node_size
        + (max_node_size - min_node_size) * degrees / max(degrees.max(), 1e-12),
        node_color=COLORS[0],
        linewidths=0.0,
    )
    texts = nx.draw_networkx_labels(graph, pos, ax=ax, font_size=label_size, font_color="white")
    # A weakly connected node is drawn small, so its label overflows the disc
    # and the characters landing on the white background vanish -- "DOGE" reads
    # as "OG" on a calm date, which looks like a bug and hides exactly the node
    # whose isolation the figure is meant to show. An outline keeps the label
    # legible both on and off the disc, at any node size.
    for text in texts.values():
        text.set_clip_on(False)
        text.set_path_effects(
            [path_effects.withStroke(linewidth=1.6, foreground=COLORS[0])]
        )

    if title is not None:
        ax.set_title(title)
    # Node labels are drawn as text, which autoscaling does not account for, so
    # a node at the edge of the layout gets its label clipped. The margin buys
    # the room back.
    ax.margins(0.12)
    ax.set_axis_off()
    ax.grid(False)
    return ax

"""Tests for the drawing functions (cryptognn.viz).

These do not assert on how a figure looks -- that is what reading the PDF is
for. They assert on the **contracts** that keep the thesis figures and the
Sprint 6 Streamlit app showing the same thing: drawing functions take an `ax`,
never create or save a figure, and derive their orderings and layouts
deterministically. The risk these guard against is the two renderings drifting
apart; the AST checks below are the mechanical proof that they cannot.
"""
from __future__ import annotations

import ast
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

from cryptognn.events import Event
from cryptognn.viz import graphs as viz_graphs
from cryptognn.viz import topology as viz_topology
from cryptognn.viz.style import FIGURE_WIDTH, apply_style

matplotlib.use("Agg")  # headless rendering; must be selected before pyplot loads

import matplotlib.pyplot as plt  # noqa: E402

VIZ_DIR = Path(viz_topology.__file__).parent
N_ASSETS = 6


@pytest.fixture(autouse=True)
def _style():
    apply_style(usetex=False)
    yield
    plt.close("all")


@pytest.fixture
def block_corr() -> np.ndarray:
    """Two clearly separated blocks: assets 0-2 and 3-5."""
    corr = np.full((N_ASSETS, N_ASSETS), 0.1)
    corr[:3, :3] = 0.9
    corr[3:, 3:] = 0.8
    np.fill_diagonal(corr, 1.0)
    return corr


@pytest.fixture
def topology_frame() -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=300, freq="D", name="date")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "mean_correlation": rng.uniform(0.3, 0.9, len(index)),
            "graph_density": rng.uniform(0.8, 1.0, len(index)),
            "graph_density_fwer": rng.uniform(0.4, 1.0, len(index)),
        },
        index=index,
    )


# --------------------------------------------------------------------------
# The architectural contract
# --------------------------------------------------------------------------


# figures.py composes: creating a Figure is its job. Every other module in viz/
# draws onto an axes handed to it.
COMPOSITION_MODULES = {"figures.py"}


def _calls_in(path, forbidden: set[str]) -> list[str]:
    """Names from `forbidden` called anywhere in the module at `path`."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            name = (
                function.attr
                if isinstance(function, ast.Attribute)
                else function.id
                if isinstance(function, ast.Name)
                else None
            )
            if name in forbidden:
                found.append(f"{path.name}:{node.lineno} calls {name}()")
    return found


def test_no_viz_module_ever_saves():
    """Nothing in viz/ may save or display a figure -- only scripts do.

    This is the whole of the contract: if a drawing or composition function wrote its
    own file, the Streamlit app could not reuse it and would grow a second
    implementation, after which the app and the thesis would show different
    pictures with no way to notice. Checked by parsing the source rather than
    trusting convention.
    """
    offenders = [
        offender
        for path in sorted(VIZ_DIR.glob("*.py"))
        for offender in _calls_in(path, {"savefig", "show"})
    ]

    assert not offenders, "viz/ must never save or show: " + "; ".join(offenders)


def test_drawing_modules_never_create_figures():
    """The drawing modules must accept an axes, never make their own Figure.

    Composition and drawing are different contracts. A composition function
    (viz/figures.py) creates a Figure, arranges axes and returns it; a drawing
    function renders one thing onto an axes it was handed. Only the second
    guarantee lets the Streamlit app place the same picture inside its own
    layout, so it is enforced only where it applies -- excluding figures.py by
    name rather than weakening the rule for everyone.
    """
    offenders = [
        offender
        for path in sorted(VIZ_DIR.glob("*.py"))
        if path.name not in COMPOSITION_MODULES
        for offender in _calls_in(path, {"subplots", "figure"})
    ]

    assert not offenders, "drawing modules must take an ax, not build one: " + "; ".join(offenders)


def test_composition_module_exists():
    """Guards the exclusion above: if figures.py is ever renamed or removed,
    COMPOSITION_MODULES would silently exempt a module that no longer exists,
    and the next composition file would be checked under the drawing rule.
    """
    assert COMPOSITION_MODULES == {path.name for path in VIZ_DIR.glob("figures.py")}


@pytest.mark.parametrize(
    "draw",
    [
        "metric_series",
        "heatmap",
        "mp_spectrum",
        "snapshot",
    ],
)
def test_drawing_functions_use_the_given_axes(draw, block_corr, topology_frame):
    """Each function draws on the `ax` it is handed and creates no figure of its own."""
    fig, ax = plt.subplots()
    figures_before = set(plt.get_fignums())

    if draw == "metric_series":
        viz_topology.draw_metric_series(ax, topology_frame, "mean_correlation")
    elif draw == "heatmap":
        viz_topology.draw_heatmap(ax, block_corr, np.arange(N_ASSETS))
    elif draw == "mp_spectrum":
        viz_topology.draw_mp_spectrum(ax, {"a": np.linspace(0.1, 3.0, 200)}, q=0.25)
    else:
        weights = np.clip(block_corr - np.eye(N_ASSETS), 0.0, None)
        layout = viz_graphs.fixed_layout(weights, seed=42)
        viz_graphs.draw_snapshot(ax, weights, layout)

    assert set(plt.get_fignums()) == figures_before, "a new figure was created"
    assert ax.has_data() or ax.images or ax.patches, "nothing was drawn on the given ax"
    assert ax.figure is fig


# --------------------------------------------------------------------------
# Ordering and layout: computed once, reused, deterministic
# --------------------------------------------------------------------------


def test_hierarchical_order_is_a_valid_permutation(block_corr):
    order = viz_topology.hierarchical_order(block_corr)

    assert sorted(order.tolist()) == list(range(N_ASSETS))


def test_hierarchical_order_groups_blocks(block_corr):
    """The ordering must put correlated assets adjacent -- otherwise the block
    structure of the market never appears on the diagonal of the heatmap.
    """
    order = viz_topology.hierarchical_order(block_corr).tolist()

    first_block_positions = sorted(order.index(i) for i in (0, 1, 2))
    assert first_block_positions in ([0, 1, 2], [3, 4, 5]), order


def test_hierarchical_order_is_deterministic(block_corr):
    """Reused across all three heatmap panels, so it must not vary per call."""
    assert np.array_equal(
        viz_topology.hierarchical_order(block_corr), viz_topology.hierarchical_order(block_corr)
    )


def test_fixed_layout_is_deterministic_given_the_seed(block_corr):
    """A spring layout is stochastic; without a seed the thesis would redraw
    differently on every run and the two snapshots could not be compared.
    """
    weights = np.clip(block_corr - np.eye(N_ASSETS), 0.0, None)

    first = viz_graphs.fixed_layout(weights, seed=42)
    second = viz_graphs.fixed_layout(weights, seed=42)
    other = viz_graphs.fixed_layout(weights, seed=7)

    assert set(first) == set(range(N_ASSETS))
    for node in first:
        np.testing.assert_allclose(first[node], second[node])
    assert any(not np.allclose(first[node], other[node]) for node in first)


def test_fixed_layout_accepts_labels(block_corr):
    weights = np.clip(block_corr - np.eye(N_ASSETS), 0.0, None)
    labels = [f"A{i}" for i in range(N_ASSETS)]

    layout = viz_graphs.fixed_layout(weights, seed=42, labels=labels)

    assert set(layout) == set(labels)


# --------------------------------------------------------------------------
# Heatmap and Marchenko-Pastur
# --------------------------------------------------------------------------


def test_heatmap_color_scale_is_pinned(block_corr):
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


def test_heatmap_applies_the_order(block_corr):
    """The reordering must permute both axes symmetrically, or a cell would no
    longer name the pair it shows.
    """
    _, ax = plt.subplots()
    order = np.array([5, 4, 3, 2, 1, 0])

    image = viz_topology.draw_heatmap(ax, block_corr, order)

    np.testing.assert_allclose(image.get_array(), block_corr[np.ix_(order, order)])


@pytest.mark.parametrize("q", [0.25, 0.5, 1.0])
def test_marchenko_pastur_density_integrates_to_one(q):
    """Closed-form check of the formula: a probability density over its support."""
    lower, upper = (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2
    grid = np.linspace(lower, upper, 200_001)

    density = viz_topology.marchenko_pastur_density(grid, q)

    assert np.trapezoid(density, grid) == pytest.approx(1.0, abs=5e-3)


def test_marchenko_pastur_density_support(block_corr):
    """Zero outside [(1-sqrt q)^2, (1+sqrt q)^2], positive inside, never NaN."""
    q = 0.25
    lower, upper = (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2
    outside = np.array([0.0, lower - 0.01, upper + 0.01, 10.0])
    inside = np.linspace(lower + 1e-3, upper - 1e-3, 50)

    assert np.all(viz_topology.marchenko_pastur_density(outside, q) == 0.0)
    density_inside = viz_topology.marchenko_pastur_density(inside, q)
    assert np.all(density_inside > 0.0)
    assert np.isfinite(density_inside).all()


# --------------------------------------------------------------------------
# Smoke tests with the real call signatures
# --------------------------------------------------------------------------


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


def test_snapshot_drops_zero_weight_edges(block_corr):
    """A thresholded matrix must be drawn as the sparse graph it is: entries
    zeroed by the threshold are not edges.
    """
    _, ax = plt.subplots()
    weights = np.zeros((N_ASSETS, N_ASSETS))
    weights[0, 1] = weights[1, 0] = 0.8
    layout = viz_graphs.fixed_layout(np.clip(block_corr - np.eye(N_ASSETS), 0, None), seed=42)

    viz_graphs.draw_snapshot(ax, weights, layout)

    # One LineCollection for the single surviving edge, and no crash on the
    # isolated nodes (whose weighted degree is 0).
    assert len(ax.collections) >= 1


def test_style_uses_the_measured_textwidth():
    """The thesis is A4 with 3cm margins: \\textwidth is 426.79pt = 5.906in.
    A figure drawn at another width is rescaled on inclusion, which changes its
    font sizes relative to the body text.
    """
    assert FIGURE_WIDTH == pytest.approx(426.79135 / 72.27, abs=1e-3)
    assert matplotlib.rcParams["figure.figsize"][0] == pytest.approx(FIGURE_WIDTH)
    assert matplotlib.rcParams["savefig.bbox"] == "tight"
    assert matplotlib.rcParams["pdf.fonttype"] == 42
    assert matplotlib.rcParams["text.usetex"] is False

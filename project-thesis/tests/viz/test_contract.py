"""The architectural contract of cryptognn.viz, checked against the source.

This module has no counterpart in src/: it tests a rule that spans the whole
package rather than any one file in it. Three layers must stay separated --
drawing takes an `ax`, composition returns a Figure, only scripts save -- and the
reason is Sprint 6. If a drawing function saved or built its own figure, the
Streamlit app could not reuse it, would grow a second implementation, and the app
and the thesis would show different pictures with no way to notice.

Checked by parsing the source rather than by convention, because a convention
that is only written down is one refactor away from being false.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from cryptognn.viz import graphs as viz_graphs
from cryptognn.viz import topology as viz_topology

import matplotlib.pyplot as plt  # isort: skip -- Agg is selected in conftest

VIZ_DIR = Path(viz_topology.__file__).parent
N_ASSETS = 6

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

    This is the whole of the contract: if a drawing or composition function wrote
    its own file, the Streamlit app could not reuse it and would grow a second
    implementation, after which the app and the thesis would show different
    pictures with no way to notice.
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
    """Each function draws on the `ax` it is handed and creates no figure of its own.

    Parametrized across both drawing modules on purpose: the property belongs to
    the layer, not to topology.py or graphs.py individually.
    """
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

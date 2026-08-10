"""Tests for cryptognn.viz.style: the rcParams every figure of the thesis inherits.

One property, and it is a measurement rather than a preference. A figure drawn at
a width other than the document's \\textwidth is rescaled when included, which
changes its font sizes relative to the body text -- so a 9pt axis label becomes
8.2pt on the page while the caption stays 10pt, and the figure stops matching the
document it sits in.
"""
from __future__ import annotations

import matplotlib
import pytest

from cryptognn.viz.style import FIGURE_WIDTH


def test_style_uses_the_measured_textwidth():
    """The thesis is A4 with 3cm margins: \\textwidth is 426.79pt = 5.906in."""
    assert FIGURE_WIDTH == pytest.approx(426.79135 / 72.27, abs=1e-3)
    assert matplotlib.rcParams["figure.figsize"][0] == pytest.approx(FIGURE_WIDTH)
    assert matplotlib.rcParams["savefig.bbox"] == "tight"
    assert matplotlib.rcParams["pdf.fonttype"] == 42
    assert matplotlib.rcParams["text.usetex"] is False

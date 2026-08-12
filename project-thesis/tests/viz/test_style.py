"""Tests for cryptognn.viz.style: the rcParams and palette every figure inherits.

Two properties. The first is a measurement rather than a preference: a figure
drawn at a width other than the document's \\textwidth is rescaled when included,
which changes its font sizes relative to the body text -- so a 9pt axis label
becomes 8.2pt on the page while the caption stays 10pt, and the figure stops
matching the document it sits in.

The second is that colour identifies a model and keeps identifying it. Section
6.5 draws seven models from a four-colour palette, so the emphasis map is what
decides what a reader can distinguish; if it ever assigned by position, drawing a
subset would repaint the survivors and a reader who learned that the GCN is blue
would be misled by the next figure.
"""
from __future__ import annotations

import matplotlib
import pytest

from cryptognn.viz.style import COLORS, FIGURE_WIDTH, MUTED, emphasis_colors


def test_style_uses_the_measured_textwidth():
    """The thesis is A4 with 3cm margins: \\textwidth is 426.79pt = 5.906in."""
    assert FIGURE_WIDTH == pytest.approx(426.79135 / 72.27, abs=1e-3)
    assert matplotlib.rcParams["figure.figsize"][0] == pytest.approx(FIGURE_WIDTH)
    assert matplotlib.rcParams["savefig.bbox"] == "tight"
    assert matplotlib.rcParams["pdf.fonttype"] == 42
    assert matplotlib.rcParams["text.usetex"] is False


class TestEmphasisColors:
    def test_slots_follow_the_order_of_highlight(self):
        colors = emphasis_colors(["a", "b", "c"], ("c", "a"))

        assert colors == {"a": COLORS[1], "b": MUTED, "c": COLORS[0]}

    def test_the_slot_does_not_depend_on_the_position_in_models(self):
        assert emphasis_colors(["a", "b", "c"], ("b",)) == emphasis_colors(["c", "b", "a"], ("b",))

    def test_dropping_a_series_does_not_repaint_the_survivors(self):
        """The property that makes two figures of one chapter agree."""
        full = emphasis_colors(["a", "b", "c"], ("a", "b"))
        subset = emphasis_colors(["a", "b"], ("a", "b"))

        assert subset == {model: full[model] for model in subset}

    def test_every_model_gets_a_colour(self):
        colors = emphasis_colors(["a", "b", "c", "d", "e"], ("a",))

        assert set(colors) == {"a", "b", "c", "d", "e"}
        assert list(colors.values()).count(MUTED) == 4

    def test_refuses_to_cycle_the_palette(self):
        """Past four, two series would share a colour -- worse than neither
        having one, because it looks like an identity the figure does not have.
        """
        with pytest.raises(ValueError, match="palette colours"):
            emphasis_colors(list("abcde"), tuple("abcde"))

    def test_highlighting_an_absent_model_is_an_error(self):
        with pytest.raises(ValueError, match=r"\['gnc'\]"):
            emphasis_colors(["gcn", "ar"], ("gnc",))

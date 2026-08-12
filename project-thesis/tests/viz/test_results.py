"""Tests for cryptognn.viz.results: the drawing functions of Section 6.5.

The property under test throughout is **emphasis**: seven models share a
four-colour palette, so what a reader can tell apart is decided by which series
gets a colour and which gets grey. That decision is load-bearing and it fails
quietly -- a figure whose emphasis went to the wrong model still renders, it just
argues for something the chapter does not claim.

What is checked here is therefore mostly identity and layering: the colour
follows the model name and not its position, the muted band is drawn behind and
labelled once, and the reference line exists. The arithmetic these functions plot
is tested where it is computed, in tests/evaluation/.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptognn.evaluation.metrics import rank_association
from cryptognn.evaluation.walkforward import make_folds
from cryptognn.events import Event
from cryptognn.viz.results import draw_fold_scheme, draw_model_series, draw_scatter_fit
from cryptognn.viz.style import COLORS, MUTED, emphasis_colors

import matplotlib.pyplot as plt  # isort: skip -- Agg is selected in conftest

MODELS = ["gcn", "gcn-nograph", "ar", "var-p5"]
N_FOLDS = 6


def by_fold(models: list[str] = MODELS, n_folds: int = N_FOLDS, seed: int = 5) -> pd.DataFrame:
    """A per-fold summary in the schema summarize_predictions(by="fold") emits."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "model": np.repeat(models, n_folds),
            "fold": np.tile(np.arange(n_folds), len(models)),
            "skill_score": rng.normal(0.0, 0.05, len(models) * n_folds),
        }
    )


class TestFoldScheme:
    def test_draws_three_blocks_per_fold(self):
        folds = make_folds(120, train=40, val=10, test=10, step=10)
        _, ax = plt.subplots()

        draw_fold_scheme(ax, folds, pd.date_range("2021-01-01", periods=120, freq="D"))

        assert len(ax.collections) == 3 * len(folds)

    def test_blocks_run_train_then_validation_then_test(self):
        """The ordering the whole anti-look-ahead argument rests on, read back off
        the axes rather than off the Fold that produced them.
        """
        folds = make_folds(120, train=40, val=10, test=10, step=10)
        _, ax = plt.subplots()

        draw_fold_scheme(ax, folds[:1], pd.date_range("2021-01-01", periods=120, freq="D"))

        starts = [collection.get_paths()[0].vertices[:, 0].min() for collection in ax.collections]
        assert starts == sorted(starts)

    def test_fold_zero_is_at_the_top(self):
        folds = make_folds(120, train=40, val=10, test=10, step=10)
        _, ax = plt.subplots()

        draw_fold_scheme(ax, folds, pd.date_range("2021-01-01", periods=120, freq="D"))

        bottom, top = ax.get_ylim()
        assert bottom > top, "the y axis must be inverted so the staircase descends"

    def test_marks_every_event(self):
        folds = make_folds(120, train=40, val=10, test=10, step=10)
        events = [Event(key="e", date=pd.Timestamp("2021-02-01").date(), label="Evento")]
        _, ax = plt.subplots()

        draw_fold_scheme(ax, folds, pd.date_range("2021-01-01", periods=120, freq="D"), events=events)

        rules = [line for line in ax.lines if len(set(line.get_xdata())) == 1]
        assert len(rules) == 1


class TestEmphasis:
    def test_colour_follows_the_model_not_its_position(self):
        frame = by_fold()
        colors = emphasis_colors(MODELS, ("gcn", "gcn-nograph"))
        _, ax = plt.subplots()

        draw_model_series(ax, frame, "fold", "skill_score", colors)

        drawn = {line.get_label(): line.get_color() for line in ax.lines if line.get_label() in MODELS}
        assert drawn["gcn"] == COLORS[0]
        assert drawn["gcn-nograph"] == COLORS[1]
        assert drawn["ar"] == MUTED and drawn["var-p5"] == MUTED

    def test_reordering_the_frame_does_not_repaint_the_series(self):
        """A reader who learned that gcn is blue must not be misled by the next
        figure. The slot follows the name, so shuffling the input changes nothing.
        """
        shuffled = list(reversed(MODELS))
        first = emphasis_colors(MODELS, ("gcn", "gcn-nograph"))
        second = emphasis_colors(shuffled, ("gcn", "gcn-nograph"))

        assert first == second

    def test_the_muted_band_is_drawn_behind(self):
        frame = by_fold()
        colors = emphasis_colors(MODELS, ("gcn",))
        _, ax = plt.subplots()

        draw_model_series(ax, frame, "fold", "skill_score", colors)

        by_name = {line.get_label(): line for line in ax.lines if line.get_label() in MODELS}
        assert by_name["ar"].get_zorder() < by_name["gcn"].get_zorder()

    def test_the_muted_series_share_one_legend_entry(self):
        frame = by_fold()
        colors = emphasis_colors(MODELS, ("gcn", "gcn-nograph"))
        _, ax = plt.subplots()

        draw_model_series(ax, frame, "fold", "skill_score", colors, muted_label="altre baseline")

        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert labels == ["gcn", "gcn-nograph", "altre baseline"]

    def test_no_muted_entry_when_every_series_is_emphasized(self):
        frame = by_fold(models=["gcn", "gcn-nograph"])
        colors = emphasis_colors(["gcn", "gcn-nograph"], ("gcn", "gcn-nograph"))
        _, ax = plt.subplots()

        draw_model_series(ax, frame, "fold", "skill_score", colors, muted_label="altre baseline")

        labels = [text.get_text() for text in ax.get_legend().get_texts()]
        assert labels == ["gcn", "gcn-nograph"]

    def test_the_reference_line_is_drawn(self):
        frame = by_fold()
        colors = emphasis_colors(MODELS, ("gcn",))
        _, ax = plt.subplots()

        draw_model_series(ax, frame, "fold", "skill_score", colors, reference=0.0)

        flat = [line for line in ax.lines if len(set(line.get_ydata())) == 1 and line.get_label() not in MODELS]
        assert len(flat) == 1
        assert flat[0].get_ydata()[0] == pytest.approx(0.0)

    def test_a_model_without_a_colour_is_an_error(self):
        frame = by_fold()
        _, ax = plt.subplots()

        with pytest.raises(ValueError, match="No colour assigned"):
            draw_model_series(ax, frame, "fold", "skill_score", {"gcn": COLORS[0]})


class TestScatterFit:
    def test_plots_every_point_and_one_line(self):
        x = np.linspace(0.9, 1.0, 24)
        y = np.linspace(-0.08, 0.04, 24)
        _, ax = plt.subplots()

        draw_scatter_fit(ax, x, y, rank_association(x, y))

        assert ax.collections[0].get_offsets().shape == (24, 2)
        assert len(ax.lines) == 1

    def test_the_line_agrees_in_sign_with_the_annotated_rho(self):
        """A figure must not contradict the statistic printed on it. Theil-Sen is
        chosen over least squares for exactly this: on the study's own data two
        low-density folds tilt the OLS line the opposite way from rho.
        """
        x = np.linspace(0.0, 1.0, 20)
        y = -x + np.array([0.4] + [0.0] * 19)  # one high-leverage point at the left end
        _, ax = plt.subplots()

        association = rank_association(x, y)
        draw_scatter_fit(ax, x, y, association)

        drawn = ax.lines[0].get_ydata()
        assert np.sign(drawn[-1] - drawn[0]) == np.sign(association.rho)

    def test_annotates_rho_p_and_the_sample_size(self):
        x = np.linspace(0.0, 1.0, 24)
        y = x**2
        _, ax = plt.subplots()

        draw_scatter_fit(ax, x, y, rank_association(x, y))

        text = ax.texts[0].get_text()
        assert "rho" in text and "p =" in text and "n = 24" in text

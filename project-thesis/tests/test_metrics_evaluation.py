"""Tests for cryptognn.evaluation.metrics: accuracy measures and the DM test.

Named apart from test_metrics.py, which covers the topological metrics of the
graph. What is checked here is arithmetic against values worked out by hand, and
one property no hand calculation would catch: that the Diebold-Mariano statistic
this module computes is the textbook one, formula for formula, rather than a
plausible rearrangement of it.

The conventions under test are the two the study declares in Section 6.4 -- a
zero forecast expresses no direction, and the panel is tested one date at a time
rather than one prediction at a time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptognn.evaluation.metrics import (
    diebold_mariano,
    diebold_mariano_matrix,
    directional_accuracy,
    mae,
    panel_loss_differential,
    rmse,
    skill_score,
    summarize_predictions,
)


def long_predictions(n_dates: int = 40, seed: int = 3) -> pd.DataFrame:
    """A long predictions frame in the harness's own schema, with three models:
    a perfect one, a zero one, and a noisy one, so every aggregate has a known
    ordering before it is computed.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_dates, freq="D", tz="UTC")
    assets = ["A", "B", "C"]
    y = rng.standard_normal((n_dates, len(assets))) * 0.03

    frames = []
    for model, predicted in (
        ("perfect", y),
        ("zero", np.zeros_like(y)),
        ("noisy", y + rng.standard_normal(y.shape) * 0.05),
    ):
        frames.append(
            pd.DataFrame(
                {
                    "fold": 0,
                    "date": dates.repeat(len(assets)),
                    "asset": np.tile(assets, n_dates),
                    "y_true": y.ravel(),
                    "y_pred": predicted.ravel(),
                    "model": model,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


class TestErrorMeasures:
    def test_rmse_and_mae_on_worked_values(self):
        y = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.0, 4.0, 0.0])

        # errors 0, -2, 3 -> mean square 13/3, mean absolute 5/3
        assert rmse(y, pred) == pytest.approx(np.sqrt(13 / 3))
        assert mae(y, pred) == pytest.approx(5 / 3)

    def test_skill_score_endpoints(self):
        y = np.array([1.0, -2.0, 3.0])
        baseline = np.zeros_like(y)

        assert skill_score(y, baseline, baseline) == pytest.approx(0.0)
        assert skill_score(y, y, baseline) == pytest.approx(1.0)
        # Twice the baseline's error in every entry is four times the MSE.
        assert skill_score(y, -y, baseline) == pytest.approx(-3.0)

    def test_skill_score_against_a_perfect_baseline_raises(self):
        y = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="undefined against a perfect reference"):
            skill_score(y, y, y)


class TestDirectionalAccuracy:
    def test_all_signs_right(self):
        y = np.array([1.0, -2.0, 3.0])

        accuracy, coverage = directional_accuracy(y, y * 0.5)

        assert accuracy == pytest.approx(1.0)
        assert coverage == pytest.approx(1.0)

    def test_a_zero_forecast_takes_no_side(self):
        """The convention of Section 6.4: no direction expressed is not a
        direction got wrong, so the metric abstains instead of scoring 0.
        """
        y = np.array([1.0, -2.0, 3.0])

        accuracy, coverage = directional_accuracy(y, np.zeros_like(y))

        assert np.isnan(accuracy)
        assert coverage == 0.0

    def test_zero_returns_and_zero_forecasts_are_excluded_symmetrically(self):
        y = np.array([1.0, -2.0, 0.0, 4.0])
        pred = np.array([2.0, 1.0, 5.0, 0.0])

        accuracy, coverage = directional_accuracy(y, pred)

        # Only the first two pairs qualify; the first agrees, the second does not.
        assert accuracy == pytest.approx(0.5)
        assert coverage == pytest.approx(0.5)


class TestDieboldMariano:
    def test_identical_errors_have_nothing_to_test(self):
        rng = np.random.default_rng(1)
        errors = rng.standard_normal(200)

        with pytest.raises(ValueError, match="Non-positive long-run variance"):
            diebold_mariano(errors, errors)

    def test_matches_the_textbook_formula(self):
        """The whole statistic, recomputed inline: mean loss differential over
        its standard error, times the Harvey-Leybourne-Newbold factor, read
        against a t with T-1 degrees of freedom.
        """
        rng = np.random.default_rng(2)
        errors_a = rng.standard_normal(300)
        errors_b = rng.standard_normal(300) * 1.4

        result = diebold_mariano(errors_a, errors_b)

        d = errors_a**2 - errors_b**2
        n = len(d)
        expected = d.mean() / np.sqrt(d.var(ddof=0) / n) * np.sqrt((n - 1) / n)
        assert result.statistic == pytest.approx(expected)
        assert result.df == n - 1
        assert result.mean_loss_differential == pytest.approx(d.mean())

    def test_sign_says_which_model_is_better(self):
        rng = np.random.default_rng(4)
        accurate = rng.standard_normal(400) * 0.1
        sloppy = rng.standard_normal(400)

        assert diebold_mariano(accurate, sloppy).statistic < 0
        assert diebold_mariano(sloppy, accurate).statistic > 0

    def test_swapping_the_arguments_flips_the_statistic_only(self):
        rng = np.random.default_rng(6)
        errors_a, errors_b = rng.standard_normal(250), rng.standard_normal(250) * 1.2

        forward = diebold_mariano(errors_a, errors_b)
        backward = diebold_mariano(errors_b, errors_a)

        assert forward.statistic == pytest.approx(-backward.statistic)
        assert forward.p_value == pytest.approx(backward.p_value)

    def test_longer_horizon_uses_newey_west_lags(self):
        """At h = 1 the variance is the sample one; beyond it the Bartlett kernel
        adds autocovariance terms, so the statistic must move.
        """
        rng = np.random.default_rng(8)
        errors_a, errors_b = rng.standard_normal(300), rng.standard_normal(300) * 1.3

        assert diebold_mariano(errors_a, errors_b, h=1).statistic != pytest.approx(
            diebold_mariano(errors_a, errors_b, h=5).statistic
        )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"errors_b": np.zeros(10)}, "differ in shape"),
            ({"h": 0}, "horizon must be at least 1"),
        ],
    )
    def test_rejects_impossible_input(self, kwargs, match):
        rng = np.random.default_rng(10)
        arguments = {"errors_a": rng.standard_normal(20), "errors_b": rng.standard_normal(20) * 2, **kwargs}
        with pytest.raises(ValueError, match=match):
            diebold_mariano(**arguments)

    def test_two_dimensional_input_is_refused(self):
        """The panel must be reduced to a daily series first: 22 680 correlated
        pairs passed as one flat sample would shrink the standard error by
        roughly sqrt(15) and manufacture significance.
        """
        rng = np.random.default_rng(12)
        with pytest.raises(ValueError, match="one-dimensional"):
            diebold_mariano(rng.standard_normal((30, 3)), rng.standard_normal((30, 3)))


class TestPanelAggregation:
    def test_loss_differential_has_one_entry_per_date(self):
        predictions = long_predictions(n_dates=40)

        errors_a, errors_b = panel_loss_differential(predictions, "noisy", "zero")

        assert errors_a.shape == errors_b.shape == (40,)
        # The values are root-mean-squared errors per date, so squaring them
        # recovers the per-date MSE the test differences.
        expected = (
            predictions[predictions["model"] == "noisy"]
            .assign(se=lambda frame: (frame["y_true"] - frame["y_pred"]) ** 2)
            .groupby("date")["se"]
            .mean()
            .to_numpy()
        )
        np.testing.assert_allclose(errors_a**2, expected)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="No predictions for model"):
            panel_loss_differential(long_predictions(), "noisy", "gcn")

    def test_summary_ranks_the_models_it_is_given(self):
        predictions = long_predictions()

        summary = summarize_predictions(predictions, baseline="zero")

        assert list(summary.columns) == [
            "model",
            "rmse",
            "mae",
            "directional_accuracy",
            "coverage",
            "skill_score",
            "n_predictions",
        ]
        assert summary["model"].iloc[0] == "perfect"  # sorted by RMSE
        assert summary.set_index("model").loc["perfect", "skill_score"] == pytest.approx(1.0)
        assert summary.set_index("model").loc["zero", "skill_score"] == pytest.approx(0.0)
        assert np.isnan(summary.set_index("model").loc["zero", "directional_accuracy"])
        assert summary.set_index("model").loc["zero", "coverage"] == 0.0

    def test_summary_requires_the_baseline_to_be_present(self):
        with pytest.raises(ValueError, match="absent from the predictions table"):
            summarize_predictions(long_predictions(), baseline="gcn")

    def test_matrix_covers_every_ordered_pair(self):
        predictions = long_predictions()

        matrix = diebold_mariano_matrix(predictions)

        assert len(matrix) == 3 * 2
        assert set(matrix["model_a"]) == {"perfect", "zero", "noisy"}
        forward = matrix[(matrix["model_a"] == "noisy") & (matrix["model_b"] == "zero")].iloc[0]
        backward = matrix[(matrix["model_a"] == "zero") & (matrix["model_b"] == "noisy")].iloc[0]
        assert forward["statistic"] == pytest.approx(-backward["statistic"])

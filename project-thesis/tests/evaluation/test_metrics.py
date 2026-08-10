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
    holm_adjusted,
    mae,
    panel_loss_differential,
    rmse,
    skill_score,
    summarize_predictions,
)

ASSETS = ["A", "B", "C"]
N_FOLDS = 2


def long_predictions(n_dates: int = 40, seed: int = 3) -> pd.DataFrame:
    """A long predictions frame in the harness's own schema, with three models:
    a perfect one, a zero one, and a noisy one, so every aggregate has a known
    ordering before it is computed.

    The dates are split into N_FOLDS contiguous blocks, exactly as the walk-forward
    lays them out, so the per-fold breakdown has something to break down.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_dates, freq="D", tz="UTC")
    y = rng.standard_normal((n_dates, len(ASSETS))) * 0.03
    fold = np.repeat(np.arange(N_FOLDS), n_dates // N_FOLDS)[:n_dates]

    frames = []
    for model, predicted in (
        ("perfect", y),
        ("zero", np.zeros_like(y)),
        ("noisy", y + rng.standard_normal(y.shape) * 0.05),
    ):
        frames.append(
            pd.DataFrame(
                {
                    "fold": np.repeat(fold, len(ASSETS)),
                    "date": dates.repeat(len(ASSETS)),
                    "asset": np.tile(ASSETS, n_dates),
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


class TestGroupedSummary:
    """The per-asset and per-fold breakdowns of Section 6.5."""

    def test_ungrouped_output_is_unchanged(self):
        """The default must stay byte-for-byte what Sprint 3 produced.

        Four scripts and the LaTeX tables of Sprint 5 read this frame; adding a
        grouping option is not licence to reshape the ungrouped one.
        """
        summary = summarize_predictions(long_predictions(), baseline="zero")

        assert list(summary.columns) == [
            "model",
            "rmse",
            "mae",
            "directional_accuracy",
            "coverage",
            "skill_score",
            "n_predictions",
        ]
        assert summary["model"].iloc[0] == "perfect"

    @pytest.mark.parametrize(("by", "expected"), [("asset", len(ASSETS)), ("fold", N_FOLDS)])
    def test_one_row_per_model_and_group(self, by, expected):
        predictions = long_predictions()

        summary = summarize_predictions(predictions, baseline="zero", by=by)

        assert len(summary) == 3 * expected
        assert summary.columns[:2].tolist() == ["model", by]
        assert set(summary[by]) == set(predictions[by])
        assert summary["n_predictions"].sum() == len(predictions)

    def test_grouped_metrics_match_the_subset_computed_by_hand(self):
        predictions = long_predictions()
        chosen = "B"

        summary = summarize_predictions(predictions, baseline="zero", by="asset")
        row = summary[(summary["model"] == "noisy") & (summary["asset"] == chosen)].iloc[0]

        subset = predictions[(predictions["model"] == "noisy") & (predictions["asset"] == chosen)]
        assert row["rmse"] == pytest.approx(rmse(subset["y_true"], subset["y_pred"]))
        assert row["mae"] == pytest.approx(mae(subset["y_true"], subset["y_pred"]))
        assert row["n_predictions"] == len(subset)

    def test_skill_score_is_computed_within_the_group(self):
        """A model that beats zero in one fold and loses in the other reports
        skill of opposite signs, which only happens if the reference is the
        baseline's error *inside each group*. Against the panel-wide zero MSE the
        two folds would be judged on a scale neither of them lives on.
        """
        predictions = long_predictions()
        # A model perfect in fold 0 and doubly wrong in fold 1.
        mixed = predictions[predictions["model"] == "zero"].copy()
        mixed["model"] = "mixed"
        first = mixed["fold"] == 0
        mixed.loc[first, "y_pred"] = mixed.loc[first, "y_true"]
        mixed.loc[~first, "y_pred"] = -2.0 * mixed.loc[~first, "y_true"]
        predictions = pd.concat([predictions, mixed], ignore_index=True)

        by_fold = summarize_predictions(predictions, baseline="zero", by="fold")
        skill = by_fold[by_fold["model"] == "mixed"].set_index("fold")["skill_score"]

        assert skill.loc[0] == pytest.approx(1.0)
        assert skill.loc[1] == pytest.approx(-8.0)  # three times the error, nine times the MSE

    def test_rejects_a_column_the_table_does_not_have(self):
        with pytest.raises(ValueError, match="Cannot group by"):
            summarize_predictions(long_predictions(), by="regime")


class TestHolmCorrection:
    """Family-wise control over the 21 distinct tests of the 7-model comparison."""

    def test_worked_example(self):
        # m = 4: sorted p times (4, 3, 2, 1), then made monotone.
        adjusted = holm_adjusted(np.array([0.01, 0.02, 0.03, 0.04]))
        np.testing.assert_allclose(adjusted, [0.04, 0.06, 0.06, 0.06])

    def test_never_lowers_a_p_value_and_never_exceeds_one(self):
        rng = np.random.default_rng(17)
        p_values = rng.uniform(size=30)

        adjusted = holm_adjusted(p_values)

        assert (adjusted >= p_values - 1e-12).all()
        assert (adjusted <= 1.0).all()

    def test_preserves_the_ordering_weakly(self):
        """A smaller raw p-value never becomes the larger adjusted one.

        Only weakly: the monotonicity step clamps values up to the running
        maximum, so neighbouring tests routinely end up tied. Demanding a strict
        order here would be demanding that Holm not be Holm.
        """
        rng = np.random.default_rng(19)
        p_values = rng.uniform(size=25)

        adjusted = holm_adjusted(p_values)
        order = np.argsort(p_values)

        assert (np.diff(adjusted[order]) >= -1e-12).all()

    def test_matrix_corrects_over_distinct_pairs_not_over_rows(self):
        """Three models make six rows but only three tests.

        Correcting over the six would count each test twice and multiply the
        smallest p-value by 6 instead of 3 -- an adjustment twice as harsh as the
        procedure calls for, and invisible in any output that does not do the
        arithmetic.
        """
        matrix = diebold_mariano_matrix(long_predictions())

        distinct = matrix.drop_duplicates(subset="p_value")
        assert len(matrix) == 6
        assert len(distinct) == 3

        smallest = distinct["p_value"].min()
        row = matrix[matrix["p_value"] == smallest].iloc[0]
        assert row["p_value_holm"] == pytest.approx(min(smallest * 3, 1.0))

    def test_both_directions_of_a_pair_carry_the_same_adjusted_value(self):
        matrix = diebold_mariano_matrix(long_predictions())

        forward = matrix[(matrix["model_a"] == "noisy") & (matrix["model_b"] == "zero")].iloc[0]
        backward = matrix[(matrix["model_a"] == "zero") & (matrix["model_b"] == "noisy")].iloc[0]

        assert forward["p_value_holm"] == pytest.approx(backward["p_value_holm"])

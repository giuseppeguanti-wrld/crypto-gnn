"""Tests for cryptognn.evaluation.backtest: the sign strategy and its costs.

Most of what is checked here is arithmetic against equity curves and turnover
counts worked out by hand. Two properties are worth more than that:

  - a short position on a log return of r earns -(exp(r) - 1) and not -r, which
    is the one silent error a long-short backtest on this panel can make -- it
    produces a plausible curve rather than a failure, so nothing downstream
    would catch it;
  - the cost is charged on position *changes*, so a book that never moves pays
    once and not every day.

The rest of the file pins the conventions Section 6.5 declares: fixed 1/N
weights, NaN rather than zero when a model takes no position at all, and a
drawdown measured from the initial capital rather than from the first close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptognn.evaluation.backtest import (
    BUY_AND_HOLD,
    buy_and_hold,
    cumulative_return,
    max_drawdown,
    run_backtest,
    sharpe,
    sign_strategy,
)

ASSETS = ["A", "B"]
SUMMARY_COLUMNS = [
    "model",
    "cost_bps",
    "n_days",
    "mean_turnover",
    "annualized_return",
    "annualized_volatility",
    "sharpe",
    "max_drawdown",
    "cumulative_return",
]


def long_predictions(y_true: np.ndarray, y_pred: np.ndarray, model: str = "m") -> pd.DataFrame:
    """The harness's own schema around two hand-written (dates x assets) matrices.

    Taking the arrays rather than generating them: every test here asserts
    against a value computed by hand from the same numbers, so the inputs have to
    be visible at the call site.
    """
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    n_dates = y_true.shape[0]
    dates = pd.date_range("2022-01-01", periods=n_dates, freq="D", tz="UTC")

    return pd.DataFrame(
        {
            "fold": 0,
            "date": dates.repeat(len(ASSETS)),
            "asset": np.tile(ASSETS, n_dates),
            "y_true": y_true.ravel(),
            "y_pred": y_pred.ravel(),
            "model": model,
        }
    )


class TestSharpe:
    def test_scales_the_ratio_by_the_root_of_the_period_count(self):
        returns = np.array([0.01, -0.02, 0.03, 0.00])

        expected = returns.mean() / returns.std(ddof=1) * np.sqrt(365)

        assert sharpe(returns) == pytest.approx(expected)
        assert sharpe(returns, periods=1) == pytest.approx(expected / np.sqrt(365))

    def test_a_book_that_never_moves_has_no_sharpe(self):
        # ZeroForecaster's case: no position, so every daily return is exactly
        # zero. 0.0 would read as flat performance rather than as no position.
        assert np.isnan(sharpe(np.zeros(50)))
        assert np.isnan(sharpe(np.full(50, 0.004)))

    def test_a_single_observation_has_no_dispersion_to_divide_by(self):
        assert np.isnan(sharpe(np.array([0.01])))

    @pytest.mark.parametrize(
        ("returns", "periods", "match"),
        [
            (np.zeros((2, 2)), 365, "one-dimensional"),
            (np.zeros(5), 0, "must be positive"),
        ],
    )
    def test_rejects_impossible_input(self, returns, periods, match):
        with pytest.raises(ValueError, match=match):
            sharpe(returns, periods=periods)


class TestMaxDrawdown:
    def test_measures_the_deepest_fall_from_a_running_peak(self):
        # Peak 1.20, trough 0.90 -> 0.90/1.20 - 1 = -0.25. The later fall from
        # 1.50 to 1.20 is only -0.20, so the first one is the reported figure.
        equity = np.array([1.00, 1.20, 0.90, 1.50, 1.20])

        assert max_drawdown(equity) == pytest.approx(-0.25)

    def test_a_curve_that_only_rises_never_draws_down(self):
        assert max_drawdown(np.array([1.0, 1.1, 1.2])) == pytest.approx(0.0)

    def test_the_peak_is_taken_over_the_series_as_given(self):
        # Without the initial capital prepended, the first day's 10% loss is
        # invisible: the running peak starts at 0.90, below par.
        after_a_losing_first_day = np.array([0.90, 0.95])

        assert max_drawdown(after_a_losing_first_day) == pytest.approx(0.0)
        assert max_drawdown(np.concatenate(([1.0], after_a_losing_first_day))) == pytest.approx(-0.10)

    def test_a_total_loss_is_the_floor(self):
        # A long-short book can lose more than its capital, and the raw ratio
        # would then print worse than -100%. Both curves below are wipe-outs.
        assert max_drawdown(np.array([1.0, 0.0, 0.5])) == pytest.approx(-1.0)
        assert max_drawdown(np.array([1.0, -0.4, 0.5])) == pytest.approx(-1.0)

    def test_rejects_a_curve_with_no_capital_to_fall_from(self):
        with pytest.raises(ValueError, match="positive capital"):
            max_drawdown(np.array([0.0, 1.0]))


class TestSignStrategy:
    def test_a_perfect_forecast_earns_on_every_day(self):
        y_true = np.array([[0.02, -0.03], [-0.01, 0.04]])

        curve = sign_strategy(long_predictions(y_true, y_pred=y_true), cost_bps=0)

        assert (curve["gross_return"] > 0).all()
        assert curve["equity"].iloc[-1] > 1.0

    def test_the_opposite_forecast_is_the_mirror_of_it(self):
        y_true = np.array([[0.02, -0.03], [-0.01, 0.04]])

        right = sign_strategy(long_predictions(y_true, y_true), cost_bps=0)
        wrong = sign_strategy(long_predictions(y_true, -y_true), cost_bps=0)

        np.testing.assert_allclose(right["gross_return"], -wrong["gross_return"])

    def test_a_forecast_of_zero_takes_no_position(self):
        y_true = np.array([[0.02, -0.03], [-0.01, 0.04]])

        curve = sign_strategy(long_predictions(y_true, np.zeros_like(y_true)), cost_bps=10)

        np.testing.assert_allclose(curve["gross_return"], 0.0)
        np.testing.assert_allclose(curve["turnover"], 0.0)
        # No position means no cost either: a flat book is not rebalanced.
        np.testing.assert_allclose(curve["equity"], 1.0)

    def test_weights_divide_by_the_whole_universe_not_by_the_active_names(self):
        # One asset held, one flat. The gross return is the held asset's simple
        # return over 2, not over 1: gross exposure stays at 1/2, it is not
        # levered back up to 1 because the other name is undecided.
        y_true = np.array([[0.10, 0.10]])
        y_pred = np.array([[1.0, 0.0]])

        curve = sign_strategy(long_predictions(y_true, y_pred), cost_bps=0)

        assert curve["gross_return"].iloc[0] == pytest.approx(np.expm1(0.10) / 2)

    def test_a_gap_in_the_panel_is_an_error_not_a_nan(self):
        # One (date, asset) cell removed. Silently it would become a NaN return,
        # and the cumulative product would empty the curve from that day on --
        # a failure that looks like a result.
        frame = long_predictions(np.array([[0.01, 0.02], [0.03, 0.04]]), np.ones((2, 2)))

        with pytest.raises(ValueError, match="not rectangular"):
            sign_strategy(frame.drop(index=1), cost_bps=0)


class TestLogReturnConversion:
    def test_a_short_position_earns_the_simple_return_not_the_log_return(self):
        # The property that protects the whole backtest. On r = 0.20 the two
        # readings differ by 1.4 percentage points, and only one of them is what
        # a short actually pays.
        log_return = 0.20
        y_true = np.array([[log_return, log_return]])

        curve = sign_strategy(long_predictions(y_true, -np.ones_like(y_true)), cost_bps=0)

        assert curve["gross_return"].iloc[0] == pytest.approx(-np.expm1(log_return))
        assert curve["gross_return"].iloc[0] != pytest.approx(-log_return)

    def test_a_long_position_compounds_to_the_price_ratio(self):
        # Two days of +10% log return on both assets: the equity curve must end
        # at exp(0.2), the price ratio itself, not at 1.2.
        y_true = np.full((2, 2), 0.10)

        curve = sign_strategy(long_predictions(y_true, np.ones_like(y_true)), cost_bps=0)

        assert curve["equity"].iloc[-1] == pytest.approx(np.exp(0.20))


class TestCosts:
    def test_a_book_that_never_moves_pays_only_to_enter(self):
        # Always long both assets: turnover is 1 on the first day (entering from
        # flat) and 0 afterwards.
        y_true = np.array([[0.01, 0.02], [0.03, -0.01], [0.00, 0.01]])

        curve = sign_strategy(long_predictions(y_true, np.ones_like(y_true)), cost_bps=10)

        np.testing.assert_allclose(curve["turnover"], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(curve["cost"], [10 * 1e-4, 0.0, 0.0])

    def test_flipping_every_name_every_day_turns_the_whole_book_over(self):
        y_true = np.array([[0.01, 0.01], [0.01, 0.01]])
        alternating = np.array([[1.0, 1.0], [-1.0, -1.0]])

        curve = sign_strategy(long_predictions(y_true, alternating), cost_bps=10)

        # Day one enters (|1/2 - 0| twice = 1); day two crosses from long to
        # short on both names (|-1/2 - 1/2| twice = 2).
        np.testing.assert_allclose(curve["turnover"], [1.0, 2.0])

    def test_the_cost_is_subtracted_from_a_gross_return_it_does_not_change(self):
        y_true = np.array([[0.02, -0.03], [-0.01, 0.04]])
        y_pred = np.array([[1.0, -1.0], [1.0, 1.0]])

        free = sign_strategy(long_predictions(y_true, y_pred), cost_bps=0)
        charged = sign_strategy(long_predictions(y_true, y_pred), cost_bps=10)

        np.testing.assert_allclose(free["gross_return"], charged["gross_return"])
        np.testing.assert_allclose(free["turnover"], charged["turnover"])
        np.testing.assert_allclose(charged["net_return"], free["net_return"] - charged["turnover"] * 1e-3)

    def test_rejects_a_negative_cost(self):
        y_true = np.array([[0.01, 0.01]])

        with pytest.raises(ValueError, match="non-negative"):
            sign_strategy(long_predictions(y_true, np.ones_like(y_true)), cost_bps=-1)


class TestBuyAndHold:
    def test_it_is_the_equal_weight_basket(self):
        y_true = np.array([[0.02, -0.03], [-0.01, 0.04]])

        curve = buy_and_hold(long_predictions(y_true, np.zeros_like(y_true)), cost_bps=0)

        np.testing.assert_allclose(curve["gross_return"], np.expm1(y_true).mean(axis=1))

    def test_it_trades_once_and_then_holds(self):
        y_true = np.array([[0.02, -0.03], [-0.01, 0.04], [0.05, 0.05]])

        curve = buy_and_hold(long_predictions(y_true, np.zeros_like(y_true)), cost_bps=10)

        np.testing.assert_allclose(curve["turnover"], [1.0, 0.0, 0.0])

    def test_it_ignores_the_forecasts_entirely(self):
        y_true = np.array([[0.02, -0.03], [-0.01, 0.04]])
        frame = long_predictions(y_true, np.ones_like(y_true))

        # Any model's rows describe the same realized returns, so stacking two
        # models must not double-count a single day.
        stacked = pd.concat([frame, frame.assign(model="other", y_pred=-frame["y_pred"])], ignore_index=True)

        np.testing.assert_allclose(
            buy_and_hold(stacked, cost_bps=0)["equity"],
            buy_and_hold(frame, cost_bps=0)["equity"],
        )


class TestRunBacktest:
    def test_one_row_per_model_and_cost_plus_the_reference(self):
        y_true = np.array([[0.02, -0.03], [-0.01, 0.04], [0.01, 0.01]])
        frame = pd.concat(
            [
                long_predictions(y_true, y_true, model="good"),
                long_predictions(y_true, np.zeros_like(y_true), model="zero"),
            ],
            ignore_index=True,
        )

        result = run_backtest(frame, cost_bps=(0, 10))

        assert list(result.summary.columns) == SUMMARY_COLUMNS
        assert len(result.summary) == 6
        assert list(result.summary["model"]) == ["good", "zero", BUY_AND_HOLD] * 2
        assert sorted(result.summary["cost_bps"].unique()) == [0.0, 10.0]

    def test_the_curves_carry_every_model_at_every_cost(self):
        y_true = np.array([[0.02, -0.03], [-0.01, 0.04], [0.01, 0.01]])

        result = run_backtest(long_predictions(y_true, y_true), cost_bps=(0, 10))

        # 2 models (the forecaster and the reference) x 2 costs x 3 days.
        assert len(result.curves) == 12
        assert set(result.curves["model"]) == {"m", BUY_AND_HOLD}

    def test_a_model_that_takes_no_position_reports_nothing_rather_than_zero(self):
        y_true = np.array([[0.02, -0.03], [-0.01, 0.04], [0.01, 0.01]])

        result = run_backtest(long_predictions(y_true, np.zeros_like(y_true), model="zero"), cost_bps=(0,))
        row = result.summary.set_index("model").loc["zero"]

        assert np.isnan(row["sharpe"])
        assert row["cumulative_return"] == pytest.approx(0.0)
        assert row["mean_turnover"] == pytest.approx(0.0)
        assert row["max_drawdown"] == pytest.approx(0.0)

    @pytest.mark.parametrize(
        ("costs", "match"),
        [
            ((), "at least one"),
            ((10, 10), "Duplicate cost levels"),
        ],
    )
    def test_rejects_an_unusable_set_of_cost_levels(self, costs, match):
        frame = long_predictions(np.array([[0.01, 0.01]]), np.ones((1, 2)))

        with pytest.raises(ValueError, match=match):
            run_backtest(frame, cost_bps=costs)

    def test_rejects_a_model_named_after_the_reference_row(self):
        frame = long_predictions(np.array([[0.01, 0.01]]), np.ones((1, 2)), model=BUY_AND_HOLD)

        with pytest.raises(ValueError, match="collides with the reference"):
            run_backtest(frame, cost_bps=(0,))


def test_cumulative_return_is_the_last_point_of_the_curve():
    assert cumulative_return(np.array([1.1, 1.21])) == pytest.approx(0.21)
    assert np.isnan(cumulative_return(np.array([])))

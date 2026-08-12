"""The economic half of the verdict: what the forecasts are worth after costs.

Section 6.4 measures accuracy; this module measures whether that accuracy is
worth acting on. The two do not have to agree. A model can carry the worse RMSE
and the better Sharpe, because squared error is dominated by the days a forecast
misses hardest while a sign strategy only cares about the days it takes a side
on -- so reporting one without the other lets a reader draw a conclusion the
evidence does not carry.

The strategy is the simplest one the forecasts admit, deliberately: hold the sign
of the prediction on each of the 15 assets, equally weighted, rebalanced daily.
Nothing is optimized here -- no position sizing, no volatility targeting, no
selection of which assets to trade. Every degree of freedom added at this stage
is one more chance to fit the test period, which is the failure the frozen grid
of Sprint 1 exists to prevent and would be silly to reintroduce at the last step.

Five conventions, all of them reported in Section 6.5:

  - **Weights are p / N with N fixed at 15**, not divided by the number of
    non-zero positions. Gross exposure is then constant at or below 1 and the
    equity curves are comparable across models; normalizing by the active count
    would quietly lever up a model on the days it happens to be undecided.
  - **Log returns are converted to simple returns** with expm1() before they are
    weighted. The panel stores r = log(P_t) - log(P_{t-1}), and a short position
    does not earn -r: it earns -(exp(r) - 1). Feeding log returns to a long-short
    backtest as if they were simple ones is the standard silent error of the
    genre, and at 4% daily moves it is not negligible.
  - **Cost is charged on turnover**, sum_i |w_t,i - w_{t-1,i}| times the rate,
    with the book starting flat so the first day pays to enter. The position is
    not liquidated at the end: the last day's exit is not a decision the strategy
    makes within the sample.
  - **Degenerate cases return NaN, never zero.** ZeroForecaster predicts exactly
    0 everywhere, so it holds nothing, and its Sharpe is 0/0. Reporting 0.0 there
    would put a number in the table that reads as a result. This is the same
    convention directional_accuracy() already applies for the same model.
  - **365 periods a year**, not 252: crypto has no market closure to remove.

Exports:
  - sharpe(), max_drawdown(), cumulative_return(): the scalar measures
  - sign_strategy(): long-short positions from the sign of the forecasts
  - buy_and_hold(): the equal-weight basket, as a reference to read the rest against
  - run_backtest(): every model at every cost level -> BacktestResult

Integration: consumed by scripts/06_run_backtest.py, whose artifacts feed
  fig_equity_curves.pdf (S5.2) and tab_backtest.tex (S5.3).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

# One basis point as a fraction. Costs are quoted in bps throughout the study
# because that is the unit exchange fee schedules are written in.
BASIS_POINT = 1e-4

# Crypto trades every day of the year, so there is no closure to remove from the
# annualization factor. data/stylized_facts.py annualizes on the same basis.
PERIODS_PER_YEAR = 365

# Not a forecaster, so it carries no risk of colliding with a model name.
BUY_AND_HOLD = "buy-and-hold"

REQUIRED_COLUMNS = ("date", "asset", "y_true", "y_pred", "model")


@dataclass(frozen=True)
class BacktestResult:
    """The two tables Sprint 5 needs, produced from one pass over the predictions.

    Separate rather than one wide frame because they answer different questions
    and have different shapes: `summary` is what tab_backtest.tex prints, one row
    per model and cost level, while `curves` is the daily series that
    fig_equity_curves.pdf is drawn from.
    """

    summary: pd.DataFrame
    curves: pd.DataFrame


def sharpe(returns: np.ndarray, periods: int = PERIODS_PER_YEAR) -> float:
    """Annualized mean over standard deviation of the period returns.

    No risk-free rate is subtracted, and the omission is deliberate rather than
    an oversight: over 2022-2026 the short rate moved between roughly zero and
    5%, so any single figure would be wrong for most of the sample, and the
    quantity the study compares is the strategies against each other -- a common
    subtraction would shift every row by nearly the same amount.

    Returns NaN on a constant series instead of dividing by zero. That is the
    ZeroForecaster's case exactly: it holds nothing, every daily return is 0, and
    a Sharpe of 0.0 would read as "flat performance" rather than "no position".
    """
    returns = np.asarray(returns, dtype=float)
    if returns.ndim != 1:
        raise ValueError(f"Expected a one-dimensional return series, got {returns.ndim} dimensions")
    if periods <= 0:
        raise ValueError(f"Periods per year must be positive, got {periods}")
    if returns.size < 2:
        return float("nan")

    deviation = float(returns.std(ddof=1))
    if deviation == 0.0:
        return float("nan")
    return float(returns.mean() / deviation * np.sqrt(periods))


def max_drawdown(equity: np.ndarray) -> float:
    """Deepest peak-to-trough fall of an equity curve, as a non-positive fraction.

    The peak runs over the series exactly as given, so a caller that wants the
    first period's loss to count must pass the starting capital as the first
    element -- otherwise the running peak begins at the value *after* day one and
    that fall is free. _summarize_curve() prepends it.

    Equity at or below zero floors the result at -1. A long-short book can in
    principle lose more than its capital, and the raw ratio would then print
    something like -1.4, which is not a drawdown any account can report: past a
    total loss there is nothing further to draw down.
    """
    equity = np.asarray(equity, dtype=float)
    if equity.ndim != 1:
        raise ValueError(f"Expected a one-dimensional equity curve, got {equity.ndim} dimensions")
    if equity.size == 0:
        return float("nan")

    peak = np.maximum.accumulate(equity)
    if peak[0] <= 0:
        raise ValueError(f"Equity curve starts at {equity[0]}: a drawdown needs positive capital to fall from")
    return float(np.min(np.maximum(equity, 0.0) / peak - 1.0))


def cumulative_return(equity: np.ndarray) -> float:
    """Total return over the whole period, from an equity curve starting at 1."""
    equity = np.asarray(equity, dtype=float)
    if equity.size == 0:
        return float("nan")
    return float(equity[-1] - 1.0)


# --------------------------------------------------------------------------
# From the long predictions table to a daily strategy
# --------------------------------------------------------------------------


def _panel(predictions: pd.DataFrame, column: str) -> pd.DataFrame:
    """One column of the long table as a dates x assets matrix.

    Raises on any resulting gap: a model that is missing a date or an asset would
    otherwise contribute a NaN return for that day, which propagates into the
    cumulative product and empties the entire curve from that point on -- a
    failure that looks like a result.
    """
    missing = [name for name in REQUIRED_COLUMNS if name not in predictions.columns]
    if missing:
        raise ValueError(f"Predictions table is missing the columns {missing}")

    wide = predictions.pivot(index="date", columns="asset", values=column)
    if wide.isna().to_numpy().any():
        gaps = int(wide.isna().to_numpy().sum())
        raise ValueError(f"Column {column!r} has {gaps} missing (date, asset) cells: the panel is not rectangular")
    return wide


def _curve(positions: pd.DataFrame, simple_returns: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """Daily gross return, turnover, cost, net return and equity of a position book.

    Shared by both strategies so the cost accounting cannot differ between the
    thing being measured and the thing it is measured against.
    """
    if cost_bps < 0:
        raise ValueError(f"Transaction cost must be non-negative, got {cost_bps} bps")

    gross = (positions * simple_returns).sum(axis=1)
    # The book starts flat, so day one's entry is a position change like any
    # other and is charged. It is never closed: liquidating on the last day is
    # not a decision the strategy makes inside the sample.
    turnover = (positions - positions.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turnover * cost_bps * BASIS_POINT
    net = gross - cost

    return pd.DataFrame(
        {
            "date": positions.index,
            "gross_return": gross.to_numpy(),
            "turnover": turnover.to_numpy(),
            "cost": cost.to_numpy(),
            "net_return": net.to_numpy(),
            "equity": (1.0 + net).cumprod().to_numpy(),
        }
    )


def sign_strategy(predictions: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """Hold the sign of each forecast, equally weighted, rebalanced daily.

    PLANNING writes this as sign_strategy(predictions, returns, cost_bps). The
    realized return is dropped as a separate argument because the long table
    already carries y_true beside y_pred on the same row: passing it again could
    only introduce a misalignment, and an alignment bug here is invisible -- it
    produces a plausible curve rather than an error.

    No shift is applied. walkforward.py records `date` as the date the return is
    realized on, not the origin the forecast was made from, so the position
    implied by a row earns that row's own y_true.
    """
    forecast = _panel(predictions, "y_pred")
    realized = np.expm1(_panel(predictions, "y_true"))
    return _curve(np.sign(forecast) / forecast.shape[1], realized, cost_bps)


def buy_and_hold(predictions: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    """The equal-weight basket, bought on the first day and held.

    A Sharpe of -0.2 means nothing on its own; against the basket over the same
    days it means something. This row is what makes the rest of the table
    readable, and it is also the only line in it that uses no forecast at all.

    Uses y_true only, so any model's rows describe it equally: duplicates on
    (date, asset) are dropped rather than requiring the caller to slice first.
    """
    realized = np.expm1(_panel(predictions.drop_duplicates(["date", "asset"]), "y_true"))
    positions = pd.DataFrame(1.0 / realized.shape[1], index=realized.index, columns=realized.columns)
    return _curve(positions, realized, cost_bps)


# --------------------------------------------------------------------------
# The table of Section 6.5
# --------------------------------------------------------------------------


def _summarize_curve(curve: pd.DataFrame, periods: int) -> dict[str, float | int]:
    """Every reported measure of one strategy at one cost level."""
    net = curve["net_return"].to_numpy()
    equity = curve["equity"].to_numpy()
    n_days = len(curve)
    final = float(equity[-1])

    return {
        "n_days": n_days,
        "mean_turnover": float(curve["turnover"].mean()),
        # Geometric, so it agrees with the cumulative return rather than
        # exceeding it by the volatility drag the arithmetic mean ignores.
        "annualized_return": float(final ** (periods / n_days) - 1.0) if final > 0 else float("nan"),
        "annualized_volatility": float(net.std(ddof=1) * np.sqrt(periods)) if n_days > 1 else float("nan"),
        "sharpe": sharpe(net, periods),
        # Prepended initial capital: see max_drawdown().
        "max_drawdown": max_drawdown(np.concatenate(([1.0], equity))),
        "cumulative_return": cumulative_return(equity),
    }


def run_backtest(
    predictions: pd.DataFrame,
    cost_bps: Sequence[float],
    periods: int = PERIODS_PER_YEAR,
) -> BacktestResult:
    """Every model's sign strategy, plus buy-and-hold, at every cost level.

    Running all cost levels in one call rather than once per level is what lets
    the table be read the way Section 6.5 needs it read: the interesting quantity
    is not the Sharpe at 10 bps but the distance between 0 and 10 bps, and a
    difference is easiest to defend when both halves came from the same code path
    over the same days.

    Models keep the order they appear in, matching diebold_mariano_matrix(), with
    buy-and-hold last in each cost block because it is the reference and not a
    competitor.
    """
    costs = [float(cost) for cost in cost_bps]
    if not costs:
        raise ValueError("No cost level given: the backtest needs at least one")
    if len(set(costs)) != len(costs):
        raise ValueError(f"Duplicate cost levels in {costs}: each would produce an identical row")

    models = list(dict.fromkeys(predictions["model"]))
    if BUY_AND_HOLD in models:
        raise ValueError(f"A model is named {BUY_AND_HOLD!r}, which collides with the reference row")

    rows, curves = [], []
    for cost in costs:
        for model in [*models, BUY_AND_HOLD]:
            curve = (
                buy_and_hold(predictions, cost)
                if model == BUY_AND_HOLD
                else sign_strategy(predictions[predictions["model"] == model], cost)
            )
            rows.append({"model": model, "cost_bps": cost} | _summarize_curve(curve, periods))
            curves.append(curve.assign(model=model, cost_bps=cost))

    return BacktestResult(
        # Stable sort: the cost levels group together and the model order above
        # survives inside each block.
        summary=pd.DataFrame(rows).sort_values("cost_bps", kind="stable", ignore_index=True),
        curves=pd.concat(curves, ignore_index=True)[
            ["model", "cost_bps", "date", "gross_return", "turnover", "cost", "net_return", "equity"]
        ],
    )

"""Entry point for Sprint 5 (S5.1): what the forecasts are worth after costs.

Turns every model's predictions into the sign strategy of Section 6.5 -- hold
the sign of the forecast on each of the 15 assets, equally weighted, rebalanced
daily -- and scores it at two cost levels, with the equal-weight basket as the
reference to read the rest against.

Two cost levels rather than one because the quantity Section 6.5 argues about is
the distance between them. A strategy whose Sharpe survives 10 bps of turnover
cost has said something; one that only works at zero cost has said that the
signal is smaller than the spread, which is a result and gets reported as one.

Recomputes no forecast: the predictions were fixed by scripts 04 and 05 and are
read back from disk, so nothing produced here can move them.

Integration: sixth script in the pipeline (scripts/01-08). Consumes
results/metrics/predictions_{baselines,gcn}.parquet; produces
results/metrics/backtest_all.parquet and backtest_curves_all.parquet, read by
07_make_figures.py for fig_equity_curves.pdf and by 08_make_tables.py for
tab_backtest.tex.

Usage:
    python scripts/06_run_backtest.py
    python scripts/06_run_backtest.py --config config/default.yaml
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cryptognn.artifacts import (
    load_predictions,
    save_backtest,
    save_backtest_curves,
)
from cryptognn.cli import build_parser, run
from cryptognn.config import load_config
from cryptognn.evaluation.backtest import BUY_AND_HOLD, PERIODS_PER_YEAR, run_backtest
from cryptognn.paths import ensure_dirs

# The no-cost arm has no home in the configuration because it is not a parameter
# of the study: it is the reference the configured cost is measured against.
FREE = 0.0


def main() -> None:
    parser = build_parser("Score every model's sign strategy, gross and net of transaction costs.")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs()

    predictions = pd.concat([load_predictions("baselines"), load_predictions("gcn")], ignore_index=True)
    models = list(dict.fromkeys(predictions["model"]))
    costs = (FREE, float(config.backtest.cost_bps))

    n_days = predictions["date"].nunique()
    n_assets = predictions["asset"].nunique()
    print(
        f"{len(models)} models over {n_days} test days x {n_assets} assets "
        f"({predictions['date'].min():%Y-%m-%d} to {predictions['date'].max():%Y-%m-%d}), "
        f"{len(predictions)} predictions"
    )
    print(
        f"sign strategy, equally weighted, annualized on {PERIODS_PER_YEAR} days; "
        f"costs {costs[0]:g} and {costs[1]:g} bps"
    )

    result = run_backtest(predictions, cost_bps=costs)

    expected_rows = (len(models) + 1) * len(costs)
    if len(result.summary) != expected_rows:
        raise ValueError(f"{len(result.summary)} summary rows, expected {expected_rows}")
    if not np.isfinite(result.curves["equity"]).all():
        raise ValueError("Non-finite equity: a strategy return fell to -100% or worse")

    print("\nSign strategy by cost level (buy-and-hold is the equal-weight basket, not a forecast):")
    print(result.summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    report(result.summary, costs)

    save_backtest(result.summary, name="all")
    save_backtest_curves(result.curves, name="all")
    print(
        f"\n  saved backtest_all.parquet ({len(result.summary)} rows) and "
        f"backtest_curves_all.parquet ({len(result.curves)} rows)"
    )


def report(summary: pd.DataFrame, costs: tuple[float, float]) -> None:
    """What the two cost levels do to each strategy, one line per model.

    The table above holds every number; this reads the one comparison the section
    is built on, so it does not have to be reconstructed by eye from sixteen rows.
    """
    free, charged = (summary[summary["cost_bps"] == cost].set_index("model") for cost in costs)

    print(f"\nCost of trading -- Sharpe at {costs[0]:g} bps against {costs[1]:g} bps:")
    for model in free.index:
        gross, net = free.loc[model, "sharpe"], charged.loc[model, "sharpe"]
        print(
            f"  {model:12s} {gross:+7.3f} -> {net:+7.3f}  "
            f"(turnover {free.loc[model, 'mean_turnover']:.3f}/day)  {_cost_reading(gross, net)}"
        )

    # Whether any forecast was worth acting on at all, which is the question the
    # backtest exists to answer and the one the thesis has to state plainly.
    tradable = charged.drop(index=BUY_AND_HOLD)
    tradable = tradable[tradable["sharpe"] > 0]
    if tradable.empty:
        print(f"\n  No model has a positive Sharpe at {costs[1]:g} bps: no forecast here is worth trading.")
    else:
        best = tradable["sharpe"].idxmax()
        basket = charged.loc[BUY_AND_HOLD, "sharpe"]
        verdict = "beats" if tradable.loc[best, "sharpe"] > basket else "still trails"
        print(
            f"\n  Best net of costs: {best} at Sharpe {tradable.loc[best, 'sharpe']:+.3f}, "
            f"which {verdict} the {basket:+.3f} of buying the basket."
        )


def _cost_reading(gross: float, net: float) -> str:
    if not np.isfinite(gross):
        return "no position taken"
    if net > 0:
        return "survives the cost"
    if gross > 0:
        return "the cost consumes the edge"
    return "unprofitable either way"


if __name__ == "__main__":
    run(main)

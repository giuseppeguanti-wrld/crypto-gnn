"""Entry point for Sprint 3 (S3.5): run the baselines and fix the yardstick.

Runs the five baseline configurations of Section 6.4 through the walk-forward
harness -- zero, historical mean, per-asset AR, VAR with BIC-selected order, and
VAR at the pre-registered fixed order -- and writes their out-of-sample
predictions, per-fold diagnostics and comparison tables.

Deliberately run before the GCN exists. The comparison is only meaningful if the
standard was set when there was no model to defend, and after this script the
standard is on disk: any later claim is measured against these numbers, not
against a target chosen once the outcome was visible.

Integration: fourth script in the pipeline (scripts/01-07). Consumes
data/processed/ (from 02_build_graphs.py) through
cryptognn.features.build_study_data(); produces
results/metrics/predictions_baselines.parquet,
diagnostics_baselines.parquet, summary_baselines.parquet and dm_baselines.parquet.

Usage:
    python scripts/04_run_baselines.py
    python scripts/04_run_baselines.py --config config/default.yaml
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cryptognn.artifacts import save_predictions, save_run_diagnostics
from cryptognn.cli import build_parser, run
from cryptognn.config import load_config
from cryptognn.evaluation.metrics import diebold_mariano_matrix, summarize_predictions
from cryptognn.evaluation.walkforward import make_folds_from_config, run_walkforward
from cryptognn.features import build_study_data
from cryptognn.models import baseline_factories
from cryptognn.paths import RESULTS_METRICS, ensure_dirs

BASELINE = "zero"

# Columns every run reports, as opposed to the model-specific ones the
# diagnostics hook contributes.
COMMON_DIAGNOSTICS = ("fold", "model", "n_train", "n_val", "n_test", "fit_seconds")


def main() -> None:
    parser = build_parser("Run the baseline forecasters over the walk-forward protocol.")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs()

    data = build_study_data(config)
    folds = make_folds_from_config(config, data.n_obs)
    expected_rows = len(folds) * len(folds[0].test) * data.n_assets

    print(
        f"{data.n_obs} observations x {data.n_assets} assets, {len(folds)} folds "
        f"({folds[0].sizes[0]} train / {folds[0].sizes[1]} val / {folds[0].sizes[2]} test)"
    )
    print(
        f"test period {data.dates[folds[0].test[0] + 1].date()} -> "
        f"{data.dates[folds[-1].test[-1] + 1].date()}, {expected_rows} predictions per model"
    )

    predictions, diagnostics = [], []
    for name, factory in baseline_factories(config).items():
        result = run_walkforward(factory, data, folds, verbose=False)

        if len(result.predictions) != expected_rows:
            raise ValueError(f"{name}: {len(result.predictions)} predictions, expected {expected_rows}")
        if not np.isfinite(result.predictions["y_pred"]).all():
            raise ValueError(f"{name}: non-finite predictions")

        predictions.append(result.predictions)
        diagnostics.append(result.diagnostics)
        print(f"  {name:8s} fitted over {len(folds)} folds in {result.diagnostics['fit_seconds'].sum():5.1f}s")

    predictions = pd.concat(predictions, ignore_index=True)
    diagnostics = pd.concat(diagnostics, ignore_index=True)

    summary = summarize_predictions(predictions, baseline=BASELINE)
    print("\nAccuracy (sorted by RMSE; skill score against the zero forecast):")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    # Directional accuracy is computed where return and forecast are both
    # non-zero, so a coverage below 1 has two possible causes and they read very
    # differently. Both are reported, rather than left for the reader to guess.
    flat_returns = float((predictions[predictions["model"] == BASELINE]["y_true"] == 0).mean())
    print(f"  coverage note: {flat_returns:.2%} of realized returns are exactly zero (no direction to predict)")
    silent = [
        model
        for model, group in predictions.groupby("model", sort=False)
        if model != BASELINE and (group["y_pred"] == 0).any()
    ]
    if silent:
        print(f"  coverage note: {silent} forecast exactly zero on part of the sample")

    dm = diebold_mariano_matrix(predictions)
    print(f"\nDiebold-Mariano against {BASELINE} (negative statistic = the other model is better):")
    for _, row in dm[dm["model_b"] == BASELINE].iterrows():
        verdict = "significant" if row["p_value"] < 0.05 else "not significant"
        print(
            f"  {row['model_a']:8s} vs {BASELINE}: DM {row['statistic']:+7.3f}  "
            f"p = {row['p_value']:.4f}  ({verdict} at 5%)"
        )

    # The parameter count against the sample size: the empirical form of the
    # argument Section 6.4 makes about multivariate baselines. The order column
    # is named per model rather than detected, since the concatenated frame
    # carries every model's columns and fills the others with NaN.
    print("\nSelected orders and parameter counts (mean over folds):")
    for model, order_column in (("ar", "ar_lag_mean"), ("var-bic", "var_lag_order"), ("var-p5", "var_lag_order")):
        rows = diagnostics[diagnostics["model"] == model]
        print(
            f"  {model:8s} order {rows[order_column].mean():.2f}, "
            f"{rows['n_params'].mean():.0f} parameters from {rows['n_train'].iloc[0]} observations "
            f"({rows['n_train'].iloc[0] * data.n_assets / rows['n_params'].mean():.1f} observations each)"
        )

    save_predictions(predictions)
    save_run_diagnostics(diagnostics)
    summary.to_parquet(RESULTS_METRICS / "summary_baselines.parquet")
    dm.to_parquet(RESULTS_METRICS / "dm_baselines.parquet")
    print(
        f"\n  saved predictions_baselines.parquet ({len(predictions)} rows), "
        "diagnostics_baselines.parquet, summary_baselines.parquet, dm_baselines.parquet"
    )


if __name__ == "__main__":
    run(main)

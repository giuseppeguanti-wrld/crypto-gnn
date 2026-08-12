"""Entry point for Sprint 4 (S4.3): run the GCN and its no-graph ablation.

Runs both arms of Section 6.5 through the walk-forward harness, each one the
frozen grid of Sprint 1 -- every (hidden, dropout) pair, five seeds apiece,
configuration chosen on the fold's validation block, test forecast averaged over
the chosen configuration's seeds.

The two arms differ in exactly one thing: whether A_hat propagates information
between assets or is replaced by the identity. Same features, same parameter
count, same grid, same seeds. That is what makes the difference between their
scores attributable to the graph rather than to anything else, and it is the
cleanest form of the study's first research question.

Integration: fifth script in the pipeline (scripts/01-08). Consumes
data/processed/ (from 02_build_graphs.py) through
cryptognn.features.build_study_data(); produces
results/metrics/predictions_gcn.parquet and diagnostics_gcn.parquet, whose schema
matches the baselines' so Sprint 4's comparison concatenates the two.

Usage:
    python scripts/05_run_gcn.py
    python scripts/05_run_gcn.py --config config/default.yaml
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cryptognn.artifacts import (
    load_predictions,
    save_dm_matrix,
    save_predictions,
    save_run_diagnostics,
    save_summary,
)
from cryptognn.cli import build_parser, run
from cryptognn.config import load_config
from cryptognn.evaluation.metrics import diebold_mariano_matrix, summarize_predictions
from cryptognn.evaluation.walkforward import make_folds_from_config, run_walkforward
from cryptognn.features import build_study_data
from cryptognn.models.gcn import gcn_factories
from cryptognn.paths import ensure_dirs

# The reference of the skill score, the same one Sprint 3 fixed.
BASELINE = "zero"
ALPHA = 0.05


def main() -> None:
    parser = build_parser("Run the GCN and its no-graph ablation over the walk-forward protocol.")
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs()

    data = build_study_data(config)
    folds = make_folds_from_config(config, data.n_obs)
    expected_rows = len(folds) * len(folds[0].test) * data.n_assets

    arms = gcn_factories(config)
    grid = config.model.gcn
    fits_per_fold = len(grid.hidden) * len(grid.dropout) * len(grid.seeds)
    print(
        f"{data.n_obs} observations x {data.n_assets} assets, {len(folds)} folds "
        f"({folds[0].sizes[0]} train / {folds[0].sizes[1]} val / {folds[0].sizes[2]} test)"
    )
    print(
        f"frozen grid: hidden {grid.hidden} x dropout {grid.dropout} x {len(grid.seeds)} seeds "
        f"= {fits_per_fold} fits per fold, {fits_per_fold * len(folds) * len(arms)} in total"
    )

    predictions, diagnostics = [], []
    for name, factory in arms.items():
        result = run_walkforward(factory, data, folds, verbose=False)

        if len(result.predictions) != expected_rows:
            raise ValueError(f"{name}: {len(result.predictions)} predictions, expected {expected_rows}")
        if not np.isfinite(result.predictions["y_pred"]).all():
            raise ValueError(f"{name}: non-finite predictions")

        predictions.append(result.predictions)
        diagnostics.append(result.diagnostics)
        print(f"  {name:12s} fitted over {len(folds)} folds in {result.diagnostics['fit_seconds'].sum():6.1f}s")

    predictions = pd.concat(predictions, ignore_index=True)
    diagnostics = pd.concat(diagnostics, ignore_index=True)

    # RMSE against the zero forecast, computed from this frame's own y_true so the
    # readout needs no baseline artifact. The comparison proper -- every model,
    # with Diebold-Mariano -- belongs to S4.4.
    zero_rmse = float(np.sqrt(np.mean(predictions["y_true"] ** 2)))
    print(f"\nOut-of-sample RMSE (the zero forecast scores {zero_rmse:.6f}):")
    for name, group in predictions.groupby("model", sort=False):
        rmse = float(np.sqrt(np.mean((group["y_true"] - group["y_pred"]) ** 2)))
        print(f"  {name:12s} {rmse:.6f}  skill {1.0 - (rmse / zero_rmse) ** 2:+.4f}")

    # How much the grid mattered: if the four configurations are chosen at
    # comparable rates and score alike, that is a finding for Section 6.5 rather
    # than a detail, and it is only visible here.
    print("\nConfiguration selected on validation, counted over folds:")
    for name, group in diagnostics.groupby("model", sort=False):
        chosen = group.groupby(["selected_hidden", "selected_dropout"]).size()
        summary = ", ".join(f"h{h}/d{d}: {count}" for (h, d), count in chosen.items())
        print(f"  {name:12s} {summary}")
        print(
            f"               mean epochs {group['epochs_mean'].mean():5.1f}, "
            f"early stopped {group['early_stopped_share'].mean():.0%} of fits, "
            f"{int(group['n_params'].iloc[0])} parameters"
        )

    save_predictions(predictions, name="gcn")
    save_run_diagnostics(diagnostics, name="gcn")
    print(
        f"\n  saved predictions_gcn.parquet ({len(predictions)} rows) and "
        f"diagnostics_gcn.parquet ({len(diagnostics)} rows)"
    )

    compare(predictions)


def compare(gcn_predictions: pd.DataFrame) -> None:
    """The comparison of Section 6.5: seven models on one table, with DM tests.

    Reads the baselines back from disk rather than re-running them: they were
    fixed in Sprint 3 precisely so that nothing produced afterwards could move
    them. If script 04 has not been run, MissingArtifactError names it.
    """
    everything = pd.concat([load_predictions("baselines"), gcn_predictions], ignore_index=True)
    models = list(dict.fromkeys(everything["model"]))

    overall = summarize_predictions(everything, baseline=BASELINE)
    by_asset = summarize_predictions(everything, baseline=BASELINE, by="asset")
    by_fold = summarize_predictions(everything, baseline=BASELINE, by="fold")
    dm = diebold_mariano_matrix(everything)

    print(f"\n{'=' * 78}\nSection 6.5 -- {len(models)} models over {everything['fold'].nunique()} folds")
    print("\nAccuracy (sorted by RMSE; skill score against the zero forecast):")
    print(overall.to_string(index=False, float_format=lambda value: f"{value:.6f}"))

    # Negative statistic = the first model is the more accurate of the two.
    print(f"\nDiebold-Mariano against {BASELINE} (negative = the model beats it):")
    for _, row in dm[dm["model_b"] == BASELINE].iterrows():
        print(
            f"  {row['model_a']:12s} DM {row['statistic']:+7.3f}  p {row['p_value']:.4f}  "
            f"p(Holm) {row['p_value_holm']:.4f}  {_verdict(row)}"
        )

    # The ablation's own test: same features, same capacity, same grid, same
    # seeds, and the graph as the only difference. This is the first research
    # question in its most direct form, so it gets its own line rather than a
    # cell in a 7x7 matrix.
    ablation = dm[(dm["model_a"] == "gcn") & (dm["model_b"] == "gcn-nograph")]
    if not ablation.empty:
        row = ablation.iloc[0]
        print("\nThe ablation -- does the graph help? gcn vs gcn-nograph:")
        print(
            f"  DM {row['statistic']:+7.3f}  p {row['p_value']:.4f}  p(Holm) {row['p_value_holm']:.4f}  "
            f"{_verdict(row)}"
        )
        print(f"  (negative would mean the graph helps; {_ablation_reading(row)})")

    for label, table, key in (("asset", by_asset, "asset"), ("fold", by_fold, "fold")):
        arm = table[table["model"] == "gcn"].sort_values("skill_score")
        print(
            f"\nGCN skill by {label}: best {key} {arm[key].iloc[-1]} at {arm['skill_score'].iloc[-1]:+.4f}, "
            f"worst {key} {arm[key].iloc[0]} at {arm['skill_score'].iloc[0]:+.4f}, "
            f"positive on {int((arm['skill_score'] > 0).sum())} of {len(arm)}"
        )

    save_summary(overall, name="all")
    save_summary(by_asset, name="all_by_asset")
    save_summary(by_fold, name="all_by_fold")
    save_dm_matrix(dm, name="all")
    print(
        f"\n  saved summary_all.parquet ({len(overall)} rows), "
        f"summary_all_by_asset.parquet ({len(by_asset)}), "
        f"summary_all_by_fold.parquet ({len(by_fold)}), dm_all.parquet ({len(dm)})"
    )


def _verdict(row: pd.Series) -> str:
    if row["p_value_holm"] < ALPHA:
        return "significant after Holm"
    if row["p_value"] < ALPHA:
        return "significant raw only"
    return "not significant"


def _ablation_reading(row: pd.Series) -> str:
    if row["p_value_holm"] < ALPHA:
        direction = "helps" if row["statistic"] < 0 else "hurts"
        return f"the graph {direction}, and the difference survives the correction"
    return "the difference is not distinguishable from noise"


if __name__ == "__main__":
    run(main)

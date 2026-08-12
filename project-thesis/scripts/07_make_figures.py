"""Entry point for the study's figures (S2.5 and S5.2).

Loads the artifacts, composes the figures, and saves them. This is the only
place savefig() appears in the project: the drawing functions of cryptognn.viz
take an `ax`, and the composition functions of cryptognn.viz.figures return a
Figure, so the Streamlit app of Sprint 6 can produce the same pictures from the
same code instead of a second, divergent implementation.

Recomputes no result: every input comes from scripts 01-06 through
cryptognn.artifacts. The folds are rebuilt here because they are a function of
the configuration rather than an artifact, and from load_returns() rather than
build_study_data(), which would read volumes, features and the graph tensor to
use nothing but their date index.

Integration: seventh script in the pipeline (scripts/01-08). Consumes
data/processed/*.npy and results/metrics/{topology,tau_calibration,
summary_all_by_fold,backtest_curves_all}.*, plus config/events.yaml; writes PDFs
to results/figures/, later copied into ../latex-thesis/figures/ by S5.3.

Usage:
    python scripts/07_make_figures.py
    python scripts/07_make_figures.py --usetex        # final run, slow
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

from cryptognn.artifacts import (
    load_backtest_curves,
    load_corr,
    load_returns,
    load_summary,
    load_tau,
    load_topology,
    load_w_full,
    load_w_thresh,
)
from cryptognn.cli import build_parser, run
from cryptognn.config import load_config
from cryptognn.evaluation.metrics import rank_association
from cryptognn.evaluation.walkforward import make_folds_from_config
from cryptognn.events import load_events
from cryptognn.paths import RESULTS_FIGURES, ensure_dirs
from cryptognn.viz.figures import (
    FIGURE_NAMES,
    POST_EVENT_OFFSET,
    figure_correlation_heatmaps,
    figure_density_vs_error,
    figure_equity_curves,
    figure_graph_snapshots,
    figure_mp_spectrum,
    figure_results_by_fold,
    figure_topology_timeseries,
    figure_walkforward_scheme,
    fold_test_means,
    select_reference_dates,
)
from cryptognn.viz.style import apply_style
from cryptognn.viz.topology import hierarchical_order

matplotlib.use("Agg")  # headless rendering; must be selected before pyplot loads

import matplotlib.pyplot as plt


def _save(fig: plt.Figure, outdir: Path, name: str) -> None:
    path = outdir / f"{name}.pdf"
    # Matplotlib stamps a CreationDate into the PDF, which would make every run
    # produce a different file for identical content -- churn in git once the
    # figures are committed, and no way to tell a real change from a rerun.
    # Suppressing it makes the figures byte-reproducible from the same inputs.
    fig.savefig(path, metadata={"CreationDate": None})
    plt.close(fig)
    print(f"  saved {path.name}")


def main() -> None:
    parser = build_parser("Compose and save the study's figures.", events=True)
    parser.add_argument("--outdir", type=Path, default=None, help="Default: results/figures/")
    parser.add_argument(
        "--usetex",
        action="store_true",
        help="Typeset text with LaTeX for an exact font match. Slow and fragile "
        "in a loop: use only for the final run.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs()
    outdir = args.outdir or RESULTS_FIGURES
    outdir.mkdir(parents=True, exist_ok=True)
    apply_style(usetex=args.usetex)

    window = config.graph.window

    corr, corr_index = load_corr(window)
    w_full = load_w_full()
    w_thresh = load_w_thresh()
    topology = load_topology()
    calibration = load_tau()
    events = load_events(args.events)

    by_fold = load_summary("all_by_fold")
    curves = load_backtest_curves()
    dates = load_returns().index
    folds = make_folds_from_config(config, len(dates))

    symbols = config.data.symbols
    q = len(symbols) / window

    # Dates come from the data, never hardcoded: calm is the least correlated
    # window, crisis is each event's first fully post-event window.
    heatmap_dates = select_reference_dates(topology, events, ("terra_luna", "ftx"))
    snapshot_dates = select_reference_dates(topology, events, ("china_crackdown",))
    calm_date = heatmap_dates["Calmo"]
    print(f"calm window: {calm_date.date()} (mean rho = {topology['mean_correlation'].min():.3f})")
    print(f"tau = {calibration.tau:.4f}, q = {q:.3f}, post-event offset = {POST_EVENT_OFFSET}d")

    # One ordering for every panel that uses it.
    order = hierarchical_order(corr.mean(axis=0))
    print(f"asset order (hierarchical): {[symbols[i] for i in order]}")

    report_association(topology, by_fold, folds, dates)

    print("Drawing figures...")
    figures = {
        "fig_topology_timeseries": figure_topology_timeseries(topology, events),
        "fig_correlation_heatmaps": figure_correlation_heatmaps(
            corr, corr_index, heatmap_dates, symbols, order
        ),
        "fig_graph_snapshots": figure_graph_snapshots(
            w_thresh, w_full, corr_index, snapshot_dates, symbols, config.seed
        ),
        "fig_mp_spectrum": figure_mp_spectrum(corr, topology, q),
        "fig_walkforward_scheme": figure_walkforward_scheme(folds, dates, events),
        "fig_results_by_fold": figure_results_by_fold(by_fold),
        "fig_equity_curves": figure_equity_curves(curves),
        "fig_density_vs_error": figure_density_vs_error(topology, by_fold, folds, dates),
    }
    # The set script 08 will copy into the thesis. Checked rather than assumed:
    # a figure added here and not there is published as a stale file from an
    # earlier run, which is the failure that looks like success.
    if tuple(figures) != FIGURE_NAMES:
        raise ValueError(f"Composed {tuple(figures)}, but viz.figures declares {FIGURE_NAMES}")

    for name, fig in figures.items():
        _save(fig, outdir, name)


def report_association(topology, by_fold, folds, dates) -> None:
    """The numbers behind fig_density_vs_error, printed for the write-up.

    The figure annotates them, but Section 6.5 has to quote them in prose too,
    and reading a rho off a PDF is how a thesis ends up with a number that does
    not match its own figure.
    """
    skill = by_fold[by_fold["model"] == "gcn"].sort_values("fold")["skill_score"].to_numpy()

    print(f"\nGCN skill vs graph density, over {len(folds)} folds:")
    for column in ("graph_density", "graph_density_fwer"):
        density = fold_test_means(topology, folds, dates, column)
        association = rank_association(density, skill)
        saturated = int((density >= 1.0).sum())
        print(
            f"  {column:20s} rho {association.rho:+.3f}  p {association.p_value:.3f}  "
            f"(range {density.min():.3f}-{density.max():.3f}, {saturated} folds at 1.000)"
        )


if __name__ == "__main__":
    run(main)

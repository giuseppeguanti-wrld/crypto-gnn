"""Entry point for the study's figures (S2.5 now, S5.2 later).

Loads the artifacts, composes the figures, and saves them. This is the only
place savefig() appears in the project: the drawing functions of cryptognn.viz
take an `ax`, and the composition functions of cryptognn.viz.figures return a
Figure, so the Streamlit app of Sprint 6 can produce the same pictures from the
same code instead of a second, divergent implementation.

Recomputes nothing: every input comes from scripts 01-03 through
cryptognn.artifacts.

Integration: sixth script in the pipeline (scripts/01-07). Consumes
data/processed/*.npy, results/metrics/{topology,tau_calibration}.* and
config/events.yaml; writes PDFs to results/figures/, later copied into
../latex-thesis/figures/ by S5.3.

Usage:
    python scripts/06_make_figures.py
    python scripts/06_make_figures.py --usetex        # final run, slow
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

from cryptognn.artifacts import load_corr, load_tau, load_topology, load_w_full, load_w_thresh
from cryptognn.cli import build_parser, run
from cryptognn.config import load_config
from cryptognn.events import load_events
from cryptognn.paths import RESULTS_FIGURES, ensure_dirs
from cryptognn.viz.figures import (
    POST_EVENT_OFFSET,
    figure_correlation_heatmaps,
    figure_graph_snapshots,
    figure_mp_spectrum,
    figure_topology_timeseries,
    select_reference_dates,
)
from cryptognn.viz.style import apply_style
from cryptognn.viz.topology import hierarchical_order

matplotlib.use("Agg")  # headless rendering; must be selected before pyplot loads

import matplotlib.pyplot as plt  # noqa: E402


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
    }
    for name, fig in figures.items():
        _save(fig, outdir, name)


if __name__ == "__main__":
    run(main)

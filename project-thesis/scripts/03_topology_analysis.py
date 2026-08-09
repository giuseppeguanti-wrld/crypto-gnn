"""Entry point for Sprint 2 (S2.3/S2.4): the topology and event study of Section 6.6.

Reduces each of the study's rolling correlation windows to the scalars that
describe the market's network structure, then reads those series around the
documented crisis dates of config/events.yaml -- together, the material Section
6.6 is written from and the figures of S2.5 are drawn from.

Reads only artifacts already on disk and recomputes nothing upstream: the
correlation tensor, the graphs, and the calibrated thresholds all come from
02_build_graphs.py. Running it twice yields the same files.

Integration: third script in the pipeline (scripts/01-07). Consumes
data/processed/{corr_*,corr_index,W_full,W_thresh}.npy,
results/metrics/tau_calibration.json and config/events.yaml; produces
results/metrics/topology.parquet and results/metrics/event_study.parquet.

Usage:
    python scripts/03_topology_analysis.py
    python scripts/03_topology_analysis.py --config config/default.yaml
    python scripts/03_topology_analysis.py --events config/events.yaml
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from cryptognn.artifacts import (
    load_corr,
    load_tau,
    load_w_full,
    load_w_thresh,
    save_event_study,
    save_topology,
)
from cryptognn.cli import build_parser, run
from cryptognn.config import load_config
from cryptognn.events import (
    DEFAULT_OFFSETS,
    event_study,
    events_without_citation,
    load_events,
)
from cryptognn.graph.build import apply_threshold, mantegna_distance
from cryptognn.graph.metrics import (
    algebraic_connectivity,
    algebraic_connectivity_combinatorial,
    eigs_outside_mp,
    graph_density,
    market_mode_share,
    mean_correlation,
    mst_length,
    spectral_entropy,
)
from cryptognn.paths import ensure_dirs


def main() -> None:
    parser = build_parser(
        "Compute the topological metric series of the dynamic correlation graph.", events=True
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs()

    window = config.graph.window

    corr, corr_index = load_corr(window)
    w_full = load_w_full()
    w_thresh = load_w_thresh()
    calibration = load_tau()

    n_windows, n_assets, _ = corr.shape
    # q = N / T_w is the aspect ratio governing the Marchenko-Pastur bulk; it
    # comes from the config's window and universe size, never hardcoded.
    q = n_assets / window
    print(
        f"{n_windows} windows x {n_assets} assets, q = {n_assets}/{window} = {q:.3f}, "
        f"MP upper edge = {(1 + np.sqrt(q)) ** 2:.4f}"
    )
    print(
        f"thresholds: tau={calibration.tau:.4f}  tau_fwer={calibration.tau_fwer:.4f}  "
        f"tau_fixed={calibration.tau_fixed:.4f}"
    )

    print("Computing topological metrics...")
    topology = pd.DataFrame(
        {
            "mean_correlation": mean_correlation(corr),
            # Density is the only metric read off the thresholded graph. It is
            # reported at all three thresholds because at the calibrated tau the
            # graph is near-complete (mean 0.974) and the series barely moves;
            # the sparser variants show whether that flatness is a property of
            # the market or of the threshold.
            "graph_density": graph_density(w_thresh),
            "graph_density_fwer": graph_density(
                apply_threshold(corr, w_full, calibration.tau_fwer)
            ),
            "graph_density_fixed": graph_density(
                apply_threshold(corr, w_full, calibration.tau_fixed)
            ),
            # Everything below is computed on the complete graph or on C itself,
            # so no window can be disconnected and lambda_2 is never a spurious 0.
            "algebraic_connectivity": algebraic_connectivity(w_full),
            "algebraic_connectivity_combinatorial": algebraic_connectivity_combinatorial(w_full),
            "mst_length": mst_length(mantegna_distance(corr)),
            "spectral_entropy": spectral_entropy(corr),
            "market_mode_share": market_mode_share(corr),
            "eigs_outside_mp": eigs_outside_mp(corr, q),
        },
        index=corr_index,
    )

    if topology.isna().any().any():
        raise ValueError(f"NaN in topology metrics: {topology.columns[topology.isna().any()].tolist()}")

    print(f"  {topology.shape[0]} rows x {topology.shape[1]} columns")
    for column in topology.columns:
        series = topology[column]
        # cv exposes which series actually carry a signal: at the calibrated tau
        # density, lambda_2 of L_sym, and the MP count are near-degenerate.
        print(
            f"  {column:38s} mean {series.mean():8.4f}  sd {series.std():7.4f}  "
            f"cv {series.std() / abs(series.mean()):6.3f}  "
            f"range [{series.min():.4f}, {series.max():.4f}]"
        )

    # Sign check on the spectral metrics: a compacting market raises the market
    # mode's share and lowers spectral entropy. If either flips sign against
    # mean correlation, an eigenvalue series is inverted.
    mode_rho = topology["market_mode_share"].corr(topology["mean_correlation"], method="spearman")
    entropy_rho = topology["spectral_entropy"].corr(topology["mean_correlation"], method="spearman")
    print(f"  Spearman vs mean_correlation: market_mode_share {mode_rho:+.3f}, spectral_entropy {entropy_rho:+.3f}")
    if mode_rho < 0 or entropy_rho > 0:
        raise ValueError(
            "Unexpected sign: market mode share should rise and spectral entropy fall "
            "as correlation rises -- suspect an inverted eigenvalue series"
        )

    save_topology(topology)
    print("  saved topology.parquet")

    # events.yaml ships with the repository rather than being produced by an
    # earlier step, so its absence is a broken checkout, not a pipeline run out
    # of order: it gets the natural FileNotFoundError, not the "run this first"
    # guard that MissingArtifactError provides for generated artifacts.
    events = load_events(args.events)
    print(f"Event study on {len(events)} events, offsets {list(DEFAULT_OFFSETS)} days...")

    uncited = events_without_citation(events)
    if uncited:
        # A warning, not an error: blocking the pipeline over a bibliographic
        # debt would also block the figures. PLANNING admits only events
        # documentable with a citation, so this must stay visible until settled.
        print(
            f"  WARNING: {len(uncited)} event(s) without a citation: "
            f"{[event.key for event in uncited]}\n"
            "           add a reference to latex-thesis/bibliography/references.bib "
            "before writing Section 6.6"
        )

    study = event_study(topology, events)

    # The headline reading: -60 -> +60 compares two windows sharing no
    # observation, so it is the change that the rolling window cannot manufacture.
    headline = ["mean_correlation", "algebraic_connectivity_combinatorial", "spectral_entropy"]
    for event in events:
        print(f"  {event.key} ({event.date}):")
        for metric in headline:
            row = study[
                (study["event_key"] == event.key)
                & (study["metric"] == metric)
                & (study["offset_days"] == DEFAULT_OFFSETS[-1])
            ].iloc[0]
            print(
                f"    {metric:38s} {row['pct_change_clean']:+7.1f}% over -60->+60, "
                f"percentile at +60: {row['percentile']:.2f}"
            )

    save_event_study(study)
    print(f"  saved event_study.parquet ({len(study)} rows)")


if __name__ == "__main__":
    run(main)

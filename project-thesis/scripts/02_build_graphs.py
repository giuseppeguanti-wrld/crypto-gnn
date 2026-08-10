"""Entry point for Sprint 1 (S1.4/S1.5/S1.6) and Sprint 2's graph construction (S2.1/S2.2).

Full run: build the price panel, validate it, compute log returns and stylized
facts, compute and validate the rolling correlation tensor, calibrate the
correlation threshold tau on a permutation null, and build the three graphs the
study consumes -- the complete Mantegna graph, the thresholded graph, and the
renormalized adjacency that serves as the GCN substrate.

--corr-only stops after the correlation tensor, which is the Sprint 1 scope and
the command named in the M1 definition of done. The full run is the default.

Integration: second script in the pipeline (scripts/01-07). Consumes data/raw/
(from 01_download_data.py) and produces data/processed/{prices,returns,volumes}.parquet,
corr_{window}.npy, corr_index.npy, W_full.npy, W_thresh.npy, A_hat.npy,
A_hat_fwer.npy, plus results/metrics/*.parquet (descriptive stats, ACF,
Ljung-Box) and results/metrics/tau_calibration.json.

Usage:
    python scripts/02_build_graphs.py
    python scripts/02_build_graphs.py --corr-only
    python scripts/02_build_graphs.py --config config/default.yaml
"""
from __future__ import annotations

import dataclasses

import numpy as np

from cryptognn.artifacts import (
    save_corr,
    save_graphs,
    save_prices,
    save_returns,
    save_stylized_facts,
    save_tau,
    save_volumes,
)
from cryptognn.cli import build_parser, run
from cryptognn.config import load_config
from cryptognn.data.returns import build_price_panel, build_volume_panel, log_returns, validate_panel
from cryptognn.data.stylized_facts import (
    check_stylized_facts,
    compute_acf,
    compute_descriptive_stats,
    compute_ljung_box,
)
from cryptognn.graph.build import (
    apply_threshold,
    edge_density,
    mantegna_weights,
    normalized_adjacency,
    validate_adjacency,
    validate_weights,
)
from cryptognn.graph.correlation import rolling_correlation, validate_correlation
from cryptognn.graph.threshold import calibrate_tau, check_tau_plausible
from cryptognn.paths import DATA_RAW, ensure_dirs


def _density_summary(weights: np.ndarray) -> dict[str, float]:
    """Distribution of graph density across windows, for tau_calibration.json."""
    density = edge_density(weights)
    return {
        "mean": float(density.mean()),
        "min": float(density.min()),
        "max": float(density.max()),
        "sd": float(density.std()),
    }


def main() -> None:
    parser = build_parser(
        "Build the price panel, returns, stylized facts, rolling correlations, and graphs."
    )
    parser.add_argument(
        "--corr-only",
        action="store_true",
        help="Stop after the rolling correlation tensor (Sprint 1 scope, the M1 definition "
        "of done). Without it, tau calibration and graph construction also run.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs()

    print("Building price and volume panels...")
    prices = build_price_panel(DATA_RAW, config.data.symbols)
    volumes = build_volume_panel(DATA_RAW, config.data.symbols)
    validate_panel(prices, volumes, config.data.start, config.data.end)
    print(f"  {prices.shape[0]} days x {prices.shape[1]} assets, validated.")

    print("Computing log returns...")
    returns = log_returns(prices)
    print(f"  {returns.shape[0]} days x {returns.shape[1]} assets.")

    save_prices(prices)
    save_returns(returns)
    save_volumes(volumes)
    print("  saved prices.parquet, returns.parquet, volumes.parquet")

    print("Computing stylized facts...")
    descriptive = compute_descriptive_stats(returns)
    acf_returns, acf_abs_returns = compute_acf(returns)
    # TRX: |ACF(r_t)| at lag 1 = 0.144, above the default 0.1 threshold. Investigated
    # 2026-08-08: negative and persistent across every calendar year 2021-2025 (not
    # a one-off event), no evidence of a download artifact (repeated-close rate in
    # line with other assets). Consistent with bid-ask-bounce mean-reversion in a
    # less-liquid asset, not a data error -- accepted as a documented limitation
    # (thesis sec. 7.3) rather than reopening the frozen 15-asset universe.
    check_stylized_facts(descriptive, acf_returns, acf_abs_returns, acf1_return_exceptions={"TRX": 0.2})
    ljung_box_returns, ljung_box_abs_returns = compute_ljung_box(returns)
    print(
        f"  excess kurtosis in [{descriptive['excess_kurtosis'].min():.2f}, "
        f"{descriptive['excess_kurtosis'].max():.2f}], stylized facts checked."
    )

    written = save_stylized_facts(
        descriptive, acf_returns, acf_abs_returns, ljung_box_returns, ljung_box_abs_returns
    )
    print(f"  saved {', '.join(path.name for path in written)}")

    window = config.graph.window
    print(f"Computing rolling correlation (window={window})...")
    corr, corr_index = rolling_correlation(returns, window)
    validate_correlation(corr)
    print(f"  {corr.shape[0]} windows, shape {corr.shape}, validated.")

    save_corr(corr, corr_index, window)
    print(f"  saved corr_{window}.npy, corr_index.npy")

    if args.corr_only:
        print("--corr-only: stopping before graph construction.")
        return

    threshold_config = config.graph.threshold
    print(
        f"Calibrating tau (permutation null, alpha={threshold_config.alpha}, "
        f"B={threshold_config.n_permutations}, "
        f"{threshold_config.n_calibration_windows} windows)..."
    )
    calibration = calibrate_tau(returns, config)
    check_tau_plausible(calibration)
    print(
        f"  tau={calibration.tau:.4f}  tau_fwer={calibration.tau_fwer:.4f}  "
        f"tau_fixed={calibration.tau_fixed:.4f}"
    )

    print("Building graphs...")
    w_full = mantegna_weights(corr)
    validate_weights(w_full)

    w_thresh = apply_threshold(corr, w_full, calibration.tau)
    validate_weights(w_thresh)

    # A_hat at the calibrated tau is the study substrate. A_hat_fwer is a
    # pre-registered robustness variant, decided before any test result was seen:
    # the calibrated threshold leaves a near-complete graph (density ~0.97), so
    # the Sprint 4 no-graph ablation would otherwise be compared against an
    # almost uniform average over nodes. The FWER threshold gives a genuinely
    # sparser substrate to repeat the comparison on, without reopening tau
    # after the fact.
    a_hat = normalized_adjacency(w_thresh, self_loops=config.graph.self_loops)
    validate_adjacency(a_hat)

    w_thresh_fwer = apply_threshold(corr, w_full, calibration.tau_fwer)
    a_hat_fwer = normalized_adjacency(w_thresh_fwer, self_loops=config.graph.self_loops)
    validate_adjacency(a_hat_fwer)

    w_thresh_fixed = apply_threshold(corr, w_full, calibration.tau_fixed)
    calibration = dataclasses.replace(
        calibration,
        density={
            "tau": _density_summary(w_thresh),
            "tau_fwer": _density_summary(w_thresh_fwer),
            "tau_fixed": _density_summary(w_thresh_fixed),
        },
    )
    for name, summary in calibration.density.items():
        print(
            f"  density at {name:9s}: mean {summary['mean']:.3f}  "
            f"min {summary['min']:.3f}  max {summary['max']:.3f}  sd {summary['sd']:.3f}"
        )
    isolated = int((w_thresh.sum(axis=-1) == 0).any(axis=-1).sum())
    print(f"  {isolated} of {len(w_thresh)} windows contain an isolated node at the calibrated tau")

    save_graphs(w_full, w_thresh, a_hat, a_hat_fwer)
    print("  saved W_full.npy, W_thresh.npy, A_hat.npy, A_hat_fwer.npy")

    save_tau(calibration)
    print("  saved tau_calibration.json")


if __name__ == "__main__":
    run(main)

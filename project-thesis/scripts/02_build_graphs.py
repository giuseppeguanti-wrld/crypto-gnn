"""Entry point for Sprint 1 (S1.4/S1.5/S1.6) and Sprint 2's graph construction.

--corr-only restricts this script to the Sprint 1 scope: build the price panel,
validate it, compute log returns, compute stylized facts (sanity-checking the
returns), compute the rolling correlation tensor, validate it, and save every
artifact downstream steps depend on. Full graph construction (Mantegna weights,
thresholding, renormalized adjacency -- Sprint 2) will extend this same script;
not implemented yet, hence the flag is required.

Integration: second script in the pipeline (scripts/01-07). Consumes data/raw/
(from 01_download_data.py) and produces data/processed/{prices,returns}.parquet,
data/processed/corr_{window}.npy, corr_index.npy, and results/metrics/*.parquet
(descriptive stats, ACF, Ljung-Box).

Usage:
    python scripts/02_build_graphs.py --corr-only
    python scripts/02_build_graphs.py --corr-only --config config/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from cryptognn.config import load_config  # noqa: E402
from cryptognn.data.returns import build_price_panel, log_returns, validate_panel  # noqa: E402
from cryptognn.data.stylized_facts import (  # noqa: E402
    check_stylized_facts,
    compute_acf,
    compute_descriptive_stats,
    compute_ljung_box,
)
from cryptognn.graph.correlation import rolling_correlation, validate_correlation  # noqa: E402
from cryptognn.paths import DATA_PROCESSED, DATA_RAW, RESULTS_METRICS, ensure_dirs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the price panel, returns, stylized facts, and rolling correlation tensor."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Path to the study config YAML (default: config/default.yaml).",
    )
    parser.add_argument(
        "--corr-only",
        action="store_true",
        required=True,
        help="Sprint 1 scope: panel + returns + stylized facts + rolling correlation only. "
        "Full graph construction (Sprint 2) is not implemented yet.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs()

    print("Building price panel...")
    prices = build_price_panel(DATA_RAW, config.data.symbols)
    validate_panel(prices, DATA_RAW, config.data.start, config.data.end)
    print(f"  {prices.shape[0]} days x {prices.shape[1]} assets, validated.")

    print("Computing log returns...")
    returns = log_returns(prices)
    print(f"  {returns.shape[0]} days x {returns.shape[1]} assets.")

    prices.to_parquet(DATA_PROCESSED / "prices.parquet")
    returns.to_parquet(DATA_PROCESSED / "returns.parquet")
    print("  saved prices.parquet, returns.parquet")

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

    descriptive.to_parquet(RESULTS_METRICS / "descriptive.parquet")
    acf_returns.to_parquet(RESULTS_METRICS / "acf_returns.parquet")
    acf_abs_returns.to_parquet(RESULTS_METRICS / "acf_abs_returns.parquet")
    ljung_box_returns.to_parquet(RESULTS_METRICS / "ljung_box_returns.parquet")
    ljung_box_abs_returns.to_parquet(RESULTS_METRICS / "ljung_box_abs_returns.parquet")
    print(
        "  saved descriptive.parquet, acf_returns.parquet, acf_abs_returns.parquet, "
        "ljung_box_returns.parquet, ljung_box_abs_returns.parquet"
    )

    window = config.graph.window
    print(f"Computing rolling correlation (window={window})...")
    corr, corr_index = rolling_correlation(returns, window)
    validate_correlation(corr)
    print(f"  {corr.shape[0]} windows, shape {corr.shape}, validated.")

    np.save(DATA_PROCESSED / f"corr_{window}.npy", corr)
    np.save(DATA_PROCESSED / "corr_index.npy", corr_index.tz_localize(None).to_numpy())
    print(f"  saved corr_{window}.npy, corr_index.npy")


if __name__ == "__main__":
    main()

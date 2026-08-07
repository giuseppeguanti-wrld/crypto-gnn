"""Entry point for Sprint 1 S1.3: download the 15-asset universe from Binance.

Loads a study config (config/default.yaml by default), makes sure the on-disk
directory tree exists, then calls download_universe() to populate data/raw/
with one cached Parquet file per symbol (BTCUSDT_1d.parquet, ETHUSDT_1d.parquet, ...).

Idempotent by default: symbols whose cache already covers the configured date
range are not re-downloaded. --force ignores the cache and re-downloads everything,
for when a symbol's history needs refreshing (e.g. the configured end date moved
forward) or the cached data is suspected corrupt.

Integration: first script in the pipeline (scripts/01-07); its output (data/raw/)
is consumed by cryptognn.data.returns.build_price_panel() in the next step (S1.4).

Usage:
    python scripts/01_download_data.py
    python scripts/01_download_data.py --config config/default.yaml --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cryptognn.config import load_config  # noqa: E402
from cryptognn.data.download import download_universe  # noqa: E402
from cryptognn.paths import ensure_dirs  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the crypto-gnn asset universe from Binance.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="Path to the study config YAML (default: config/default.yaml).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download every symbol, ignoring the on-disk cache in data/raw/.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs()

    data = download_universe(config, force=args.force)

    for symbol, df in data.items():
        print(f"{symbol}: {len(df)} rows, {df.index.min().date()} -> {df.index.max().date()}")


if __name__ == "__main__":
    main()

"""Entry point for Sprint 1 S1.3: download the 15-asset universe from Binance.

Loads a study config (config/default.yaml by default), makes sure the on-disk
directory tree exists, then calls download_universe() to populate data/raw/
with one cached Parquet file per symbol (BTCUSDT_1d.parquet, ETHUSDT_1d.parquet, ...).

Idempotent by default: symbols whose cache already covers the configured date
range are not re-downloaded. --force ignores the cache and re-downloads everything,
for when a symbol's history needs refreshing (e.g. the configured end date moved
forward) or the cached data is suspected corrupt.

Integration: first script in the pipeline (scripts/01-08); its output (data/raw/)
is consumed by cryptognn.data.returns.build_price_panel() in the next step (S1.4).

Usage:
    python scripts/01_download_data.py
    python scripts/01_download_data.py --config config/default.yaml --force
"""
from __future__ import annotations

from cryptognn.cli import build_parser, run
from cryptognn.config import load_config
from cryptognn.data.download import download_universe
from cryptognn.paths import ensure_dirs


def main() -> None:
    parser = build_parser("Download the crypto-gnn asset universe from Binance.")
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
    run(main)

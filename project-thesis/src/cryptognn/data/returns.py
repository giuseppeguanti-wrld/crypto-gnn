"""Price panel construction and return computation for the crypto-gnn study.

Turns the per-symbol cached klines in data/raw/ (written by cryptognn.data.download)
into the two artifacts every downstream step depends on: a wide price panel and a
log-return panel, both indexed by UTC date with one column per asset.

Exports:
  - build_price_panel(): joins each symbol's cached close price into one wide DataFrame
  - build_volume_panel(): the same, for traded volume
  - validate_panel(): raises on gaps, NaNs, non-positive prices/volumes, misaligned start/end
  - log_returns(): r_t = log P_t - log P_{t-1}, first row dropped

Integration: called by scripts/02_build_graphs.py; its output
  (prices.parquet, returns.parquet) feeds cryptognn.graph.correlation and cryptognn.features.
Why it exists: keeps "raw klines -> clean aligned panel" logic in one place, so every
  downstream module can assume a validated, gap-free (T, N) matrix and never touch
  data/raw/ directly.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


def build_price_panel(raw_dir: Path | str, symbols: list[str], quote: str = "USDT") -> pd.DataFrame:
    """Join the cached close-price series of every symbol into one wide panel.

    Reads data/raw/{SYMBOL}{quote}_1d.parquet for each symbol (as cached by
    cryptognn.data.download.fetch_klines, quote defaulting to "USDT" to match
    config.data.quote) and extracts its `close` column. The resulting DataFrame
    has one column per symbol (in the given order) and a UTC date index that is
    the union of all symbols' dates -- any symbol missing a given date gets NaN
    there. No alignment, filling, or interpolation happens here: validate_panel()
    (next step) is responsible for catching and rejecting gaps.
    """
    raw_dir = Path(raw_dir)
    closes = {symbol: pd.read_parquet(raw_dir / f"{symbol}{quote}_1d.parquet")["close"] for symbol in symbols}
    return pd.DataFrame(closes)


def build_volume_panel(raw_dir: Path | str, symbols: list[str], quote: str = "USDT") -> pd.DataFrame:
    """Join the cached traded volume of every symbol into one wide panel.

    Structurally identical to build_price_panel(), on the `volume` column: the
    base-asset volume (coins traded), not `quote_volume` (turnover in USDT).
    Quote volume is roughly volume x price, so its log carries the price path,
    and the twenty-day z-score of Section 6.3's eighth node feature would then
    partly re-encode the momentum already present in the five lagged returns.
    Base volume measures trading activity and nothing else; taking logs makes
    the z-score scale-free, so DOGE trading in billions of units and BTC in
    thousands are directly comparable.

    Returned on the price index (T = 2007 for this study), one row longer than
    the return panel: the alignment happens in cryptognn.features, where the
    two are used together.
    """
    raw_dir = Path(raw_dir)
    volumes = {symbol: pd.read_parquet(raw_dir / f"{symbol}{quote}_1d.parquet")["volume"] for symbol in symbols}
    return pd.DataFrame(volumes)


def validate_panel(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    start: date | str,
    end: date | str,
) -> None:
    """Validate the price and volume panels together, raising on the first
    violation found rather than merely reporting it -- a bad panel must never
    silently reach the rest of the pipeline. Checks, in order:

      1. Alignment: the two panels carry the same dates and the same symbols, in
         the same order. They are built from the same files by
         build_price_panel() and build_volume_panel(), so a mismatch means one of
         the two was built from a different universe or a different cache.
      2. Continuity: the index has no internal gaps -- it matches
         pd.date_range(min, max, freq="D") exactly. Crypto trades every day, so
         any gap is a data problem, not a holiday.
      3. Completeness: zero NaN anywhere. A NaN means some symbol is missing a
         candle on a date every other symbol has; the fix is to reconsider the
         universe (drop or replace that symbol), never to interpolate.
      4. Sanity: no price <= 0, and no zero-volume bar for any symbol.
      5. Scope: the panel's first and last date match `start`/`end` exactly --
         independent of check 2, since a fully continuous panel could still cover
         the wrong period.

    Takes the volume panel rather than the directory it came from. The earlier
    signature accepted `raw_dir` and re-read all fifteen per-symbol Parquet files
    to reach the volume column, which meant a validation function performed I/O,
    read the same files the caller was about to read again, and could not be
    exercised without a filesystem. Since build_volume_panel() exists the caller
    already holds what this needs.
    """
    start_ts = pd.Timestamp(start, tz="UTC").normalize()
    end_ts = pd.Timestamp(end, tz="UTC").normalize()

    if not prices.index.equals(volumes.index):
        raise ValueError(
            f"Price and volume panels differ in dates: {len(prices)} vs {len(volumes)} rows, "
            "so they were not built from the same cache"
        )
    if list(prices.columns) != list(volumes.columns):
        raise ValueError(
            f"Price and volume panels differ in symbols: {list(prices.columns)} vs {list(volumes.columns)}"
        )

    continuous_index = pd.date_range(prices.index.min(), prices.index.max(), freq="D", tz="UTC")
    if not prices.index.equals(continuous_index):
        missing = continuous_index.difference(prices.index)
        raise ValueError(f"Panel index has gaps -- missing dates: {list(missing.date)}")

    for name, panel in (("Price", prices), ("Volume", volumes)):
        if panel.isna().any().any():
            nan_report = {
                symbol: panel.index[panel[symbol].isna()].date.tolist()
                for symbol in panel.columns
                if panel[symbol].isna().any()
            }
            raise ValueError(
                f"{name} panel contains NaN values -- reconsider the universe, never interpolate: {nan_report}"
            )

    non_positive = {
        symbol: prices.index[prices[symbol] <= 0].date.tolist()
        for symbol in prices.columns
        if (prices[symbol] <= 0).any()
    }
    if non_positive:
        raise ValueError(f"Panel contains non-positive price(s): {non_positive}")

    zero_volume = {
        symbol: volumes.index[volumes[symbol] == 0].date.tolist()
        for symbol in volumes.columns
        if (volumes[symbol] == 0).any()
    }
    if zero_volume:
        raise ValueError(f"Zero-volume bar(s) found: {zero_volume}")

    if prices.index.min() != start_ts or prices.index.max() != end_ts:
        raise ValueError(
            f"Panel range [{prices.index.min().date()}, {prices.index.max().date()}] does not match "
            f"expected [{start_ts.date()}, {end_ts.date()}]"
        )


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log returns r_t = log(P_t) - log(P_{t-1}) for every asset.

    The first row has no prior price to diff against and is dropped, so a
    (T, N) price panel produces a (T-1, N) return panel -- e.g. the study's
    (2007, 15) price panel becomes a (2006, 15) return panel. Assumes `prices`
    has already passed validate_panel(): no NaN, no non-positive values.
    """
    return np.log(prices).diff().iloc[1:]

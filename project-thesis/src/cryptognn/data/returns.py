"""Price panel construction and return computation for the crypto-gnn study.

Turns the per-symbol cached klines in data/raw/ (written by cryptognn.data.download)
into the two artifacts every downstream step depends on: a wide price panel and a
log-return panel, both indexed by UTC date with one column per asset.

Exports (built incrementally):
  - build_price_panel(): joins each symbol's cached close price into one wide DataFrame
  - validate_panel(): raises on gaps, NaNs, non-positive prices/volumes, misaligned start/end
  - log_returns(): r_t = log P_t - log P_{t-1}, first row dropped

Integration: called by scripts/02_build_graphs.py (per PLANNING.md M1 DoD); its output
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


def build_price_panel(raw_dir: Path | str, symbols: list[str]) -> pd.DataFrame:
    """Join the cached close-price series of every symbol into one wide panel.

    Reads data/raw/{SYMBOL}USDT_1d.parquet for each symbol (as cached by
    cryptognn.data.download.fetch_klines) and extracts its `close` column. The
    resulting DataFrame has one column per symbol (in the given order) and a
    UTC date index that is the union of all symbols' dates -- any symbol missing
    a given date gets NaN there. No alignment, filling, or interpolation happens
    here: validate_panel() (next step) is responsible for catching and rejecting gaps.
    """
    raw_dir = Path(raw_dir)
    closes = {symbol: pd.read_parquet(raw_dir / f"{symbol}USDT_1d.parquet")["close"] for symbol in symbols}
    return pd.DataFrame(closes)


def validate_panel(
    df: pd.DataFrame,
    raw_dir: Path | str,
    start: date | str,
    end: date | str,
) -> None:
    """Validate a price panel built by build_price_panel(), raising on the first
    violation found rather than merely reporting it -- a bad panel must never
    silently reach the rest of the pipeline. Checks, in order:

      1. Continuity: the index has no internal gaps -- it matches
         pd.date_range(df.index.min(), df.index.max(), freq="D") exactly. Crypto
         trades every day, so any gap is a data problem, not a holiday.
      2. Completeness: zero NaN anywhere. A NaN means some symbol is missing a
         candle on a date every other symbol has; the fix is to reconsider the
         universe (drop or replace that symbol), never to interpolate.
      3. Sanity: no price <= 0, and no zero-volume bar for any symbol (volume is
         read from the raw per-symbol Parquet cache in raw_dir, since
         build_price_panel() keeps only the close column).
      4. Scope: the panel's first and last date match `start`/`end` exactly --
         independent of check 1, since a fully continuous panel could still cover
         the wrong period.

    `raw_dir` must be the same directory build_price_panel() read from.
    """
    raw_dir = Path(raw_dir)
    start_ts = pd.Timestamp(start, tz="UTC").normalize()
    end_ts = pd.Timestamp(end, tz="UTC").normalize()

    continuous_index = pd.date_range(df.index.min(), df.index.max(), freq="D", tz="UTC")
    if not df.index.equals(continuous_index):
        missing = continuous_index.difference(df.index)
        raise ValueError(f"Panel index has gaps -- missing dates: {list(missing.date)}")

    if df.isna().any().any():
        nan_report = {
            symbol: df.index[df[symbol].isna()].date.tolist() for symbol in df.columns if df[symbol].isna().any()
        }
        raise ValueError(
            f"Panel contains NaN values -- reconsider the universe, never interpolate: {nan_report}"
        )

    non_positive = {symbol: df.index[df[symbol] <= 0].date.tolist() for symbol in df.columns if (df[symbol] <= 0).any()}
    if non_positive:
        raise ValueError(f"Panel contains non-positive price(s): {non_positive}")

    zero_volume: dict[str, list] = {}
    for symbol in df.columns:
        volume = pd.read_parquet(raw_dir / f"{symbol}USDT_1d.parquet")["volume"]
        zero_dates = volume.index[volume == 0]
        if len(zero_dates) > 0:
            zero_volume[symbol] = list(zero_dates.date)
    if zero_volume:
        raise ValueError(f"Zero-volume bar(s) found: {zero_volume}")

    if df.index.min() != start_ts or df.index.max() != end_ts:
        raise ValueError(
            f"Panel range [{df.index.min().date()}, {df.index.max().date()}] does not match "
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

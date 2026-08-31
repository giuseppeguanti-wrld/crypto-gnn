"""Binance market data download for the crypto-gnn study.

Fetches daily OHLCV klines for the 15-asset universe from Binance's public REST API
(no API key required) and caches them to disk as Parquet.

Exports (built incrementally):
  - _fetch_klines_batch(): single GET request to /api/v3/klines
  - _fetch_klines_paginated(): loops _fetch_klines_batch() across startTime until end is reached
  - fetch_klines(): full download for one symbol, DataFrame + on-disk Parquet cache
  - download_universe(): iterates fetch_klines() over every symbol in config.data.symbols

Integration: Called by scripts/01_download_data.py; output feeds cryptognn.data.returns.
Why it exists: Binance klines are the sole price source for the study; isolating
  the HTTP concern here keeps returns.py free of network code.
"""
from __future__ import annotations

import time
from datetime import date

import pandas as pd
import requests
from tqdm import tqdm

from cryptognn.config import Config
from cryptognn.paths import DATA_RAW

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"

_KLINE_COLUMNS = [
    "open_time",        # The exact moment the candle started to form
    "open",             # The very first price at which a trade was executed at the beginning of this time interval
    "high",             # The absolute highest price reached by the asset during this timeframe
    "low",              # The absolute lowest price reached
    "close",            # The very last traded price before the timeframe expired
    "volume",           # The total amount of the asset you are trading (the "base asset") that changed hands in this timeframe
    "close_time",       # The exact moment the candle closes, in milliseconds
    "quote_volume",     # The total amount of the asset you use to pay (the "quote asset") traded in this timeframe
    "trades",           # The total count of individual transactions (executed orders) that took place between buyers and sellers to form this candle
    "taker_buy_base",   # The aggressive buying pressure in the base asset
    "taker_buy_quote",  # It shows how many USDT were spent by aggressive market buyers
    "ignore",           # This field was historically used for Binance's internal purposes
]


def _fetch_klines_batch(
    symbol: str,
    interval: str,
    start_time_ms: int,
    limit: int = 1000,
) -> list[list]:
    """Single GET request to Binance's /api/v3/klines endpoint.

    Returns the raw list of klines as given by Binance (each kline is a list of
    12 fields: open_time, open, high, low, close, volume, close_time, quote_volume,
    trades, taker_buy_base, taker_buy_quote, ignore). No pagination, parsing into a
    DataFrame, or caching yet -- those are separate steps.
    """
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time_ms,
        "limit": limit,
    }
    response = requests.get(BINANCE_KLINES_URL, params=params)
    response.raise_for_status()
    return response.json()


def _fetch_klines_paginated(
    symbol: str,
    interval: str,
    start_time_ms: int,
    end_time_ms: int,
    limit: int = 1000,
) -> list[list]:
    """Repeatedly calls _fetch_klines_batch(), advancing startTime after each call,
    until the last candle received is at or past end_time_ms (or Binance returns no
    more data). Sleeps 0.15s between calls to stay well within Binance's public rate
    limits. Still returns raw klines -- DataFrame construction and caching come later.
    """
    all_klines: list[list] = []
    current_start = start_time_ms

    while True:
        batch = _fetch_klines_batch(symbol, interval, current_start, limit=limit)
        if not batch:
            break

        all_klines.extend(batch)
        last_open_time = batch[-1][0]

        if last_open_time >= end_time_ms:
            break

        current_start = last_open_time + 1
        time.sleep(0.15)

    return all_klines


def fetch_klines(
    symbol: str,
    start: date | str,
    end: date | str,
    interval: str,
    quote: str = "USDT",
    force: bool = False,
) -> pd.DataFrame:
    """Download daily OHLCV history for one asset and return it as a tidy DataFrame.

    `symbol` is the base asset (e.g. "BTC"); `quote` is the asset it is priced
    in (config.data.quote, "USDT" by default here to match the frozen decision
    to trade only /USDT pairs -- see download_universe()). Returns a DataFrame
    indexed by open_time (UTC, normalized to midnight) with columns close,
    volume, quote_volume, trades, trimmed to exactly [start, end].

    Caches to data/raw/{pair}_{interval}.parquet: if that file already exists
    and its date range covers [start, end], the cached data is returned and no
    network call is made. Otherwise the full range is (re)downloaded and the
    cache file is overwritten. `force=True` skips the cache check unconditionally
    (used by scripts/01_download_data.py --force).
    """
    pair = f"{symbol}{quote}"
    start_ts = pd.Timestamp(start, tz="UTC").normalize()
    end_ts = pd.Timestamp(end, tz="UTC").normalize()

    cache_path = DATA_RAW / f"{pair}_{interval}.parquet"
    if not force and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        if cached.index.min() <= start_ts and cached.index.max() >= end_ts:
            return cached.loc[start_ts:end_ts]

    raw = _fetch_klines_paginated(
        pair,
        interval,
        int(start_ts.timestamp() * 1000),
        int(end_ts.timestamp() * 1000),
    )

    df = pd.DataFrame(raw, columns=_KLINE_COLUMNS)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.normalize()
    df = df.set_index("open_time")

    df["close"] = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    df["quote_volume"] = df["quote_volume"].astype(float)
    df["trades"] = df["trades"].astype(int)

    df = df.loc[start_ts:end_ts, ["close", "volume", "quote_volume", "trades"]]

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)

    return df


def download_universe(config: Config, force: bool = False) -> dict[str, pd.DataFrame]:
    """Download (or load from cache) daily OHLCV data for every symbol in the
    configured universe (config.data.symbols), over config.data.start/end at
    config.data.interval. Shows a tqdm progress bar since sequential rate-limited
    downloads of 15 symbols take noticeably longer than a single fetch_klines() call.
    `force=True` is forwarded to fetch_klines() to bypass the on-disk cache.

    Returns a dict mapping each base symbol (e.g. "BTC") to its DataFrame, in the
    same shape produced by fetch_klines().
    """
    return {
        symbol: fetch_klines(
            symbol, config.data.start, config.data.end, config.data.interval, quote=config.data.quote, force=force
        )
        for symbol in tqdm(config.data.symbols, desc="Downloading universe")
    }

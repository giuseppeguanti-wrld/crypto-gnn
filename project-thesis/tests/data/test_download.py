"""Tests for cryptognn.data.download: pagination, parsing and the on-disk cache.

No network. `requests.get` is replaced with a stub that serves synthetic klines
in Binance's own list-of-lists shape, so the pagination loop, the type coercion
and the cache policy are exercised for real while the suite stays runnable on a
fresh clone with no connectivity -- which is what tests/conftest.py promises.

The cache is the part worth testing hardest. It is what makes the pipeline
re-runnable at all (`scripts/01` is otherwise a two-minute download), and a cache
that returns data outside the requested range, or refuses to refresh when the
range grows, would corrupt the price panel without ever raising.
"""
from __future__ import annotations

import itertools

import pandas as pd
import pytest

from cryptognn.config import load_config
from cryptognn.data import download
from cryptognn.paths import DEFAULT_CONFIG

SYMBOL = "BTC"
INTERVAL = "1d"
START, END = "2021-01-01", "2021-01-10"
DAY_MS = 86_400_000


def kline(open_time_ms: int, close: float) -> list:
    """One kline in Binance's twelve-field list form."""
    return [
        open_time_ms,
        "100.0",
        "110.0",
        "90.0",
        f"{close}",
        "1234.5",
        open_time_ms + DAY_MS - 1,
        "98765.4",
        4321,
        "600.0",
        "50000.0",
        "0",
    ]


class FakeResponse:
    def __init__(self, payload: list) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> list:
        return self._payload


@pytest.fixture
def klines_server(monkeypatch):
    """A stub Binance that serves `page_size` candles per request, from startTime.

    Records every request it received, so a test can assert how the loop advanced
    rather than only what it ended up with.
    """
    calls: list[dict] = []
    state = {"page_size": 1000, "last_open_ms": int(pd.Timestamp("2021-02-01", tz="UTC").timestamp() * 1000)}

    def fake_get(url, params=None, **kwargs):
        calls.append(dict(params))
        start = params["startTime"]
        # Align to midnight so the parsed index normalizes cleanly.
        first = ((start + DAY_MS - 1) // DAY_MS) * DAY_MS if start % DAY_MS else start
        payload = [
            kline(first + i * DAY_MS, 100.0 + i)
            for i in range(state["page_size"])
            if first + i * DAY_MS <= state["last_open_ms"]
        ]
        return FakeResponse(payload)

    monkeypatch.setattr(download.requests, "get", fake_get)
    monkeypatch.setattr(download.time, "sleep", lambda _seconds: None)
    return calls, state


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """Redirect the module's DATA_RAW, which it imported by value."""
    monkeypatch.setattr(download, "DATA_RAW", tmp_path)
    return tmp_path


class TestPagination:
    def test_advances_start_time_past_the_last_candle_received(self, klines_server):
        calls, state = klines_server
        state["page_size"] = 3
        start_ms = int(pd.Timestamp(START, tz="UTC").timestamp() * 1000)
        end_ms = int(pd.Timestamp("2021-01-09", tz="UTC").timestamp() * 1000)

        klines = download._fetch_klines_paginated(SYMBOL, INTERVAL, start_ms, end_ms)

        assert len(calls) > 1, "a three-candle page cannot cover nine days in one request"
        # Each request starts strictly after the previous batch's last candle, so
        # no candle is fetched twice and none is skipped.
        for previous, current in itertools.pairwise(calls):
            assert current["startTime"] > previous["startTime"]
        open_times = [row[0] for row in klines]
        assert open_times == sorted(open_times)
        assert open_times[-1] >= end_ms

    def test_stops_on_an_empty_batch(self, klines_server, monkeypatch):
        """Binance returns [] past the end of a symbol's history; without this
        exit the loop would spin forever on a delisted or mistyped pair.
        """
        monkeypatch.setattr(download.requests, "get", lambda *a, **k: FakeResponse([]))

        assert download._fetch_klines_paginated(SYMBOL, INTERVAL, 0, DAY_MS * 10) == []

    def test_passes_the_symbol_and_interval_through(self, klines_server):
        calls, _ = klines_server
        download._fetch_klines_paginated("ETH", "1h", 0, DAY_MS)

        assert calls[0]["symbol"] == "ETH"
        assert calls[0]["interval"] == "1h"


class TestFetchKlines:
    def test_returns_typed_columns_trimmed_to_the_requested_range(self, klines_server, cache_dir):
        frame = download.fetch_klines(SYMBOL, START, END, INTERVAL)

        assert list(frame.columns) == ["close", "volume", "quote_volume", "trades"]
        assert frame.index.min() == pd.Timestamp(START, tz="UTC")
        assert frame.index.max() == pd.Timestamp(END, tz="UTC")
        # Binance sends every numeric field as a string; leaving them that way
        # would make log(price) fail much later, in the return computation.
        assert frame["close"].dtype.kind == "f"
        assert frame["volume"].dtype.kind == "f"
        assert frame["trades"].dtype.kind == "i"

    def test_writes_the_cache_under_the_pair_name(self, klines_server, cache_dir):
        download.fetch_klines(SYMBOL, START, END, INTERVAL)

        assert (cache_dir / f"{SYMBOL}USDT{'_'}{INTERVAL}.parquet").exists()

    def test_a_cache_covering_the_range_prevents_any_request(self, klines_server, cache_dir):
        calls, _ = klines_server
        download.fetch_klines(SYMBOL, START, END, INTERVAL)
        before = len(calls)

        again = download.fetch_klines(SYMBOL, START, END, INTERVAL)

        assert len(calls) == before, "the second call must be served entirely from disk"
        assert again.index.min() == pd.Timestamp(START, tz="UTC")

    def test_a_cache_that_falls_short_triggers_a_refresh(self, klines_server, cache_dir):
        """The failure mode this prevents: extending the study period and silently
        getting back the old, shorter history instead of the new data.
        """
        calls, _ = klines_server
        download.fetch_klines(SYMBOL, START, "2021-01-05", INTERVAL)
        before = len(calls)

        extended = download.fetch_klines(SYMBOL, START, "2021-01-20", INTERVAL)

        assert len(calls) > before
        assert extended.index.max() == pd.Timestamp("2021-01-20", tz="UTC")

    def test_force_bypasses_a_valid_cache(self, klines_server, cache_dir):
        calls, _ = klines_server
        download.fetch_klines(SYMBOL, START, END, INTERVAL)
        before = len(calls)

        download.fetch_klines(SYMBOL, START, END, INTERVAL, force=True)

        assert len(calls) > before


class TestDownloadUniverse:
    def test_covers_every_configured_symbol(self, klines_server, cache_dir):
        config = load_config(DEFAULT_CONFIG)

        universe = download.download_universe(config)

        assert list(universe) == config.data.symbols
        assert all(isinstance(frame, pd.DataFrame) and not frame.empty for frame in universe.values())

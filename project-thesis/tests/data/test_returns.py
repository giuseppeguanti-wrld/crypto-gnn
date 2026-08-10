"""Tests for cryptognn.data.returns: the panel builders and the gate that guards them.

`validate_panel()` is the only thing standing between a defect in the downloaded
data and every number the study produces. It is written to **raise**, so the
tests that matter are the ones that hand it a broken panel and demand that it
does: a guard nobody has watched fail is not a guard.

Each violation is injected on its own, into an otherwise valid pair of panels, so
a failure identifies the check that fired rather than merely reporting that
something was wrong. That is the same discipline the anti-look-ahead tests of
Sprint 3 follow, applied to data ingestion, which had none until now.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptognn.data.returns import build_price_panel, build_volume_panel, log_returns, validate_panel

SYMBOLS = ["BTC", "ETH", "SOL"]
START, END = "2021-01-01", "2021-01-10"
N_DAYS = 10


def raw_frame(seed: int, n_days: int = N_DAYS, start: str = START) -> pd.DataFrame:
    """One symbol's cached klines, in the shape fetch_klines() writes them."""
    rng = np.random.default_rng(seed)
    index = pd.date_range(start, periods=n_days, freq="D", tz="UTC", name="open_time")
    return pd.DataFrame(
        {
            "close": 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, n_days))),
            "volume": rng.uniform(1e3, 1e5, n_days),
            "quote_volume": rng.uniform(1e5, 1e7, n_days),
            "trades": rng.integers(100, 10_000, n_days),
        },
        index=index,
    )


@pytest.fixture
def raw_dir(tmp_path):
    """A data/raw/ directory holding one Parquet per symbol."""
    for offset, symbol in enumerate(SYMBOLS):
        raw_frame(seed=offset).to_parquet(tmp_path / f"{symbol}USDT_1d.parquet")
    return tmp_path


@pytest.fixture
def panels(raw_dir):
    return build_price_panel(raw_dir, SYMBOLS), build_volume_panel(raw_dir, SYMBOLS)


class TestPanelBuilders:
    def test_columns_follow_the_requested_order(self, raw_dir):
        """Column order is the universe order from the config, not the directory's.

        Every downstream array is positional -- returns, features and the graph
        share an asset axis -- so a panel built in filesystem order would silently
        relabel every asset.
        """
        reversed_order = list(reversed(SYMBOLS))

        assert list(build_price_panel(raw_dir, SYMBOLS).columns) == SYMBOLS
        assert list(build_price_panel(raw_dir, reversed_order).columns) == reversed_order
        assert list(build_volume_panel(raw_dir, SYMBOLS).columns) == SYMBOLS

    def test_price_and_volume_panels_share_an_index(self, panels):
        prices, volumes = panels

        assert prices.shape == volumes.shape == (N_DAYS, len(SYMBOLS))
        assert prices.index.equals(volumes.index)
        assert prices.index.tz is not None

    def test_a_symbol_missing_a_date_becomes_nan_rather_than_a_shifted_column(self, raw_dir):
        """No alignment, filling or interpolation happens in the builder.

        A short series must produce NaN on the dates it lacks -- which
        validate_panel() then rejects -- and not quietly shift its values up to
        fill the panel, which would misalign that asset's whole history.
        """
        raw_frame(seed=9, n_days=N_DAYS - 3).to_parquet(raw_dir / "ETHUSDT_1d.parquet")

        prices = build_price_panel(raw_dir, SYMBOLS)

        assert prices["ETH"].isna().sum() == 3
        assert prices["ETH"].iloc[:-3].notna().all()  # the gap is at the end, where it belongs


class TestLogReturns:
    def test_first_row_is_dropped_and_values_are_log_differences(self, panels):
        prices, _ = panels

        returns = log_returns(prices)

        assert returns.shape == (N_DAYS - 1, len(SYMBOLS))
        assert returns.index.equals(prices.index[1:])
        expected = np.log(prices["BTC"].iloc[1]) - np.log(prices["BTC"].iloc[0])
        assert returns["BTC"].iloc[0] == pytest.approx(expected)

    def test_a_constant_price_gives_exactly_zero(self):
        index = pd.date_range(START, periods=4, freq="D", tz="UTC")
        flat = pd.DataFrame({"BTC": [50.0] * 4}, index=index)

        assert (log_returns(flat)["BTC"] == 0.0).all()


class TestValidatePanel:
    """One injected violation per test, into an otherwise valid pair of panels."""

    def test_accepts_a_well_formed_pair(self, panels):
        prices, volumes = panels

        validate_panel(prices, volumes, START, END)  # must not raise

    def test_rejects_a_gap_in_the_index(self, panels):
        """Crypto trades every calendar day, so a missing date is a data defect,
        never a holiday -- the assumption the whole daily pipeline rests on.
        """
        prices, volumes = panels
        dropped = prices.index[4]

        with pytest.raises(ValueError, match="index has gaps"):
            validate_panel(prices.drop(dropped), volumes.drop(dropped), START, END)

    def test_rejects_a_nan_and_names_the_symbol_and_date(self, panels):
        prices, volumes = panels
        prices = prices.copy()
        prices.iloc[3, prices.columns.get_loc("ETH")] = np.nan

        with pytest.raises(ValueError, match="NaN values") as error:
            validate_panel(prices, volumes, START, END)

        assert "ETH" in str(error.value)
        assert "never interpolate" in str(error.value)

    def test_rejects_a_nan_in_the_volume_panel_too(self, panels):
        """Volume is an input of the eighth node feature, so a hole there is as
        disqualifying as a hole in the prices -- and the old signature, which
        reached for volume itself, checked only prices for NaN.
        """
        prices, volumes = panels
        volumes = volumes.copy()
        volumes.iloc[2, 0] = np.nan

        with pytest.raises(ValueError, match="Volume panel contains NaN"):
            validate_panel(prices, volumes, START, END)

    def test_rejects_a_non_positive_price(self, panels):
        prices, volumes = panels
        prices = prices.copy()
        prices.iloc[5, 0] = 0.0

        with pytest.raises(ValueError, match="non-positive price"):
            validate_panel(prices, volumes, START, END)

    def test_rejects_a_zero_volume_bar(self, panels):
        """A zero-volume candle is a quote, not a trade: the close it carries was
        never transacted, so the return computed across it is fictitious.
        """
        prices, volumes = panels
        volumes = volumes.copy()
        volumes.iloc[7, 1] = 0.0

        with pytest.raises(ValueError, match="Zero-volume bar"):
            validate_panel(prices, volumes, START, END)

    def test_rejects_the_wrong_period_even_when_the_panel_is_continuous(self, panels):
        """Independent of the continuity check: a perfectly gap-free panel can
        still cover a different period than the config asked for.
        """
        prices, volumes = panels

        with pytest.raises(ValueError, match="does not match"):
            validate_panel(prices, volumes, START, "2021-02-01")

    @pytest.mark.parametrize(
        ("mutate", "match"),
        [
            (lambda prices, volumes: (prices, volumes.iloc[:-1]), "differ in dates"),
            (lambda prices, volumes: (prices, volumes[list(reversed(SYMBOLS))]), "differ in symbols"),
        ],
    )
    def test_rejects_panels_that_were_not_built_together(self, panels, mutate, match):
        """The two panels are positional partners; a mismatch means one of them
        came from a different universe or a different cache.
        """
        with pytest.raises(ValueError, match=match):
            validate_panel(*mutate(*panels), START, END)

    def test_needs_no_filesystem(self, panels):
        """The signature takes the volume panel, not the directory it came from.

        The point of the change: a validation function performs no I/O, reads
        nothing its caller is about to read again, and is exercisable from
        in-memory frames -- which is what made these tests writable at all.
        """
        prices, volumes = panels

        validate_panel(prices.copy(), volumes.copy(), START, END)

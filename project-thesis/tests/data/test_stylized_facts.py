"""Tests for cryptognn.data.stylized_facts: the sanity check on the raw data.

`check_stylized_facts()` is the second gate of the pipeline, after
`validate_panel()`: it asserts that the returns behave like daily crypto returns
at all -- fat tails, no first-order predictability, clear volatility clustering.
If those fail, the data is wrong (bad download, misaligned dates), not the
market. Like every guard in this study it is only worth what its failures are
worth, so each of the three facts is tested against a series that violates it.

The per-asset exception mechanism is tested too. It currently carries exactly one
entry -- TRX, whose lag-1 autocorrelation was investigated and accepted as a
documented limitation -- and nothing verified that the escape hatch works, or
that it stays narrow enough to let a genuine violation through elsewhere.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from cryptognn.data.stylized_facts import (
    check_stylized_facts,
    compute_acf,
    compute_descriptive_stats,
    compute_ljung_box,
)

ASSETS = ["A", "B", "C"]
N_OBS = 800
TRADING_DAYS = 365


def panel(values: np.ndarray) -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=len(values), freq="D", tz="UTC")
    return pd.DataFrame(values, index=index, columns=ASSETS[: values.shape[1]])


@pytest.fixture
def crypto_like() -> pd.DataFrame:
    """Returns with the three stylized facts built in: a Student-t innovation for
    fat tails, scaled by a persistent volatility process for clustering.
    """
    rng = np.random.default_rng(4)
    shocks = rng.standard_t(df=4, size=(N_OBS, len(ASSETS)))

    volatility = np.empty_like(shocks)
    volatility[0] = 0.02
    for t in range(1, N_OBS):
        volatility[t] = 0.9 * volatility[t - 1] + 0.1 * 0.02 * (1 + np.abs(shocks[t - 1]))

    return panel(0.01 * volatility * shocks / volatility.mean())


@pytest.fixture
def gaussian_white_noise() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    return panel(rng.normal(0, 0.02, (N_OBS, len(ASSETS))))


class TestDescriptiveStats:
    def test_matches_the_formulas_it_documents(self, crypto_like):
        stats_frame = compute_descriptive_stats(crypto_like)

        assert list(stats_frame.index) == ASSETS
        assert list(stats_frame.columns) == ["mean", "volatility_annualized", "skewness", "excess_kurtosis"]

        column = crypto_like["A"]
        assert stats_frame.loc["A", "mean"] == pytest.approx(column.mean())
        # 365 days, not 252: crypto has no holidays, and the same convention is
        # used by the Sharpe ratio of Sprint 5.
        assert stats_frame.loc["A", "volatility_annualized"] == pytest.approx(column.std() * np.sqrt(TRADING_DAYS))
        assert stats_frame.loc["A", "excess_kurtosis"] == pytest.approx(stats.kurtosis(column))

    def test_excess_kurtosis_is_excess_not_raw(self):
        """Zero for a normal sample, not three -- the convention the check relies on."""
        rng = np.random.default_rng(2)
        normal = panel(rng.normal(0, 1, (20_000, 1)))

        assert compute_descriptive_stats(normal).loc["A", "excess_kurtosis"] == pytest.approx(0.0, abs=0.1)


class TestACF:
    def test_shape_and_lag_zero(self, crypto_like):
        acf_returns, acf_abs = compute_acf(crypto_like, n_lags=30)

        assert acf_returns.shape == acf_abs.shape == (31, len(ASSETS))
        assert acf_returns.index.name == "lag"
        np.testing.assert_allclose(acf_returns.loc[0], 1.0)
        np.testing.assert_allclose(acf_abs.loc[0], 1.0)

    def test_separates_return_from_volatility_autocorrelation(self, crypto_like):
        """The contrast the two series exist to show: r_t near unpredictable from
        its own past, |r_t| clearly persistent.
        """
        acf_returns, acf_abs = compute_acf(crypto_like)

        assert acf_returns.loc[1].abs().max() < 0.1
        assert (acf_abs.loc[1] > 0.05).all()


class TestCheckStylizedFacts:
    """One violated fact per test: a guard is worth what its failures are worth."""

    def test_accepts_data_that_behaves_like_crypto(self, crypto_like):
        descriptive = compute_descriptive_stats(crypto_like)
        acf_returns, acf_abs = compute_acf(crypto_like)

        check_stylized_facts(descriptive, acf_returns, acf_abs)  # must not raise

    def test_rejects_thin_tails(self, crypto_like):
        """Gaussian returns are the signature of simulated or smoothed data, not
        of a calm market: daily crypto has fat tails in every period.
        """
        descriptive = compute_descriptive_stats(crypto_like)
        acf_returns, acf_abs = compute_acf(crypto_like)
        descriptive.loc["B", "excess_kurtosis"] = -0.2

        with pytest.raises(ValueError, match="Excess kurtosis") as error:
            check_stylized_facts(descriptive, acf_returns, acf_abs)
        assert "'B'" in str(error.value) or "B" in str(error.value)

    def test_rejects_strongly_autocorrelated_returns(self, crypto_like):
        descriptive = compute_descriptive_stats(crypto_like)
        acf_returns, acf_abs = compute_acf(crypto_like)
        acf_returns.loc[1, "A"] = 0.45

        with pytest.raises(ValueError, match=r"ACF\(r_t\)"):
            check_stylized_facts(descriptive, acf_returns, acf_abs)

    def test_rejects_absent_volatility_clustering(self, gaussian_white_noise):
        """White noise has no clustering, so this is the fact that catches a
        return panel accidentally built from shuffled or synthetic data.
        """
        descriptive = compute_descriptive_stats(gaussian_white_noise)
        descriptive["excess_kurtosis"] = 1.0  # isolate the clustering check
        acf_returns, acf_abs = compute_acf(gaussian_white_noise)

        with pytest.raises(ValueError, match="volatility clustering"):
            check_stylized_facts(descriptive, acf_returns, acf_abs)

    def test_per_asset_exception_admits_only_the_asset_it_names(self, crypto_like):
        """The mechanism that currently carries TRX and nothing else.

        Raising the bar for one asset must not raise it for the others, or a
        documented exception would quietly become a blanket one.
        """
        descriptive = compute_descriptive_stats(crypto_like)
        acf_returns, acf_abs = compute_acf(crypto_like)
        acf_returns.loc[1, "A"] = 0.15

        with pytest.raises(ValueError, match=r"ACF\(r_t\)"):
            check_stylized_facts(descriptive, acf_returns, acf_abs)

        check_stylized_facts(descriptive, acf_returns, acf_abs, acf1_return_exceptions={"A": 0.2})

        acf_returns.loc[1, "C"] = 0.15
        with pytest.raises(ValueError, match="'C'"):
            check_stylized_facts(descriptive, acf_returns, acf_abs, acf1_return_exceptions={"A": 0.2})


class TestLjungBox:
    def test_reports_both_series_without_raising(self, crypto_like):
        """Deliberately non-raising, unlike check_stylized_facts(): at T ~ 2000
        even negligible autocorrelation is significant, so a low p-value is for a
        human to interpret, not for the pipeline to reject.
        """
        lb_returns, lb_abs = compute_ljung_box(crypto_like)

        for frame in (lb_returns, lb_abs):
            assert list(frame.index) == ASSETS
            assert list(frame.columns) == ["lb_stat", "lb_pvalue"]
            assert (frame["lb_stat"] >= 0).all()
            assert frame["lb_pvalue"].between(0, 1).all()

    def test_rejects_the_null_on_a_strongly_autocorrelated_series(self):
        """Sanity on the test itself: an AR(1) with phi = 0.8 must come out
        significant, or the statistic is not measuring what it claims.
        """
        rng = np.random.default_rng(6)
        values = np.zeros((N_OBS, 1))
        noise = rng.normal(0, 0.02, (N_OBS, 1))
        for t in range(1, N_OBS):
            values[t] = 0.8 * values[t - 1] + noise[t]

        lb_returns, _ = compute_ljung_box(panel(values))

        assert lb_returns.loc["A", "lb_pvalue"] < 1e-6

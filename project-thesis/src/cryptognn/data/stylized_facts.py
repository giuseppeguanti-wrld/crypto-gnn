"""Stylized facts (descriptive statistics) for the crypto-gnn study.

Computes univariate statistics for each asset's return series to detect data
quality issues (fat tails, skewness, volatility regime) and provide material for
thesis Section 6.1 (stylized facts of the cryptocurrency market).

Exports:
  - compute_descriptive_stats(): mean, annualized volatility, skewness, kurtosis per asset
  - compute_acf(): autocorrelation of r_t and |r_t| per asset, up to n_lags
  - check_stylized_facts(): asserts the expected stylized facts hold, raises otherwise
  - compute_ljung_box(): formal portmanteau test for autocorrelation in r_t and |r_t|

Integration: called by scripts/02_build_graphs.py (after returns are built) as a
  quality-assurance step; output saved to results/metrics/descriptive.parquet.
Why it exists: stylized facts are cheap to compute and serve two purposes: detect
  bad data early (zero variance, unexpected kurtosis < 3 for cryptos), and provide
  tables/plots for the thesis.
"""
from __future__ import annotations

import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf as _acf


def compute_descriptive_stats(returns: pd.DataFrame) -> pd.DataFrame:
    """Per-asset descriptive statistics: mean, annualized volatility, skewness, kurtosis.

    `returns` is expected to be the output of cryptognn.data.returns.log_returns():
    a (T, N) wide DataFrame of log-returns indexed by UTC date, one column per asset.
    Volatility is annualized as std(r_t) * sqrt(365): crypto trades every calendar
    day (no holidays), matching the 365-day convention used elsewhere in the study
    (e.g. evaluation.backtest.sharpe(periods=365)), not the 252 trading-day convention
    of traditional equity markets.
    Kurtosis is computed as the excess kurtosis (relative to normal, which has excess 0).

    Returns a wide DataFrame with one row per asset and columns:
      mean, volatility_annualized, skewness, excess_kurtosis
    """
    stats_dict = {}
    for asset in returns.columns:
        r = returns[asset]
        stats_dict[asset] = {
            "mean": r.mean(),
            "volatility_annualized": r.std() * (365 ** 0.5),
            "skewness": stats.skew(r),
            "excess_kurtosis": stats.kurtosis(r),
        }
    return pd.DataFrame(stats_dict).T


def compute_acf(returns: pd.DataFrame, n_lags: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Autocorrelation function of r_t and |r_t| for every asset, up to n_lags.

    Two ACF series are computed per asset because they probe different stylized
    facts: ACF(r_t) is expected near zero at all lags (returns are close to
    unpredictable from their own past), while ACF(|r_t|) is expected positive and
    slow-decaying (volatility clustering). Uses statsmodels' FFT-based estimator
    for efficiency and correct variance normalization.

    Returns (acf_returns, acf_abs_returns), two DataFrames shaped (n_lags+1, N)
    with lag 0..n_lags as the index (named "lag") and assets as columns. Lag 0
    is always 1.0 by construction and included for convenience.
    """
    lag_index = pd.RangeIndex(n_lags + 1, name="lag")
    acf_returns = {}
    acf_abs_returns = {}
    for asset in returns.columns:
        r = returns[asset]
        acf_returns[asset] = _acf(r, nlags=n_lags, fft=True)
        acf_abs_returns[asset] = _acf(r.abs(), nlags=n_lags, fft=True)

    return (
        pd.DataFrame(acf_returns, index=lag_index),
        pd.DataFrame(acf_abs_returns, index=lag_index),
    )


def check_stylized_facts(
    descriptive: pd.DataFrame,
    acf_returns: pd.DataFrame,
    acf_abs_returns: pd.DataFrame,
    *,
    acf1_return_threshold: float = 0.1,
    acf1_abs_return_threshold: float = 0.05,
    acf1_return_exceptions: dict[str, float] | None = None,
) -> None:
    """Assert the three stylized facts expected of daily crypto returns, raising
    a ValueError naming the offending asset(s) on the first violation -- if these
    fail, the data itself is wrong (bad download, misaligned dates, corrupted
    prices), not evidence that crypto stopped behaving like crypto. These are
    pragmatic sanity thresholds, not a formal hypothesis test (Ljung-Box, a
    separate step, provides that rigor).

      1. Excess kurtosis > 0 for every asset (i.e. raw kurtosis > 3): fat tails
         are a near-universal property of daily crypto returns.
      2. |ACF(r_t)| at lag 1 below `acf1_return_threshold` for every asset, unless
         overridden per-asset via `acf1_return_exceptions` (e.g. {"TRX": 0.2}):
         returns should not be strongly autocorrelated with their own previous day.
         `acf1_return_exceptions` exists for assets where a higher threshold was
         deliberately accepted after investigation (documented at the call site),
         not as a blanket escape hatch -- each entry should point to that rationale.
      3. ACF(|r_t|) at lag 1 above `acf1_abs_return_threshold` for every asset:
         volatility clustering should be clearly, not just marginally, positive.
    """
    flat_kurtosis = descriptive.loc[descriptive["excess_kurtosis"] <= 0]
    if not flat_kurtosis.empty:
        raise ValueError(
            f"Excess kurtosis <= 0 (no fat tails) for: {list(flat_kurtosis.index)} -- data is suspect"
        )

    exceptions = acf1_return_exceptions or {}
    acf1_return_limits = pd.Series(acf1_return_threshold, index=acf_returns.columns)
    acf1_return_limits.update(pd.Series(exceptions))
    strong_acf1 = acf_returns.columns[acf_returns.loc[1].abs() >= acf1_return_limits]
    if len(strong_acf1) > 0:
        raise ValueError(
            f"|ACF(r_t)| at lag 1 >= threshold (default {acf1_return_threshold}, "
            f"exceptions {exceptions}) for: {list(strong_acf1)} -- "
            "returns should be close to unpredictable from their own past"
        )

    weak_clustering = acf_abs_returns.columns[acf_abs_returns.loc[1] <= acf1_abs_return_threshold]
    if len(weak_clustering) > 0:
        raise ValueError(
            f"ACF(|r_t|) at lag 1 <= {acf1_abs_return_threshold} for: {list(weak_clustering)} -- "
            "expected clear volatility clustering"
        )


def compute_ljung_box(returns: pd.DataFrame, lags: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ljung-Box portmanteau test for autocorrelation in r_t and |r_t|, for every asset.

    Formalizes, with a citable p-value, what check_stylized_facts() only checks
    heuristically via a single ACF(1) threshold: H0 is "no autocorrelation up to
    `lags` lags" (statistic ~ chi-squared(lags) under H0). Tested at the same
    horizon as compute_acf()'s default so the joint test and the ACF plot agree
    on what "up to lag 30" means.

    Expected outcome on well-behaved daily crypto data: p-value on r_t not
    overwhelmingly significant (returns close to unpredictable); p-value on
    |r_t| strongly significant (H0 rejected), confirming volatility clustering.

    Unlike check_stylized_facts(), this does NOT raise: with T ~ 2000, even
    economically negligible autocorrelation is often statistically significant,
    so a low p-value alone is not proof of a data error. Statistic and p-value
    are returned for reporting (results/metrics, thesis Section 6.1) and for a
    human to interpret in context.

    Returns (ljung_box_returns, ljung_box_abs_returns), two DataFrames with one
    row per asset and columns lb_stat, lb_pvalue, evaluated at `lags`.
    """
    lb_returns = {}
    lb_abs_returns = {}
    for asset in returns.columns:
        r = returns[asset]
        lb_r = acorr_ljungbox(r, lags=[lags], return_df=True).iloc[0]
        lb_abs = acorr_ljungbox(r.abs(), lags=[lags], return_df=True).iloc[0]
        lb_returns[asset] = {"lb_stat": lb_r["lb_stat"], "lb_pvalue": lb_r["lb_pvalue"]}
        lb_abs_returns[asset] = {"lb_stat": lb_abs["lb_stat"], "lb_pvalue": lb_abs["lb_pvalue"]}

    return pd.DataFrame(lb_returns).T, pd.DataFrame(lb_abs_returns).T

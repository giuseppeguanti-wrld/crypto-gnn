"""Fixtures shared across the test packages.

The tree under tests/ mirrors src/cryptognn/, but these three fixtures do not
belong to any one package: graph/test_correlation, graph/test_threshold,
graph/test_build, graph/test_metrics and data/test_stylized_facts all exercise
different stages of the same pipeline and work from the same synthetic inputs --
a return panel with a known factor structure, and a small correlation matrix
chosen to span the cases thresholding has to separate. evaluation/test_walkforward
and test_features work from a third: a small study container whose values are
readable by eye. A fixture used from four packages lives at their common root.

The constants describing their shapes are in tests/synthetic.py, not here: see
that module for why a file other modules import from should not be a conftest.

Nothing here reads data/ or results/: the suite must run on a fresh clone,
before any script has been executed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from synthetic import (
    N_ASSETS,
    WF_GRAPH_OFFSET,
    WF_LOOKBACK,
    WF_N_ASSETS,
    WF_N_FEATURES,
    WF_N_OBS,
)

from cryptognn.evaluation.walkforward import WalkforwardData


@pytest.fixture
def correlated_returns() -> pd.DataFrame:
    """A (400, 6) return panel with a strong common factor, so every pair is
    correlated at roughly 0.8 -- far enough from zero that a null which failed
    to break the dependence is unmistakable.
    """
    rng = np.random.default_rng(123)
    n_obs = 400
    market = rng.standard_normal((n_obs, 1))
    idiosyncratic = rng.standard_normal((n_obs, N_ASSETS))
    values = 0.02 * (2.0 * market + idiosyncratic)
    index = pd.date_range("2021-01-01", periods=n_obs, freq="D", tz="UTC")
    return pd.DataFrame(values, index=index, columns=[f"A{i}" for i in range(N_ASSETS)])


@pytest.fixture
def synthetic_volumes(correlated_returns: pd.DataFrame) -> pd.DataFrame:
    """A positive volume panel on the return panel's index, with assets whose
    units differ by orders of magnitude (column i scaled by 10^i).

    The spread is deliberate: the eighth node feature is the z-score of *log*
    volume, which should be blind to an asset's unit of account. A panel where
    every column had the same magnitude could not tell a correct implementation
    from one that happens to work only on comparable scales.
    """
    rng = np.random.default_rng(321)
    values = np.exp(rng.normal(loc=10.0, scale=1.0, size=correlated_returns.shape))
    scales = 10.0 ** np.arange(correlated_returns.shape[1])
    return pd.DataFrame(
        values * scales, index=correlated_returns.index, columns=correlated_returns.columns
    )


@pytest.fixture
def sample_corr() -> np.ndarray:
    """A 4x4 correlation matrix spanning the cases the threshold must separate:
    one strongly anticorrelated pair (-0.5), one weakly positive pair (0.1) that
    still falls below tau, and three pairs comfortably above it.
    """
    return np.array(
        [
            [1.0, 0.8, 0.3, -0.5],
            [0.8, 1.0, 0.6, 0.1],
            [0.3, 0.6, 1.0, 0.9],
            [-0.5, 0.1, 0.9, 1.0],
        ]
    )


@pytest.fixture
def synthetic_walkforward_data() -> WalkforwardData:
    """A (120, 4) study container whose every value announces where it came from.

    returns[t, j] == t + j/10, so a misalignment by one row or one asset is
    visible in the number itself rather than hidden in a plausible float: the
    target of position t must read t+1, and the most recent lag must read t.
    features[t, j, f] == returns[t, j] + 100*f keeps the feature axis equally
    identifiable, and the graph is NaN before WF_GRAPH_OFFSET exactly as
    align_graph() leaves it on the study data.
    """
    dates = pd.date_range("2021-01-01", periods=WF_N_OBS, freq="D", tz="UTC")
    returns = np.arange(WF_N_OBS)[:, None] + np.arange(WF_N_ASSETS)[None, :] / 10.0
    features = returns[:, :, None] + 100.0 * np.arange(WF_N_FEATURES)[None, None, :]

    a_hat = np.full((WF_N_OBS, WF_N_ASSETS, WF_N_ASSETS), np.nan)
    a_hat[WF_GRAPH_OFFSET:] = np.eye(WF_N_ASSETS)

    return WalkforwardData(
        dates=dates,
        assets=tuple(f"A{i}" for i in range(WF_N_ASSETS)),
        returns=returns,
        graph_offset=WF_GRAPH_OFFSET,
        lookback=WF_LOOKBACK,
        features=features,
        a_hat=a_hat,
    )

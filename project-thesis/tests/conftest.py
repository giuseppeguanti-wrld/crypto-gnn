"""Fixtures shared by the graph test modules.

test_correlation, test_threshold, test_build and test_metrics all exercise
different stages of the same pipeline, so they work from the same two synthetic
inputs: a return panel with a known factor structure, and a small correlation
matrix chosen to span the cases thresholding has to separate.

Nothing here reads data/ or results/: the suite must run on a fresh clone,
before any script has been executed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

N_ASSETS = 6
N_PAIRS = N_ASSETS * (N_ASSETS - 1) // 2

# The threshold calibrated on the study data, used across the construction and
# metric tests as a realistic value rather than a round invented one.
TAU = 0.2145


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

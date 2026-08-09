"""Rolling correlation matrices for the crypto-gnn study's dynamic graph.

Turns the (T, N) log-return panel into a (T-window+1, N, N) sequence of rolling
Pearson correlation matrices -- the raw material every downstream graph step
(Mantegna weighting, thresholding, topological metrics) is built from.

Exports (built incrementally):
  - correlation_from_windows(): batched correlation kernel over stacked windows
  - rolling_correlation(): vectorized rolling correlation via sliding_window_view
  - validate_correlation(): asserts diagonal/symmetry/range invariants, raises otherwise
  - (more to be added: on-disk save)

Integration: called by scripts/02_build_graphs.py; output (corr_60.npy, corr_index.npy)
  feeds cryptognn.graph.threshold (tau calibration) and cryptognn.graph.build.
Why it exists: pandas' df.rolling().corr() re-fits from scratch per window per pair
  and is roughly 100x slower at this scale (~2000 windows x 15 assets) than a batched
  numpy computation over sliding_window_view's zero-copy windows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def correlation_from_windows(windows: np.ndarray) -> np.ndarray:
    """Pearson correlation matrix of every window in a stack, in one batched pass.

    `windows` is (K, N, W): K windows of W observations for each of N assets.
    Returns (K, N, N) in float64, where result[k] is the correlation matrix of
    windows[k].

    Factored out of rolling_correlation() because the permutation null of
    cryptognn.graph.threshold needs exactly the same kernel over a different
    stack of windows (B shuffled replicas of one window, rather than T-window+1
    consecutive windows of the panel).
    """
    centered = windows - windows.mean(axis=2, keepdims=True)
    cov = np.einsum("knw,kmw->knm", centered, centered)
    variance = np.einsum("knw,knw->kn", centered, centered)
    std = np.sqrt(variance)

    return cov / (std[:, :, None] * std[:, None, :])


def rolling_correlation(returns: pd.DataFrame, window: int) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Rolling Pearson correlation matrix of every asset pair, for every window.

    Vectorized: sliding_window_view() produces all (T-window+1) windows as
    zero-copy views of the underlying array, then a single batched computation
    (via einsum) derives the covariance and correlation of every window at once
    -- no Python-level loop over windows, no pandas per-pair rolling machinery.

    Returns:
      corr: np.ndarray (T-window+1, N, N), float32. corr[k] is the correlation
        matrix of the window ending at corr_index[k] (inclusive).
      corr_index: the date of the last observation in each window -- the pivot
        of the anti-look-ahead discipline: a model predicting r_{t+1} may only
        use corr[k] where corr_index[k] <= t.
    """
    values = returns.to_numpy(dtype=np.float64)  # (T, N)
    windows = sliding_window_view(values, window, axis=0)  # (T-window+1, N, window)

    corr = correlation_from_windows(windows)
    corr_index = returns.index[window - 1 :]

    return corr.astype(np.float32), corr_index


def validate_correlation(corr: np.ndarray, atol: float = 1e-6) -> None:
    """Assert the mathematical invariants of a rolling correlation tensor,
    raising a ValueError on the first violation rather than merely reporting it.
    These are properties any valid correlation matrix must have, so a failure
    here means a bug in rolling_correlation(), never a property of the data.

      1. Diagonal == 1 within `atol` for every window (each asset is perfectly
         correlated with itself).
      2. Symmetry: corr[k] == corr[k].T within `atol`, for every window.
      3. Every value in [-1, 1] (within `atol`, to tolerate float rounding).
    """
    diagonal = np.diagonal(corr, axis1=1, axis2=2)
    if not np.allclose(diagonal, 1.0, atol=atol):
        bad = np.argwhere(~np.isclose(diagonal, 1.0, atol=atol))
        raise ValueError(f"Diagonal not 1 within {atol} at (window, asset) indices: {bad[:10].tolist()}")

    if not np.allclose(corr, corr.transpose(0, 2, 1), atol=atol):
        bad = np.argwhere(~np.isclose(corr, corr.transpose(0, 2, 1), atol=atol))
        raise ValueError(f"Correlation matrices not symmetric within {atol} at indices: {bad[:10].tolist()}")

    out_of_range = (corr < -1.0 - atol) | (corr > 1.0 + atol)
    if out_of_range.any():
        bad = np.argwhere(out_of_range)
        raise ValueError(f"Correlation values outside [-1, 1] at indices: {bad[:10].tolist()}")

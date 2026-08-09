"""Tests for cryptognn.graph.correlation.

Both the batched kernel and the rolling wrapper are checked against numpy's
reference implementation window by window: the vectorized version exists purely
for speed, so any disagreement with the obvious implementation is a bug in the
optimization, never a different definition of correlation.
"""
from __future__ import annotations

import numpy as np
from conftest import N_ASSETS

from cryptognn.graph.correlation import correlation_from_windows, rolling_correlation


class TestCorrelationFromWindows:
    def test_matches_numpy(self, correlated_returns):
        """The shared batched kernel must agree with numpy's reference implementation."""
        values = correlated_returns.to_numpy()
        windows = np.stack([values[0:60].T, values[100:160].T])  # (2, N, W)

        result = correlation_from_windows(windows)

        assert result.shape == (2, N_ASSETS, N_ASSETS)
        np.testing.assert_allclose(result[0], np.corrcoef(values[0:60].T), atol=1e-12)
        np.testing.assert_allclose(result[1], np.corrcoef(values[100:160].T), atol=1e-12)


class TestRollingCorrelation:
    def test_matches_numpy_window_by_window(self, correlated_returns):
        """Regression guard on the extraction of correlation_from_windows():
        rolling_correlation() must still match numpy window by window.
        """
        corr, corr_index = rolling_correlation(correlated_returns, window=60)
        values = correlated_returns.to_numpy()

        assert corr.shape == (len(correlated_returns) - 59, N_ASSETS, N_ASSETS)
        assert corr_index[0] == correlated_returns.index[59]
        np.testing.assert_allclose(corr[0], np.corrcoef(values[0:60].T), atol=1e-6)
        np.testing.assert_allclose(corr[-1], np.corrcoef(values[-60:].T), atol=1e-6)

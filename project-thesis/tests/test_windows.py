"""Tests for cryptognn.windows: the causal rolling window.

Small module, disproportionate importance. Every model input of the study passes
through this function -- the lagged returns and volatilities of features.py, the
lag history of every Segment -- so its contract is the one the anti-look-ahead
argument ultimately rests on: **row t contains rows [t - window + 1, t] and
nothing later**.

It is therefore checked against the obvious implementation, a Python loop that
slices the panel one row at a time. The vectorized version exists to avoid
copying the panel once per window; any disagreement with the loop is a bug in
the optimization, never a different definition.
"""
from __future__ import annotations

import numpy as np
import pytest

from cryptognn.windows import causal_windows

N_ROWS, N_COLUMNS = 40, 3


@pytest.fixture
def panel() -> np.ndarray:
    """values[t, j] == t + j/10, so a row is identifiable at a glance."""
    return np.arange(N_ROWS)[:, None] + np.arange(N_COLUMNS)[None, :] / 10.0


class TestCausalWindows:
    def test_shape_and_orientation(self, panel):
        windows = causal_windows(panel, 5)

        assert windows.shape == (N_ROWS, 5, N_COLUMNS)
        # Chronological along the middle axis: the last entry is the row itself.
        np.testing.assert_array_equal(windows[10, -1], panel[10])
        np.testing.assert_array_equal(windows[10, -2], panel[9])
        np.testing.assert_array_equal(windows[10, 0], panel[6])

    def test_matches_the_naive_loop(self, panel):
        window = 7
        result = causal_windows(panel, window)

        for position in range(window - 1, N_ROWS):
            np.testing.assert_array_equal(result[position], panel[position - window + 1 : position + 1])

    def test_warm_up_rows_are_nan_and_nothing_else_is(self, panel):
        window = 6
        result = causal_windows(panel, window)

        assert np.isnan(result[: window - 1]).any(axis=(1, 2)).all()
        assert np.isfinite(result[window - 1 :]).all()
        # The NaN sits at the *old* end of an incomplete window, never at the
        # recent end: row 0 knows itself and nothing before it.
        np.testing.assert_array_equal(result[0, -1], panel[0])
        assert np.isnan(result[0, :-1]).all()

    def test_a_window_of_one_is_the_panel_itself(self, panel):
        np.testing.assert_array_equal(causal_windows(panel, 1)[:, 0, :], panel)

    def test_ignores_the_future(self, panel):
        """The property the whole module exists for, checked by corruption."""
        cutoff = 25
        baseline = causal_windows(panel, 5)
        corrupted = panel.copy()
        corrupted[cutoff + 1 :] = -999.0

        rebuilt = causal_windows(corrupted, 5)

        np.testing.assert_array_equal(rebuilt[: cutoff + 1], baseline[: cutoff + 1])
        assert not np.array_equal(rebuilt[cutoff + 1], baseline[cutoff + 1])

    def test_does_not_mutate_the_input(self, panel):
        original = panel.copy()
        causal_windows(panel, 4)
        np.testing.assert_array_equal(panel, original)

    @pytest.mark.parametrize(
        ("values", "window", "match"),
        [
            (np.zeros((10,)), 3, "Expected a"),
            (np.zeros((10, 2)), 0, "at least 1"),
            (np.zeros((10, 2)), 11, "exceeds the 10 rows"),
        ],
    )
    def test_rejects_impossible_input(self, values, window, match):
        with pytest.raises(ValueError, match=match):
            causal_windows(values, window)

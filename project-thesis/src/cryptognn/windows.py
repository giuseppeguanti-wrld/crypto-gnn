"""The causal rolling window, in one place.

A single primitive: for every row t of a panel, the `window` rows ending at t
**inclusive**. It is the operation that makes "row t knows nothing after t" true
rather than intended, and every input a model receives passes through it -- the
lagged returns and realized volatilities of cryptognn.features, and the lag
history the walk-forward harness attaches to each Segment.

It lives in its own module for exactly that reason. The implementation was
written twice, once in each of those callers, and a duplicated primitive is a
correction waiting to reach only half the code that needs it -- here, the half
that the anti-look-ahead tests of Sprint 3 exist to protect.

Exports:
  - causal_windows(): (T, window, N), chronological, NaN-padded at the start

Integration: imported by cryptognn.features and
  cryptognn.evaluation.walkforward. Depends on numpy alone, so it can be used
  from anywhere without creating an import cycle.
Why cryptognn.graph.correlation does not use it: rolling_correlation() has the
  opposite contract for incomplete windows. It *drops* the first window-1 rows,
  returning a shorter array indexed by its own dates, because a correlation
  matrix of a partial window is not a value anyone should carry around. Here
  the array keeps the panel's length -- callers index it by position, and the
  positions that have no history must stay aligned and be NaN.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def causal_windows(values: np.ndarray, window: int) -> np.ndarray:
    """For each row t of `values` (T, N), the rows [t - window + 1, t] as (T, window, N).

    The middle axis runs chronologically: `result[t, -1]` is row t itself and
    `result[t, -1 - k]` is row t - k. Callers therefore index backwards from the
    end, which reads the same way the lag notation does.

    The first `window - 1` rows have no complete history and come back NaN. They
    are not filled: any value invented there could only be derived from the rows
    that follow, which is the look-ahead this primitive exists to prevent. The
    padding also keeps the result aligned with the input, so position t means
    the same thing in this array as in every other array of the study.

    The windows are views into one padded copy, so the cost is a single copy of
    the panel regardless of `window`.
    """
    if values.ndim != 2:
        raise ValueError(f"Expected a (T, N) panel, got {values.ndim} dimensions")
    if window < 1:
        raise ValueError(f"Window must be at least 1, got {window}")
    if window > len(values):
        raise ValueError(f"Window of {window} exceeds the {len(values)} rows available")

    pad = np.full((window - 1, values.shape[1]), np.nan)
    padded = np.vstack([pad, values])
    return sliding_window_view(padded, window, axis=0).transpose(0, 2, 1)

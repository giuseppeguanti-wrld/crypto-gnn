"""The two baselines that use no dynamics at all.

Both exist to make the question "does the model beat doing nothing?" answerable
with a number rather than a shrug, and the first of them is the harder opponent
of the two: daily crypto returns have a mean statistically indistinguishable
from zero, so a forecast of zero already sits close to the minimum achievable
squared error. Anything that fails to beat it has learned nothing about the
conditional mean, whatever its R-squared on the training block says.

Exports:
  - ZeroForecaster: r_hat = 0, the reference of the skill score
  - HistoricalMeanForecaster: the train block's mean, per asset

Integration: implement cryptognn.models.base.Forecaster, run through
  cryptognn.evaluation.walkforward.run_walkforward like every other model.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    # Annotation-only, and deliberately so: the harness imports models.base to
    # detect the diagnostics hook, so a model importing the harness back at
    # runtime would close a cycle through cryptognn.models.__init__.
    from cryptognn.evaluation.walkforward import Segment


class ZeroForecaster:
    """Forecasts zero, always.

    The skill score of Section 6.4 is defined against this model, which makes
    the headline question direct: a positive skill means the forecast carries
    information beyond "returns are unpredictable and centred on nothing".
    """

    name = "zero"

    def fit(self, train: Segment, val: Segment) -> None:
        """Nothing to estimate. The signature is the harness's, not this model's."""

    def predict(self, segment: Segment) -> np.ndarray:
        return np.zeros((len(segment), segment.n_assets))


class HistoricalMeanForecaster:
    """Forecasts each asset's mean return over the fold's training block.

    The mean is computed from `train.returns` and from nothing else, so its
    freedom from look-ahead is structural: the segment simply does not contain a
    row the model should not see. On daily returns the estimate is tiny and
    noisy -- roughly 1e-4 against a standard deviation of 4e-2 -- which is why
    this baseline is expected to land marginally *worse* than forecasting zero,
    and why that outcome is informative rather than a bug.
    """

    name = "mean"

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None

    def fit(self, train: Segment, val: Segment) -> None:
        self.mean_ = train.returns.mean(axis=0)

    def predict(self, segment: Segment) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("HistoricalMeanForecaster.predict() called before fit()")
        return np.tile(self.mean_, (len(segment), 1))

    def diagnostics(self) -> dict[str, float | int | str]:
        return {"mean_abs_level": float(np.abs(self.mean_).mean())}

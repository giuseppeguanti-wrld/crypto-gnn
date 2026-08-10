"""The multivariate baseline: a vector autoregression on the 15 return series.

The linear answer to the study's question. A VAR lets every asset's past enter
every other asset's forecast, which is exactly the cross-asset dependence a
graph model claims to exploit -- without the graph, and with a coefficient for
each of the N^2 p possible channels.

Two configurations, both run:

  - **var-bic** -- order chosen by BIC, as `config/default.yaml` specifies. On
    this data it selects p = 0 on all 24 folds: the criterion declines to
    estimate a single cross-asset coefficient. That is the strongest form of the
    over-parametrization argument the thesis makes, since it is not an opinion
    about 1140 coefficients being too many but a selection criterion refusing
    them.
  - **var-p5** -- order fixed at `config.model.var.fixed_lag`, a robustness
    variant registered before any test result was seen (the same footing as
    A_hat_fwer in Sprint 2). Without it the study would have no multivariate
    comparator at all, and no empirical picture of what estimating
    N(Np + 1) = 1140 coefficients from 365 observations does out of sample.

Exports:
  - VARForecaster: OLS VAR with either BIC-selected or fixed lag order

Integration: implements cryptognn.evaluation.protocols.Forecaster and SupportsDiagnostics.
Why the forecast is computed here rather than by results.forecast(): the same
  reason as in models/ar.py -- the input stays what the harness certifies as
  past -- plus one this model makes concrete, in that the whole test block
  becomes a single contraction instead of 63 library calls. The two agree by
  construction, and a test holds them to it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from statsmodels.tsa.api import VAR

from cryptognn.config import Config

if TYPE_CHECKING:
    from cryptognn.evaluation.walkforward import Segment


class VARForecaster:
    """VAR(p) on the return panel, p by information criterion or fixed.

    `lags=None` selects the order by `config.model.var.ic` up to `max_lag`;
    an integer fixes it. The fitted coefficients are reshaped once into
    (p, N_source, N_target) so a forecast is a contraction over the lag and
    source axes, matching the algebra of y_hat = c + sum_i A_i y_{t+1-i}.
    """

    def __init__(self, config: Config, lags: int | None = None) -> None:
        self.max_lag = config.model.var.max_lag
        self.ic = config.model.var.ic
        self.lags = lags
        self.name = "var-bic" if lags is None else f"var-p{lags}"
        self.const_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.k_ar_: int | None = None
        self.n_train_: int | None = None

    def fit(self, train: Segment, val: Segment) -> None:
        model = VAR(train.returns)
        results = model.fit(maxlags=self.max_lag, ic=self.ic) if self.lags is None else model.fit(self.lags)

        params = np.asarray(results.params, dtype=float)  # (1 + N*k, N), trend 'c'
        self.k_ar_ = int(results.k_ar)
        self.const_ = params[0]
        # Rows of params run lag-major: [const, L1.y1..L1.yN, L2.y1..L2.yN, ...],
        # columns are the equations. The reshape therefore lands on
        # (lag, source, target), which is the order the einsum below expects.
        self.coef_ = params[1:].reshape(self.k_ar_, len(self.const_), len(self.const_))
        self.n_train_ = len(train)

    def predict(self, segment: Segment) -> np.ndarray:
        if self.const_ is None or self.k_ar_ is None:
            raise RuntimeError("VARForecaster.predict() called before fit()")

        if self.k_ar_ == 0:
            # Not a defensive branch: BIC reaches it on every fold of this study.
            return np.tile(self.const_, (len(segment), 1))
        if segment.lags.shape[1] < self.k_ar_:
            raise ValueError(
                f"VAR order {self.k_ar_} exceeds the {segment.lags.shape[1]} lags the segment carries; "
                "raise WalkforwardData.lookback"
            )

        recent = segment.lags[:, -self.k_ar_ :, :][:, ::-1, :]  # (n, k, N), index 0 = lag 1
        return self.const_ + np.einsum("nis,ist->nt", recent, self.coef_)

    def diagnostics(self) -> dict[str, float | int | str]:
        """The parameter count against the sample size -- the table of Section 6.4."""
        n_assets = len(self.const_)
        n_params = n_assets * (n_assets * self.k_ar_ + 1)
        return {
            "var_lag_order": self.k_ar_,
            "n_params": n_params,
            "obs_per_param": float(self.n_train_ * n_assets / n_params),
        }

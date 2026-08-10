"""The univariate baseline: one autoregression per asset, order chosen by BIC.

This is the literal *modello univariato* of the study's first research question --
each asset predicted from its own past alone, with no channel through which one
asset's history can inform another's forecast. Whatever a graph model gains, it
has to gain against this.

On the study's data the answer BIC gives is stark: order 0 in 85% of the
per-asset fits, which is to say "the past does not help, forecast the mean".
That is a result to report, not a defect to work around, and the diagnostics
below exist so it can be reported with numbers.

Exports:
  - PerAssetARForecaster: N independent AutoReg fits, BIC order selection

Integration: implements cryptognn.models.base.Forecaster and
  SupportsDiagnostics; run by cryptognn.evaluation.walkforward.run_walkforward.
Why the forecast does not call statsmodels: predicting through the library would
  mean handing it a series and trusting where it reads from. Applying the fitted
  coefficients to Segment.lags keeps the input to what the harness guarantees is
  dated at or before the prediction origin -- and makes the whole test block one
  matrix multiplication instead of 63 calls.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from statsmodels.tsa.ar_model import ar_select_order

from cryptognn.config import Config

if TYPE_CHECKING:
    from cryptognn.evaluation.walkforward import Segment


class PerAssetARForecaster:
    """AR(p) per asset, p selected by information criterion on the train block.

    Each asset gets its own order: the selection is run N times on N series, not
    once on a pooled one, because an order that suits BTC has no reason to suit
    XLM. Orders and coefficients are stored as dense (N,) and (N, p_max) arrays
    with zeros where an asset uses fewer lags, so the whole panel is forecast in
    one einsum rather than in a loop over assets.
    """

    name = "ar"

    def __init__(self, config: Config) -> None:
        self.max_lag = config.model.ar.max_lag
        self.ic = config.model.ar.ic
        self.const_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.orders_: np.ndarray | None = None

    def fit(self, train: Segment, val: Segment) -> None:
        values = train.returns
        n_assets = values.shape[1]

        const = np.zeros(n_assets)
        coef = np.zeros((n_assets, self.max_lag))
        orders = np.zeros(n_assets, dtype=int)

        for asset in range(n_assets):
            selection = ar_select_order(values[:, asset], maxlag=self.max_lag, ic=self.ic, trend="c")
            # ar_lags is None when the criterion picks no lag at all: the fitted
            # model is then an intercept, and the forecast its constant.
            lags = list(selection.ar_lags or [])
            params = np.asarray(selection.model.fit().params, dtype=float)

            const[asset] = params[0]
            orders[asset] = len(lags)
            for position, lag in enumerate(lags):
                coef[asset, lag - 1] = params[1 + position]

        self.const_, self.coef_, self.orders_ = const, coef, orders

    def predict(self, segment: Segment) -> np.ndarray:
        if self.const_ is None or self.coef_ is None or self.orders_ is None:
            raise RuntimeError("PerAssetARForecaster.predict() called before fit()")

        order = int(self.orders_.max())
        if order == 0:
            return np.tile(self.const_, (len(segment), 1))
        if segment.lags.shape[1] < order:
            raise ValueError(
                f"Selected AR order {order} exceeds the {segment.lags.shape[1]} lags the segment carries; "
                "raise WalkforwardData.lookback"
            )

        # lags run oldest-first, so reversing makes index k-1 the k-th lag and
        # lines the window up with the coefficient columns.
        recent = segment.lags[:, -order:, :][:, ::-1, :]
        return self.const_ + np.einsum("nka,ak->na", recent, self.coef_[:, :order])

    def diagnostics(self) -> dict[str, float | int | str]:
        """What the criterion decided, which on this data is the finding itself."""
        return {
            "ar_lag_mean": float(self.orders_.mean()),
            "ar_lag_max": int(self.orders_.max()),
            "ar_zero_order_share": float((self.orders_ == 0).mean()),
            "n_params": int((self.orders_ + 1).sum()),
        }

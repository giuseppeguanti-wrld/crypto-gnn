"""The study's forecasters, and the roster of baselines the comparison runs.

Exports:
  - baseline_factories(): name -> zero-argument factory, one entry per baseline
  - Forecaster, SupportsDiagnostics: re-exported from cryptognn.evaluation.protocols,
    where the contract lives, so `from cryptognn.models import Forecaster` keeps
    working for callers that think of it as part of the model layer

Integration: scripts/04_run_baselines.py iterates this mapping and hands each
  factory to run_walkforward, so adding or removing a baseline is a change here
  and nowhere else. The factories take no arguments because run_walkforward
  builds a fresh model per fold and must not need to know what any of them
  requires to be constructed.
"""
from __future__ import annotations

from collections.abc import Callable

from cryptognn.config import Config
from cryptognn.evaluation.protocols import Forecaster, SupportsDiagnostics
from cryptognn.models.ar import PerAssetARForecaster
from cryptognn.models.naive import HistoricalMeanForecaster, ZeroForecaster
from cryptognn.models.var import VARForecaster

__all__ = [
    "Forecaster",
    "HistoricalMeanForecaster",
    "PerAssetARForecaster",
    "SupportsDiagnostics",
    "VARForecaster",
    "ZeroForecaster",
    "baseline_factories",
]


def baseline_factories(config: Config) -> dict[str, Callable[[], Forecaster]]:
    """The five baseline runs of Section 6.4, in the order they are reported.

    Two VAR entries rather than one: `var-bic` is the model the frozen config
    specifies, `var-p5` the pre-registered fixed-order variant that exists
    because BIC selects p = 0 on every fold and would otherwise leave the study
    without a multivariate comparator.
    """
    return {
        "zero": ZeroForecaster,
        "mean": HistoricalMeanForecaster,
        "ar": lambda: PerAssetARForecaster(config),
        "var-bic": lambda: VARForecaster(config),
        "var-p5": lambda: VARForecaster(config, lags=config.model.var.fixed_lag),
    }

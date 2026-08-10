"""The contract every forecaster of the study satisfies.

One interface for four baselines and a GCN, so the walk-forward harness runs
them through the identical loop and any difference in their scores comes from
the models rather than from how each was called.

The interface is a Protocol, not a base class: structural typing keeps the
models free of an inheritance requirement -- a wrapper around statsmodels and an
nn.Module subclass can both satisfy it without sharing an ancestor -- while a
type checker still verifies the signatures at every call site.

**It lives beside the harness, not beside the models, and that is the point.**
The requirement originates here: the loop in walkforward.py is what needs to fit
and predict, and the implementations conform to it. Housing the protocol in
cryptognn.models instead made the harness import the model package, which meant
importing the evaluation protocol pulled in statsmodels -- and would have pulled
in torch as soon as Sprint 4 added the GCN, including for the Streamlit app,
which needs neither. Dependencies now run one way: models -> evaluation.

Exports:
  - Forecaster: fit(train, val) / predict(segment), plus a `name` used in results
  - SupportsDiagnostics: the optional per-fold reporting hook

Integration: implemented by cryptognn.models.{naive,ar,var,gcn}, consumed by
  cryptognn.evaluation.walkforward.run_walkforward().
Why fit() always takes a validation split: only the GCN early-stops on it, but a
  signature that varies per model forces the harness to branch on model type,
  and a harness that knows which model it is running is one edit away from
  treating them unequally. The baselines accept the argument and ignore it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from cryptognn.evaluation.walkforward import Segment


@runtime_checkable
class Forecaster(Protocol):
    """One-step-ahead multi-asset forecaster over a walk-forward fold."""

    name: str

    def fit(self, train: Segment, val: Segment) -> None:
        """Estimate the model on the train split; `val` is available for models
        that select or early-stop on it, and must be ignored by the others.
        Neither split may be used to alter anything after test predictions are
        seen -- that is a rule of the study, not of the type system.
        """
        ...

    def predict(self, segment: Segment) -> np.ndarray:
        """Forecast the target of every position in `segment`, shape (len(segment), n_assets).

        Row i must depend only on information dated at or before `segment.positions[i]`:
        `segment.lags[i]`, `segment.features[i]` and `segment.a_hat[i]` all satisfy
        that by construction, which is why they exist.

        `segment.y` is NaN here: the harness withholds the targets it is about to
        score against, so reading them is not an option even by accident.
        """
        ...


@runtime_checkable
class SupportsDiagnostics(Protocol):
    """Optional hook for what a fitted model has to say about itself.

    Kept separate from Forecaster so implementing it stays voluntary: the harness
    checks with isinstance() and merges the returned mapping into the fold's
    diagnostic row. This is how the VAR reports the lag order BIC selected and
    the resulting parameter count -- the empirical form of the over-parametrization
    argument the thesis makes about VAR baselines -- and how the GCN will report
    the epochs early stopping actually ran.
    """

    def diagnostics(self) -> dict[str, float | int | str]:
        """Scalars describing the fit just performed, one value per column."""
        ...

"""Walk-forward evaluation harness: fold geometry, data views, and the run loop.

The protocol by which every model of the study is judged, written before any
model exists so that the yardstick cannot be shaped around a result. Three
pieces, in increasing order of responsibility:

  - **Fold / make_folds()** -- the geometry. Which positions of the return panel
    belong to train, to validation and to test, for each of the 24 folds.
  - **WalkforwardData / Segment** -- the views. A model never receives the panel;
    it receives a Segment holding the rows of one split only, so a whole class of
    look-ahead mistakes is unavailable rather than merely discouraged.
  - **run_walkforward()** -- the loop. Fit on train (with validation for models
    that early-stop), predict the test block, collect predictions in long format
    and one diagnostic row per fold.

Everything is indexed by *position in the return panel*, never by date: dates are
attached at the end, when the results leave the harness. Position t is a
prediction origin -- the model standing at the close of day t, forecasting
r_{t+horizon} from information available up to and including t.

Exports:
  - Fold, make_folds(), make_folds_from_config()
  - WalkforwardData, Segment, align_graph()
  - WalkforwardResult, run_walkforward()

Integration: consumed by scripts/04_run_baselines.py and 05_run_gcn.py through
  the models of cryptognn.models, which implement the Forecaster protocol of
  cryptognn.evaluation.protocols.
Why it exists: an error in this file invalidates every result downstream and is
  discovered late (risk R2 of the plan). Concentrating fold geometry and data
  slicing in one module makes that risk testable in one test module -- which is
  the point of writing it before the models rather than alongside them.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from cryptognn.evaluation.protocols import Forecaster, SupportsDiagnostics
from cryptognn.windows import causal_windows

if TYPE_CHECKING:
    from cryptognn.config import Config

MODES = ("rolling", "expanding")


# --------------------------------------------------------------------------
# Fold geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Fold:
    """The three index blocks of one walk-forward fold.

    Positions index the rows of the return panel. The invariants are checked on
    construction rather than asserted by the caller: a malformed fold cannot come
    into existence, so every consumer downstream may take the ordering for
    granted. `eq=False` because the fields are arrays, and the generated __eq__
    would compare them elementwise and return an array.
    """

    index: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def __post_init__(self) -> None:
        for name in ("train", "val", "test"):
            block = np.asarray(getattr(self, name), dtype=int)
            object.__setattr__(self, name, block)
            if block.size == 0:
                raise ValueError(f"Fold {self.index}: empty {name} block")
            if not np.array_equal(block, np.arange(block[0], block[-1] + 1)):
                raise ValueError(f"Fold {self.index}: {name} block is not a contiguous range")

        if not (self.train[-1] < self.val[0] and self.val[-1] < self.test[0]):
            raise ValueError(
                f"Fold {self.index}: blocks out of order -- "
                f"train ends {self.train[-1]}, val spans {self.val[0]}-{self.val[-1]}, "
                f"test starts {self.test[0]}"
            )

    @property
    def sizes(self) -> tuple[int, int, int]:
        return len(self.train), len(self.val), len(self.test)


def make_folds(
    n_obs: int,
    train: int,
    val: int,
    test: int,
    step: int,
    offset: int = 0,
    mode: str = "rolling",
    horizon: int = 1,
) -> list[Fold]:
    """Lay out the walk-forward folds over `n_obs` panel rows.

    Args:
      n_obs: rows of the return panel.
      train, val, test: block lengths in observations.
      step: distance between the starts of consecutive folds.
      offset: first usable position. For this study it is `window - 1` = 59: no
        correlation graph exists before the sixtieth observation, so no earlier
        position can serve as a prediction origin.
      mode: "rolling" keeps the train block at fixed length; "expanding" starts
        every train block at `offset`, so it grows fold after fold.
      horizon: forecast horizon. The last position of a fold is a prediction
        origin, and its target lies `horizon` rows further on, so positions above
        `n_obs - 1 - horizon` are unusable. Spelled out as a parameter rather
        than subtracted by hand, because a silently misplaced -1 here shifts
        every target of the study by one day.

    Raises:
      ValueError: on an unknown mode, a non-positive length, a negative offset,
        or a parameter set that yields no fold at all -- returning an empty list
        would let the caller run a study on nothing.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown walk-forward mode {mode!r}; expected one of {MODES}")
    for name, value in (("train", train), ("val", val), ("test", test), ("step", step), ("horizon", horizon)):
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")

    span = train + val + test
    last_origin = n_obs - 1 - horizon

    folds: list[Fold] = []
    while True:
        start = offset + len(folds) * step
        if start + span - 1 > last_origin:
            break
        train_start = offset if mode == "expanding" else start
        folds.append(
            Fold(
                index=len(folds),
                train=np.arange(train_start, start + train),
                val=np.arange(start + train, start + train + val),
                test=np.arange(start + train + val, start + span),
            )
        )

    if not folds:
        raise ValueError(
            f"No fold fits: {n_obs} observations, offset {offset}, horizon {horizon} leave "
            f"{max(last_origin - offset + 1, 0)} usable positions for a span of {span} "
            f"({train} train + {val} val + {test} test)"
        )
    return folds


def make_folds_from_config(config: Config, n_obs: int) -> list[Fold]:
    """Fold layout of the study, with every parameter read from the config.

    The offset is derived here, and only here, as `graph.window - 1`: it is a
    consequence of the correlation window, not an independent setting, and
    writing it as a literal anywhere else would let the two drift apart.
    """
    return make_folds(
        n_obs=n_obs,
        train=config.walkforward.train,
        val=config.walkforward.val,
        test=config.walkforward.test,
        step=config.walkforward.step,
        offset=config.graph.window - 1,
        mode=config.walkforward.mode,
    )


# --------------------------------------------------------------------------
# Data container and views
# --------------------------------------------------------------------------


def align_graph(a_hat: np.ndarray, corr_index: pd.DatetimeIndex, dates: pd.DatetimeIndex) -> np.ndarray:
    """Re-index the graph tensor onto the rows of the return panel.

    The correlation tensor is shorter than the panel by `window - 1` rows: its
    first matrix belongs to the window closing on the sixtieth return. This maps
    each graph onto the position of its closing date -- the anchor of the
    anti-look-ahead discipline, since the graph at position t is then, by
    construction, built from returns up to and including t.

    Rows with no graph are filled with NaN rather than zeros or an identity:
    a zero adjacency is a valid-looking input that would train a model on a
    disconnected market, whereas NaN propagates and is caught by the guard in
    run_walkforward(). Missing data must fail loudly, not plausibly.

    The two indices reach this function with different timezone awareness --
    returns.parquet round-trips tz-aware UTC, while corr_index.npy is stored
    naive because .npy has no timezone concept -- and are reconciled here. Left
    alone, that mismatch makes every date look absent from the panel: a loud
    failure, but one every caller would otherwise have to fix for itself.
    """
    if a_hat.shape[0] != len(corr_index):
        raise ValueError(f"Graph tensor has {a_hat.shape[0]} matrices but corr_index has {len(corr_index)} dates")

    panel = pd.DatetimeIndex(dates)
    graph_dates = pd.DatetimeIndex(corr_index)
    if panel.tz is not None and graph_dates.tz is None:
        graph_dates = graph_dates.tz_localize(panel.tz)
    elif panel.tz is None and graph_dates.tz is not None:
        graph_dates = graph_dates.tz_localize(None)

    positions = panel.get_indexer(graph_dates)
    if (positions < 0).any():
        missing = graph_dates[positions < 0]
        raise ValueError(f"{len(missing)} correlation dates absent from the return panel, first {missing[0]}")

    aligned = np.full((len(dates), *a_hat.shape[1:]), np.nan, dtype=float)
    aligned[positions] = a_hat
    return aligned


@dataclass(frozen=True, eq=False)
class Segment:
    """The slice of the study data a model is allowed to see for one split.

    Every array is indexed by the same axis of `positions`, each of which is a
    prediction origin t. What the fields guarantee:

      - `returns`, `features`, `a_hat` at row i describe the state at t_i, so
        nothing in them postdates t_i.
      - `lags[i]` holds the L most recent returns *up to and including* t_i, in
        chronological order, so `lags[i, -1] == returns[i]`. This is what lets a
        VAR forecast the first day of a test block without reaching into the
        panel for the history sitting in the train block.
      - `y[i]` is the target r_{t_i + horizon}, and is the one field a model may
        not always have: run_walkforward() withholds it at prediction time via
        without_target(). See that method for why.
    """

    positions: np.ndarray
    dates: pd.DatetimeIndex
    target_dates: pd.DatetimeIndex
    assets: tuple[str, ...]
    returns: np.ndarray
    lags: np.ndarray
    y: np.ndarray
    features: np.ndarray | None = None
    a_hat: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.positions)

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    def without_target(self) -> Segment:
        """The same segment with `y` blanked to NaN, for the prediction call.

        A forecaster is fitted on a split whose target it must see, then asked to
        predict a split whose target it must not. Handing it one object for both
        makes `segment.y` reachable inside predict(), where reading it -- by a
        slip, not by design -- yields a perfect forecast and a result nobody can
        explain. Withholding the column costs one array copy per fold and removes
        the possibility: the value simply is not there, and a model that reaches
        for it produces NaN, which run_walkforward() rejects on the spot.

        Everything else is shared with the original, not copied: the arrays a
        model is *supposed* to read are large and are never written to.
        """
        return replace(self, y=np.full_like(self.y, np.nan))


@dataclass(frozen=True, eq=False)
class WalkforwardData:
    """The study's data, aligned on one axis so fold positions mean one thing.

    Every array is indexed by the rows of the return panel (T = 2006 for this
    study), including the graph tensor, which align_graph() pads to that length.
    A fold position is therefore simultaneously a row of `returns`, a row of
    `features`, a matrix of `a_hat` and a date of `dates` -- the property that
    makes the anti-look-ahead tests of S3.2 checkable at all.

    `features` and `a_hat` are optional so the harness is usable before the node
    features of S3.3 exist, and so the no-graph ablation of S4.1 can be run on a
    container that simply has none.
    """

    dates: pd.DatetimeIndex
    assets: tuple[str, ...]
    returns: np.ndarray
    graph_offset: int = 0
    lookback: int = 1
    horizon: int = 1
    features: np.ndarray | None = None
    a_hat: np.ndarray | None = None

    def __post_init__(self) -> None:
        n_obs, n_assets = self.returns.shape
        if len(self.dates) != n_obs:
            raise ValueError(f"{len(self.dates)} dates for {n_obs} return rows")
        if len(self.assets) != n_assets:
            raise ValueError(f"{len(self.assets)} asset names for {n_assets} return columns")
        if self.lookback < 1 or self.horizon < 1:
            raise ValueError(f"lookback and horizon must be positive, got {self.lookback} and {self.horizon}")
        if self.features is not None and self.features.shape[:2] != (n_obs, n_assets):
            raise ValueError(f"features {self.features.shape} do not align with returns {self.returns.shape}")
        if self.a_hat is not None and self.a_hat.shape != (n_obs, n_assets, n_assets):
            raise ValueError(f"a_hat {self.a_hat.shape} is not ({n_obs}, {n_assets}, {n_assets})")

    @property
    def n_obs(self) -> int:
        return self.returns.shape[0]

    @property
    def n_assets(self) -> int:
        return self.returns.shape[1]

    def segment(self, positions: np.ndarray) -> Segment:
        """The view of one split: rows at `positions`, plus their lag history.

        The lag history comes from causal_windows(), the study's single
        implementation of "the rows up to and including t": position t always
        maps to the window [t - lookback + 1, t], and early positions -- which
        have no full history -- carry NaN instead of silently borrowing rows from
        the wrong end. The study's own offset of 59 keeps every fold well clear
        of that boundary.
        """
        positions = np.asarray(positions, dtype=int)
        if positions.size == 0:
            raise ValueError("Cannot build a segment from an empty position array")
        if positions.min() < 0 or positions.max() + self.horizon >= self.n_obs:
            raise ValueError(
                f"Positions {positions.min()}-{positions.max()} leave the panel of {self.n_obs} rows "
                f"at horizon {self.horizon}"
            )

        return Segment(
            positions=positions,
            dates=self.dates[positions],
            target_dates=self.dates[positions + self.horizon],
            assets=self.assets,
            returns=self.returns[positions],
            lags=causal_windows(self.returns, self.lookback)[positions],  # (n, L, N), oldest first
            y=self.returns[positions + self.horizon],
            features=None if self.features is None else self.features[positions],
            a_hat=None if self.a_hat is None else self.a_hat[positions],
        )


# --------------------------------------------------------------------------
# The run loop
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class WalkforwardResult:
    """What one model produced over the whole walk-forward.

    `predictions` is the long table every metric, figure and backtest of Sprints
    3-5 is computed from. `diagnostics` carries what the run revealed about the
    model rather than about its errors -- the lag order a VAR selected, the
    epochs a GCN needed -- one row per fold, which is the granularity Section 6.4
    reports the VAR parameter count at.
    """

    predictions: pd.DataFrame
    diagnostics: pd.DataFrame


def _validate_fold(data: WalkforwardData, fold: Fold) -> None:
    """Reject a fold whose positions the data cannot honestly serve."""
    first = int(fold.train[0])
    if data.a_hat is not None and first < data.graph_offset:
        raise ValueError(
            f"Fold {fold.index} starts at position {first}, before the first available graph "
            f"at {data.graph_offset}: no correlation window closes there"
        )
    if fold.test[-1] + data.horizon >= data.n_obs:
        raise ValueError(f"Fold {fold.index} test block runs past the panel at horizon {data.horizon}")


def _check_finite(segment: Segment, fold: Fold, split: str) -> None:
    """Catch the NaN padding of align_graph() before a model trains on it."""
    for name in ("features", "a_hat"):
        block = getattr(segment, name)
        if block is not None and not np.isfinite(block).all():
            raise ValueError(
                f"Fold {fold.index}: non-finite values in {name} of the {split} split -- "
                "the split most likely reaches back before the first available graph"
            )


def run_walkforward(
    factory: Callable[[], Forecaster],
    data: WalkforwardData,
    folds: list[Fold],
    name: str | None = None,
    verbose: bool = True,
) -> WalkforwardResult:
    """Fit and score one model over every fold, returning predictions and diagnostics.

    A fresh forecaster is built per fold, from `factory`, so no state -- fitted
    parameters, standardization statistics, early-stopping history -- can survive
    from one fold into the next. Validation is passed to every model, whether or
    not it uses it: a single call signature keeps the baselines and the
    early-stopping GCN on one code path.

    fit() sees the targets of train and validation; predict() does not see the
    targets of test, which are withheld by Segment.without_target() and kept here
    for scoring.

    Only the test block is predicted, and only the test block is returned: train
    and validation predictions have no role in the comparison and their presence
    in the results table would be an invitation to report them.
    """
    predictions: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    n_assets = data.n_assets
    assets = np.asarray(data.assets)

    for fold in folds:
        _validate_fold(data, fold)
        train_segment = data.segment(fold.train)
        val_segment = data.segment(fold.val)
        test_segment = data.segment(fold.test)
        for segment, split in ((train_segment, "train"), (val_segment, "val"), (test_segment, "test")):
            _check_finite(segment, fold, split)

        model = factory()
        model_name = name if name is not None else getattr(model, "name", type(model).__name__)

        started = time.perf_counter()
        model.fit(train_segment, val_segment)
        fit_seconds = time.perf_counter() - started

        predicted = np.asarray(model.predict(test_segment.without_target()), dtype=float)
        if predicted.shape != (len(test_segment), n_assets):
            raise ValueError(
                f"Fold {fold.index}: {model_name} predicted {predicted.shape}, "
                f"expected {(len(test_segment), n_assets)}"
            )
        if not np.isfinite(predicted).all():
            raise ValueError(f"Fold {fold.index}: {model_name} produced non-finite predictions")

        predictions.append(
            pd.DataFrame(
                {
                    "fold": fold.index,
                    # The realized return's own date, not the origin's: it is the
                    # date the value is true of, and the one the backtest of S5.1
                    # books the position's payoff on.
                    "date": test_segment.target_dates.repeat(n_assets),
                    "asset": np.tile(assets, len(test_segment)),
                    "y_true": test_segment.y.ravel(),
                    "y_pred": predicted.ravel(),
                    "model": model_name,
                }
            )
        )

        n_train, n_val, n_test = fold.sizes
        row: dict[str, object] = {
            "fold": fold.index,
            "model": model_name,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "fit_seconds": fit_seconds,
        }
        if isinstance(model, SupportsDiagnostics):
            row.update(model.diagnostics())
        diagnostics.append(row)

        if verbose:
            print(
                f"  fold {fold.index:2d}  train {n_train:4d} val {n_val:3d} test {n_test:3d}  "
                f"test {test_segment.target_dates[0].date()} -> {test_segment.target_dates[-1].date()}  "
                f"{fit_seconds:6.2f}s"
            )

    return WalkforwardResult(
        predictions=pd.concat(predictions, ignore_index=True),
        diagnostics=pd.DataFrame(diagnostics),
    )

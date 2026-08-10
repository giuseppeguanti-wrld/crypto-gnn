"""Forecast accuracy metrics and the test that decides whether a gap is real.

Everything Section 6.4 reports is computed here: the error measures, the skill
score against the zero forecast, and the Diebold-Mariano test that turns "0.5%
better" into a statement with a p-value attached. On this data that last part is
not a formality -- the baselines sit within half a percent of MSE of one another,
which is exactly the regime where a table of point estimates invites conclusions
it cannot support.

Two conventions worth stating, because both are visible in the thesis:

  - **Directional accuracy** is computed where the return *and* the forecast are
    both non-zero, and reported next to the coverage. A forecast of exactly zero
    expresses no direction: scoring it wrong would say the model always picks the
    wrong side, when it picks no side. ZeroForecaster therefore reports NaN at
    coverage 0, which is the honest reading and the one the sign strategy of
    Sprint 5 will act on.
  - **Diebold-Mariano** is run on the loss differential averaged across the 15
    assets at each date, giving 1512 observations rather than 22 680. The
    22 680 are not independent -- crypto returns move together, and so do the
    errors of any model that follows them -- and pooling them as if they were
    would shrink the standard error by a factor of roughly sqrt(15), manufacturing
    significance out of cross-sectional correlation.

Exports:
  - rmse(), mae(), directional_accuracy(), skill_score()
  - diebold_mariano(): HLN-corrected DM statistic, p-value and degrees of freedom
  - panel_loss_differential(): long predictions -> one daily series per model pair
  - summarize_predictions(), diebold_mariano_matrix(): the tables of Section 6.4

Integration: consumed by scripts/04_run_baselines.py and, from Sprint 4, by
  05_run_gcn.py, which reuse summarize_predictions() so the baseline table and
  the final comparison cannot be computed two different ways.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def directional_accuracy(y: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Share of correctly signed forecasts, and the share of pairs it was computed on.

    Both zeros are excluded, symmetrically: a realized return of exactly zero has
    no direction to get right, and a forecast of exactly zero states none. The
    second returned value is the coverage -- without it, an accuracy computed on
    3% of the sample would look like an accuracy computed on all of it.

    Returns (nan, 0.0) when the model never takes a side, rather than 0.0, which
    would read as "always wrong".
    """
    y = np.asarray(y)
    pred = np.asarray(pred)
    taken = (y != 0) & (pred != 0)
    coverage = float(taken.mean()) if taken.size else 0.0

    if not taken.any():
        return float("nan"), coverage
    return float((np.sign(y[taken]) == np.sign(pred[taken])).mean()), coverage


def skill_score(y: np.ndarray, pred: np.ndarray, baseline_pred: np.ndarray) -> float:
    """1 - MSE(model)/MSE(baseline): the fraction of the baseline's squared error removed.

    Zero means the model matches the baseline, positive means it beats it,
    negative means it does worse. Against ZeroForecaster this is the direct form
    of the study's question, which is why the reference is that model and not,
    say, the historical mean.
    """
    baseline_mse = float(np.mean((np.asarray(y) - np.asarray(baseline_pred)) ** 2))
    if baseline_mse <= 0:
        raise ValueError("Baseline MSE is zero: the skill score is undefined against a perfect reference")
    return 1.0 - float(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)) / baseline_mse


@dataclass(frozen=True)
class DieboldMariano:
    """The outcome of one DM test, with what is needed to report it."""

    statistic: float
    p_value: float
    df: int
    mean_loss_differential: float


def _newey_west_variance(d: np.ndarray, lags: int) -> float:
    """Long-run variance of `d` with a Bartlett kernel over `lags` lags.

    For h = 1 the loop does not execute and this is the sample variance, which is
    what Diebold and Mariano's original one-step-ahead formulation uses: the loss
    differential of an optimal one-step forecast is serially uncorrelated.
    """
    centered = d - d.mean()
    n = len(d)
    variance = float(np.dot(centered, centered) / n)
    for lag in range(1, lags + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        variance += 2.0 * (1.0 - lag / (lags + 1.0)) * covariance
    return variance


def diebold_mariano(errors_a: np.ndarray, errors_b: np.ndarray, h: int = 1) -> DieboldMariano:
    """Test whether model A and model B have equal squared-error accuracy.

    The statistic is the mean of d_t = e_a^2 - e_b^2 over its own standard error,
    so it is **negative when A is the better model** -- A's losses are smaller.
    The variance is Newey-West with h-1 lags, and the whole statistic carries the
    Harvey-Leybourne-Newbold small-sample correction and is read against a
    t distribution with T-1 degrees of freedom rather than a normal. At T = 1512
    the correction changes little; it is applied because the citable test is HLN's
    version, and a test reported as HLN should be HLN.

    Raises rather than returning NaN when the series disagree in length or the
    estimated long-run variance is not positive: both mean the inputs are not
    what the test assumes, and a NaN there would travel silently into a table.
    """
    errors_a = np.asarray(errors_a, dtype=float)
    errors_b = np.asarray(errors_b, dtype=float)
    if errors_a.shape != errors_b.shape:
        raise ValueError(f"Error series differ in shape: {errors_a.shape} vs {errors_b.shape}")
    if errors_a.ndim != 1:
        raise ValueError(f"Expected one-dimensional error series, got {errors_a.ndim} dimensions")
    if h < 1:
        raise ValueError(f"Forecast horizon must be at least 1, got {h}")

    d = errors_a**2 - errors_b**2
    n = len(d)
    if n <= h:
        raise ValueError(f"Need more than h = {h} observations to test, got {n}")

    variance = _newey_west_variance(d, lags=h - 1)
    if variance <= 0:
        raise ValueError(
            "Non-positive long-run variance of the loss differential: the two models "
            "most likely produced identical errors, in which case there is nothing to test"
        )

    statistic = d.mean() / np.sqrt(variance / n)
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    statistic *= correction

    return DieboldMariano(
        statistic=float(statistic),
        p_value=float(2.0 * stats.t.sf(abs(statistic), df=n - 1)),
        df=n - 1,
        mean_loss_differential=float(d.mean()),
    )


# --------------------------------------------------------------------------
# From the long predictions table to the tables of Section 6.4
# --------------------------------------------------------------------------


def panel_loss_differential(predictions: pd.DataFrame, model_a: str, model_b: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-date squared-error series of two models, aligned on their common dates.

    Averaging the *squared* errors across assets, not the errors themselves:
    the test compares accuracy, and errors of opposite sign on two assets would
    otherwise cancel into an apparently perfect day.
    """
    frame = predictions[predictions["model"].isin([model_a, model_b])].copy()
    frame["squared_error"] = (frame["y_true"] - frame["y_pred"]) ** 2
    daily = frame.groupby(["model", "date"])["squared_error"].mean().unstack("model")

    for model in (model_a, model_b):
        if model not in daily.columns:
            raise ValueError(f"No predictions for model {model!r}")

    aligned = daily[[model_a, model_b]].dropna()
    # The DM below squares its inputs, so hand it root-mean-squared errors.
    return np.sqrt(aligned[model_a].to_numpy()), np.sqrt(aligned[model_b].to_numpy())


def summarize_predictions(predictions: pd.DataFrame, baseline: str = "zero") -> pd.DataFrame:
    """One row per model: RMSE, MAE, directional accuracy, coverage, skill score.

    The single aggregation point for every accuracy table in the study, so the
    baseline comparison of Sprint 3 and the full comparison of Sprint 4 cannot be
    computed two subtly different ways.
    """
    if baseline not in set(predictions["model"]):
        raise ValueError(f"Baseline model {baseline!r} absent from the predictions table")

    reference = predictions[predictions["model"] == baseline].set_index(["date", "asset"])["y_pred"]

    rows = []
    for model, group in predictions.groupby("model", sort=False):
        indexed = group.set_index(["date", "asset"])
        baseline_pred = reference.reindex(indexed.index)
        if baseline_pred.isna().any():
            raise ValueError(f"Model {model!r} covers dates or assets the baseline {baseline!r} does not")

        accuracy, coverage = directional_accuracy(indexed["y_true"], indexed["y_pred"])
        rows.append(
            {
                "model": model,
                "rmse": rmse(indexed["y_true"], indexed["y_pred"]),
                "mae": mae(indexed["y_true"], indexed["y_pred"]),
                "directional_accuracy": accuracy,
                "coverage": coverage,
                "skill_score": skill_score(indexed["y_true"], indexed["y_pred"], baseline_pred),
                "n_predictions": len(indexed),
            }
        )

    return pd.DataFrame(rows).sort_values("rmse", ignore_index=True)


def diebold_mariano_matrix(predictions: pd.DataFrame, h: int = 1) -> pd.DataFrame:
    """Every ordered pair of models tested against each other.

    Ordered, not just the upper triangle: the statistic is antisymmetric and the
    p-value symmetric, so the full matrix reads the same in both directions while
    letting a table be sliced by either model without transposing anything.
    """
    models = list(dict.fromkeys(predictions["model"]))

    rows = []
    for model_a in models:
        for model_b in models:
            if model_a == model_b:
                continue
            errors_a, errors_b = panel_loss_differential(predictions, model_a, model_b)
            result = diebold_mariano(errors_a, errors_b, h=h)
            rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "statistic": result.statistic,
                    "p_value": result.p_value,
                    "df": result.df,
                    "mean_loss_differential": result.mean_loss_differential,
                }
            )

    return pd.DataFrame(rows)

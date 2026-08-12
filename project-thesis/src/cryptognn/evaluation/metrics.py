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
  - holm_adjusted(): family-wise correction over a set of p-values
  - panel_loss_differential(): long predictions -> one daily series per model pair
  - summarize_predictions(), diebold_mariano_matrix(): the tables of Section 6.4
  - rank_association(): monotone association between two fold-level quantities

Integration: consumed by scripts/04_run_baselines.py and, from Sprint 4, by
  05_run_gcn.py, which reuse summarize_predictions() so the baseline table and
  the final comparison cannot be computed two different ways.
"""
from __future__ import annotations

from collections.abc import Sequence
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


def _accuracy_row(indexed: pd.DataFrame, baseline_pred: pd.Series) -> dict[str, float | int]:
    """Every accuracy measure of one model over one block of predictions.

    Extracted so the overall table and the per-asset and per-fold breakdowns are
    literally the same computation. A second implementation for the grouped case
    is how a table ends up disagreeing with its own subtotals.
    """
    accuracy, coverage = directional_accuracy(indexed["y_true"], indexed["y_pred"])
    return {
        "rmse": rmse(indexed["y_true"], indexed["y_pred"]),
        "mae": mae(indexed["y_true"], indexed["y_pred"]),
        "directional_accuracy": accuracy,
        "coverage": coverage,
        "skill_score": skill_score(indexed["y_true"], indexed["y_pred"], baseline_pred),
        "n_predictions": len(indexed),
    }


def summarize_predictions(
    predictions: pd.DataFrame,
    baseline: str = "zero",
    by: str | Sequence[str] | None = None,
) -> pd.DataFrame:
    """One row per model: RMSE, MAE, directional accuracy, coverage, skill score.

    The single aggregation point for every accuracy table in the study, so the
    baseline comparison of Sprint 3 and the full comparison of Sprint 4 cannot be
    computed two subtly different ways.

    `by` breaks the table down further -- "asset" for the per-asset view of
    Section 6.5, "fold" for the per-fold one that the skill-by-fold figure of
    Sprint 5 is drawn from. The skill score is then computed **inside each
    group**, against the baseline's predictions for that same group: the skill of
    a model on BTC means its improvement over forecasting zero on BTC, not its
    improvement over the whole panel's zero forecast, and the two differ whenever
    volatility varies across the grouping -- which on this data it does.

    Left at None the output is exactly what Sprint 3 produced, columns and order
    included, because four scripts and a test already depend on it.
    """
    if baseline not in set(predictions["model"]):
        raise ValueError(f"Baseline model {baseline!r} absent from the predictions table")

    keys = [by] if isinstance(by, str) else list(by or [])
    missing = [key for key in keys if key not in predictions.columns]
    if missing:
        raise ValueError(f"Cannot group by {missing}: absent from the predictions table")

    reference = predictions[predictions["model"] == baseline].set_index(["date", "asset"])["y_pred"]

    rows = []
    for values, group in predictions.groupby(["model", *keys], sort=False):
        labels = dict(zip(["model", *keys], values if isinstance(values, tuple) else (values,), strict=True))
        indexed = group.set_index(["date", "asset"])
        baseline_pred = reference.reindex(indexed.index)
        if baseline_pred.isna().any():
            raise ValueError(f"Model {labels['model']!r} covers dates or assets the baseline {baseline!r} does not")

        rows.append(labels | _accuracy_row(indexed, baseline_pred))

    return pd.DataFrame(rows).sort_values([*keys, "rmse"], ignore_index=True)


def holm_adjusted(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, in the input's order.

    Sort ascending, multiply the k-th smallest of m by (m - k), then enforce
    monotonicity so an adjusted value never falls below one that came before it,
    and clip at 1. Comparing the result to alpha controls the family-wise error
    rate exactly as the sequential procedure does, with the advantage that the
    number can be printed in a table.

    Holm rather than Bonferroni because it is uniformly more powerful and rests
    on no additional assumption: there is no reason to prefer the weaker of two
    tests that control the same thing.
    """
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1:
        raise ValueError(f"Expected a one-dimensional array of p-values, got {p_values.ndim} dimensions")
    if p_values.size == 0:
        return p_values

    order = np.argsort(p_values)
    ranked = p_values[order] * (len(p_values) - np.arange(len(p_values)))
    adjusted = np.empty_like(ranked)
    adjusted[order] = np.minimum(np.maximum.accumulate(ranked), 1.0)
    return adjusted


def diebold_mariano_matrix(predictions: pd.DataFrame, h: int = 1) -> pd.DataFrame:
    """Every ordered pair of models tested against each other.

    Ordered, not just the upper triangle: the statistic is antisymmetric and the
    p-value symmetric, so the full matrix reads the same in both directions while
    letting a table be sliced by either model without transposing anything.

    `p_value_holm` controls the family-wise error rate over the comparison as a
    whole. Seven models make 21 distinct tests, and at 5% roughly one of them is
    expected to come out significant on noise alone -- the data-snooping concern
    the thesis raises with `white2000reality`, arriving here in its most literal
    form. Reporting both columns lets Section 6.4 state a result as significant
    after accounting for how many tests were run, rather than leaving the reader
    to make the adjustment mentally.

    The correction is applied over the **unordered** pairs, then written onto
    both directions of each: the 42 rows contain 21 tests, and correcting over 42
    would count every test twice and inflate the adjustment. The family is every
    pair rather than only the comparisons against `zero`, which is the more
    conservative choice and spares the thesis an argument about which subfamily
    is the one that counts.
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
                    "pair": frozenset((model_a, model_b)),
                    "statistic": result.statistic,
                    "p_value": result.p_value,
                    "df": result.df,
                    "mean_loss_differential": result.mean_loss_differential,
                }
            )

    matrix = pd.DataFrame(rows)
    if matrix.empty:
        return matrix.drop(columns="pair")

    distinct = matrix.drop_duplicates("pair").set_index("pair")["p_value"]
    adjusted = pd.Series(holm_adjusted(distinct.to_numpy()), index=distinct.index)
    matrix["p_value_holm"] = matrix["pair"].map(adjusted)
    return matrix.drop(columns="pair")


# --------------------------------------------------------------------------
# Association between two fold-level quantities (Section 6.5)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RankAssociation:
    """A monotone association, with the Theil-Sen line a scatter is drawn through."""

    rho: float
    p_value: float
    slope: float
    intercept: float
    n: int


def rank_association(x: np.ndarray, y: np.ndarray) -> RankAssociation:
    """Spearman's rho between two series, plus a robust line through them.

    The test behind the figure that closes Section 6.5: is the GCN's skill related
    to how dense the graph was over the fold it was measured on? That is the
    study's two research questions meeting -- structure is most pronounced exactly
    when it is hardest to exploit -- so the association has to come with a p-value
    rather than with an eyeballed slope.

    Spearman rather than Pearson because the hypothesis is monotonicity, not
    linearity, and because at the calibrated threshold a third of the folds sit at
    a density of exactly 1: a tied block like that is a saturating measurement,
    which ranks handle and a product-moment correlation does not.

    The line is Theil-Sen, the median of the pairwise slopes, and not ordinary
    least squares. Both were tried on this data and they disagree in sign: two
    folds at the low-density end carry enough leverage to tilt the OLS line
    upward while rho is negative, which would put a figure at odds with the
    statistic printed on it. Theil-Sen is the rank-based estimator that belongs
    beside a rank-based correlation -- it depends on the ordering of the points
    rather than on their distances, so it cannot be levered by two of them, and
    it agrees with rho wherever rho has a direction to agree with. The claim is
    still carried by rho and its p-value; the line only makes the shape visible.

    Raises on fewer than three points or on a constant series: rho is undefined
    there, and scipy signals it with a NaN that would otherwise be printed as a
    result.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"Series differ in shape: {x.shape} vs {y.shape}")
    if x.ndim != 1:
        raise ValueError(f"Expected one-dimensional series, got {x.ndim} dimensions")
    if x.size < 3:
        raise ValueError(f"Need at least three points to associate, got {x.size}")
    for name, values in (("x", x), ("y", y)):
        if np.ptp(values) == 0:
            raise ValueError(f"Series {name} is constant: no association is defined against it")

    result = stats.spearmanr(x, y)
    line = stats.theilslopes(y, x)
    return RankAssociation(
        rho=float(result.statistic),
        p_value=float(result.pvalue),
        slope=float(line.slope),
        intercept=float(line.intercept),
        n=int(x.size),
    )

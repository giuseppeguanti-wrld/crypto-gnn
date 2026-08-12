"""Fixtures shared across the test packages.

The tree under tests/ mirrors src/cryptognn/, but these fixtures do not belong to
any one package. Two groups of them:

  - **Pipeline inputs.** graph/test_correlation, graph/test_threshold,
    graph/test_build, graph/test_metrics and data/test_stylized_facts all
    exercise different stages of the same pipeline and work from the same
    synthetic inputs -- a return panel with a known factor structure, and a small
    correlation matrix chosen to span the cases thresholding has to separate.
    evaluation/test_walkforward and test_features work from a third: a small
    study container whose values are readable by eye.
  - **Chapter outputs.** test_tables and test_summary format the *same*
    artifacts into two different documents, so they need one description of what
    a finished run looks like. Kept as one set rather than two: a config or a
    calibration that differed between them would let a table and the summary
    disagree in the fixtures while agreeing in production, which is the failure
    the pair of modules exists to catch.

The constants describing their shapes are in tests/synthetic.py, not here: see
that module for why a file other modules import from should not be a conftest.

Nothing here reads data/ or results/: the suite must run on a fresh clone,
before any script has been executed.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest
from synthetic import (
    BACKTEST_REFERENCE,
    DESCRIBED,
    GCN_ARMS,
    MODELS,
    N_ASSETS,
    N_FOLDS,
    SYMBOLS,
    WF_GRAPH_OFFSET,
    WF_LOOKBACK,
    WF_N_ASSETS,
    WF_N_FEATURES,
    WF_N_OBS,
)

from cryptognn.config import (
    ARConfig,
    BacktestConfig,
    Config,
    DataConfig,
    FeaturesConfig,
    GCNConfig,
    GraphConfig,
    ModelConfig,
    ThresholdConfig,
    VARConfig,
    WalkforwardConfig,
)
from cryptognn.evaluation.walkforward import WalkforwardData
from cryptognn.graph.threshold import TauCalibration


@pytest.fixture
def correlated_returns() -> pd.DataFrame:
    """A (400, 6) return panel with a strong common factor, so every pair is
    correlated at roughly 0.8 -- far enough from zero that a null which failed
    to break the dependence is unmistakable.
    """
    rng = np.random.default_rng(123)
    n_obs = 400
    market = rng.standard_normal((n_obs, 1))
    idiosyncratic = rng.standard_normal((n_obs, N_ASSETS))
    values = 0.02 * (2.0 * market + idiosyncratic)
    index = pd.date_range("2021-01-01", periods=n_obs, freq="D", tz="UTC")
    return pd.DataFrame(values, index=index, columns=[f"A{i}" for i in range(N_ASSETS)])


@pytest.fixture
def synthetic_volumes(correlated_returns: pd.DataFrame) -> pd.DataFrame:
    """A positive volume panel on the return panel's index, with assets whose
    units differ by orders of magnitude (column i scaled by 10^i).

    The spread is deliberate: the eighth node feature is the z-score of *log*
    volume, which should be blind to an asset's unit of account. A panel where
    every column had the same magnitude could not tell a correct implementation
    from one that happens to work only on comparable scales.
    """
    rng = np.random.default_rng(321)
    values = np.exp(rng.normal(loc=10.0, scale=1.0, size=correlated_returns.shape))
    scales = 10.0 ** np.arange(correlated_returns.shape[1])
    return pd.DataFrame(
        values * scales, index=correlated_returns.index, columns=correlated_returns.columns
    )


@pytest.fixture
def sample_corr() -> np.ndarray:
    """A 4x4 correlation matrix spanning the cases the threshold must separate:
    one strongly anticorrelated pair (-0.5), one weakly positive pair (0.1) that
    still falls below tau, and three pairs comfortably above it.
    """
    return np.array(
        [
            [1.0, 0.8, 0.3, -0.5],
            [0.8, 1.0, 0.6, 0.1],
            [0.3, 0.6, 1.0, 0.9],
            [-0.5, 0.1, 0.9, 1.0],
        ]
    )


@pytest.fixture
def synthetic_walkforward_data() -> WalkforwardData:
    """A (120, 4) study container whose every value announces where it came from.

    returns[t, j] == t + j/10, so a misalignment by one row or one asset is
    visible in the number itself rather than hidden in a plausible float: the
    target of position t must read t+1, and the most recent lag must read t.
    features[t, j, f] == returns[t, j] + 100*f keeps the feature axis equally
    identifiable, and the graph is NaN before WF_GRAPH_OFFSET exactly as
    align_graph() leaves it on the study data.
    """
    dates = pd.date_range("2021-01-01", periods=WF_N_OBS, freq="D", tz="UTC")
    returns = np.arange(WF_N_OBS)[:, None] + np.arange(WF_N_ASSETS)[None, :] / 10.0
    features = returns[:, :, None] + 100.0 * np.arange(WF_N_FEATURES)[None, None, :]

    a_hat = np.full((WF_N_OBS, WF_N_ASSETS, WF_N_ASSETS), np.nan)
    a_hat[WF_GRAPH_OFFSET:] = np.eye(WF_N_ASSETS)

    return WalkforwardData(
        dates=dates,
        assets=tuple(f"A{i}" for i in range(WF_N_ASSETS)),
        returns=returns,
        graph_offset=WF_GRAPH_OFFSET,
        lookback=WF_LOOKBACK,
        features=features,
        a_hat=a_hat,
    )


# --------------------------------------------------------------------------
# Chapter outputs: one description of a finished run, formatted two ways
# --------------------------------------------------------------------------


@pytest.fixture
def config() -> Config:
    """The study's own settings, not a reduced stand-in.

    The generated documents quote these numbers back -- 24 folds, four grid
    cells, 15 assets -- so a fixture with convenient small values would exercise
    formatting that production never reaches.
    """
    return Config(
        data=DataConfig(
            source="binance",
            quote="USDT",
            interval="1d",
            start=dt.date(2021, 1, 1),
            end=dt.date(2026, 6, 30),
            symbols=list(SYMBOLS),
        ),
        graph=GraphConfig(
            window=60,
            weight="mantegna",
            self_loops=True,
            threshold=ThresholdConfig(
                method="permutation",
                alpha=0.05,
                n_permutations=500,
                n_calibration_windows=24,
                statistic="pooled",
                tau_fixed=0.30,
            ),
        ),
        features=FeaturesConfig(lags=5, vol_windows=[5, 20], use_volume=True),
        walkforward=WalkforwardConfig(train=365, val=63, test=63, step=63, mode="rolling"),
        model=ModelConfig(
            gcn=GCNConfig(
                hidden=[16, 32],
                dropout=[0.2, 0.5],
                lr=0.005,
                weight_decay=5e-4,
                epochs=300,
                patience=30,
                seeds=[0, 1, 2, 3, 4],
            ),
            var=VARConfig(max_lag=5, ic="bic", fixed_lag=5),
            ar=ARConfig(max_lag=5, ic="bic"),
        ),
        backtest=BacktestConfig(cost_bps=10),
        seed=42,
    )


@pytest.fixture
def calibration() -> TauCalibration:
    return TauCalibration(
        tau=0.2145,
        tau_fwer=0.4311,
        tau_fixed=0.30,
        alpha=0.05,
        n_permutations=500,
        n_calibration_windows=24,
        window=60,
        n_pairs=105,
        statistic="pooled",
        seed=42,
        window_end_dates=["2021-03-02"],
        per_window_tau=[0.2145],
        per_window_tau_fwer=[0.4311],
        density={
            "tau": {"mean": 0.974, "min": 0.73, "max": 1.0, "sd": 0.05},
            "tau_fwer": {"mean": 0.887, "min": 0.37, "max": 1.0, "sd": 0.13},
            "tau_fixed": {"mean": 0.950, "min": 0.56, "max": 1.0, "sd": 0.08},
        },
    )


@pytest.fixture
def descriptive() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mean": [0.000345, 0.000383, 0.001332],
            "volatility_annualized": [0.5779, 0.7805, 0.7943],
            "skewness": [-0.1176, -0.2024, 0.8276],
            "excess_kurtosis": [3.999, 5.395, 24.296],
        },
        index=list(DESCRIBED),
    )


@pytest.fixture
def summary() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "model": list(MODELS),
            "rmse": np.linspace(0.0414, 0.0498, len(MODELS)),
            "mae": np.linspace(0.0276, 0.0347, len(MODELS)),
            # The zero forecast takes no side, so it has no directional accuracy.
            "directional_accuracy": [float("nan"), *rng.uniform(0.50, 0.52, len(MODELS) - 1)],
            "coverage": [0.0, *[0.995] * (len(MODELS) - 1)],
            "skill_score": np.linspace(0.0, -0.44, len(MODELS)),
            "n_predictions": [22680] * len(MODELS),
        }
    )


@pytest.fixture
def dm() -> pd.DataFrame:
    """Every ordered pair, not only the comparisons against the zero forecast.

    The results table reads the column against `zero`; the summary also reads the
    single gcn-against-gcn-nograph cell, which is the ablation the first research
    question turns on. A fixture holding only the first would let that lookup
    quietly find nothing and the section disappear.
    """
    rng = np.random.default_rng(11)
    rows = []
    for model_a in MODELS:
        for model_b in MODELS:
            if model_a == model_b:
                continue
            rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "statistic": float(rng.uniform(-3.0, 11.0)),
                    "p_value": float(rng.uniform(1e-26, 0.2)),
                    "df": 1511,
                    "mean_loss_differential": float(rng.uniform(1e-5, 8e-4)),
                    "p_value_holm": float(rng.uniform(0.01, 1.0)),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def backtest() -> pd.DataFrame:
    models = [*MODELS, BACKTEST_REFERENCE]
    rows = []
    for cost in (0.0, 10.0):
        for position, model in enumerate(models):
            rows.append(
                {
                    "model": model,
                    "cost_bps": cost,
                    "n_days": 1512,
                    "mean_turnover": 0.1 * position,
                    "annualized_return": -0.05 * position,
                    "annualized_volatility": 0.5,
                    # The zero forecast holds nothing, so its Sharpe is undefined.
                    "sharpe": float("nan") if model == "zero" else 0.5 - 0.1 * position,
                    "max_drawdown": -0.1 * position,
                    "cumulative_return": 0.2 - 0.1 * position,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-fold diagnostics of both run groups, carrying every column either document reads.

    The union rather than a minimal set per module: the two documents read
    overlapping subsets, and a column present in production but absent here is a
    formatting path the suite never exercises.
    """
    rng = np.random.default_rng(7)
    baseline_params = {"zero": float("nan"), "mean": 15.0, "ar": 17.67, "var-bic": 15.0, "var-p5": 1140.0}

    baseline_rows = []
    for fold in range(N_FOLDS):
        for model, n_params in baseline_params.items():
            baseline_rows.append(
                {
                    "fold": fold,
                    "model": model,
                    "n_params": n_params,
                    "fit_seconds": float(rng.uniform(1e-6, 0.02)),
                    # BIC selects order 0 on every fold: the finding the chapter reports.
                    "var_lag_order": {"var-bic": 0.0, "var-p5": 5.0}.get(model, float("nan")),
                    "obs_per_param": {"var-bic": 365.0, "var-p5": 4.802632}.get(model, float("nan")),
                    "ar_lag_mean": 0.178 if model == "ar" else float("nan"),
                    "ar_zero_order_share": 0.85 if model == "ar" else float("nan"),
                }
            )

    gcn_rows = []
    for fold in range(N_FOLDS):
        for arm in GCN_ARMS:
            row = {
                "fold": fold,
                "model": arm,
                "selected_hidden": int(rng.choice([16, 32])),
                "selected_dropout": float(rng.choice([0.2, 0.5])),
                "val_mse": float(rng.uniform(0.0017, 0.0018)),
                "epochs_mean": float(rng.uniform(50.0, 100.0)),
                "early_stopped_share": float(rng.uniform(0.95, 1.0)),
                "n_params": 247.0 if arm == "gcn" else 267.0,
                "n_fits": 20,
                "fit_seconds": float(rng.uniform(2.0, 4.0)),
            }
            for hidden in (16, 32):
                for dropout in (0.2, 0.5):
                    row[f"val_mse_h{hidden}_d{dropout}"] = float(rng.uniform(0.0017, 0.0018))
            gcn_rows.append(row)

    return pd.DataFrame(baseline_rows), pd.DataFrame(gcn_rows)

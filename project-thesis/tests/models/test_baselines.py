"""Tests for the four baselines of Section 6.4 (five runs, counting both VARs).

Each model is checked on data whose answer is known by construction: an AR(1)
panel with phi = 0.6, a VAR(1) panel with a chosen coefficient matrix, and white
noise, where the right behaviour is to select no lags at all. That last case is
not a corner: it is what the study's own data does, so a baseline that crashed
or quietly produced NaN there would take the whole comparison with it.

The VAR is additionally held to statsmodels' own forecast, row by row. Its
prediction is computed here from the coefficient matrix for speed and to keep
the input restricted to what the harness certifies as past, so any disagreement
with the library is a bug in this implementation, never a different model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.api import VAR

from cryptognn.config import load_config
from cryptognn.evaluation.protocols import Forecaster, SupportsDiagnostics
from cryptognn.evaluation.walkforward import WalkforwardData, make_folds, run_walkforward
from cryptognn.models import baseline_factories
from cryptognn.models.ar import PerAssetARForecaster
from cryptognn.models.naive import HistoricalMeanForecaster, ZeroForecaster
from cryptognn.models.var import VARForecaster
from cryptognn.paths import DEFAULT_CONFIG

N_ASSETS = 3
PHI = 0.6
# y_t = A y_{t-1} + e: asymmetric on purpose, so a transposed coefficient block
# would change the forecast instead of cancelling out.
VAR_COEFFICIENTS = np.array([[0.5, 0.2, 0.0], [0.0, 0.4, 0.1], [0.1, 0.0, 0.3]])

TRAIN, VAL, TEST = 150, 20, 20
BLOCKS = {"train": TRAIN, "val": VAL, "test": TEST, "step": TEST}


def as_data(values: np.ndarray, lookback: int = 5) -> WalkforwardData:
    """Wrap a raw return panel in the container the models are fed from."""
    return WalkforwardData(
        dates=pd.date_range("2021-01-01", periods=len(values), freq="D", tz="UTC"),
        assets=tuple(f"A{i}" for i in range(values.shape[1])),
        returns=values,
        lookback=lookback,
    )


@pytest.fixture
def config():
    return load_config(DEFAULT_CONFIG)


@pytest.fixture
def white_noise() -> np.ndarray:
    rng = np.random.default_rng(5)
    return rng.standard_normal((400, N_ASSETS)) * 0.02


@pytest.fixture
def ar_panel() -> np.ndarray:
    """Three independent AR(1) series with phi = 0.6 -- strong enough that any
    criterion refusing to select a lag would be the thing under suspicion.
    """
    rng = np.random.default_rng(7)
    values = np.zeros((400, N_ASSETS))
    noise = rng.standard_normal((400, N_ASSETS)) * 0.02
    for t in range(1, len(values)):
        values[t] = PHI * values[t - 1] + noise[t]
    return values


@pytest.fixture
def var_panel() -> np.ndarray:
    rng = np.random.default_rng(9)
    values = np.zeros((400, N_ASSETS))
    noise = rng.standard_normal((400, N_ASSETS)) * 0.02
    for t in range(1, len(values)):
        values[t] = VAR_COEFFICIENTS @ values[t - 1] + noise[t]
    return values


class TestProtocolConformance:
    def test_every_baseline_satisfies_the_forecaster_protocol(self, config):
        factories = baseline_factories(config)

        assert list(factories) == ["zero", "mean", "ar", "var-bic", "var-p5"]
        for name, factory in factories.items():
            model = factory()
            assert isinstance(model, Forecaster)
            assert model.name == name

    def test_every_baseline_runs_through_the_harness(self, config, var_panel):
        data = as_data(var_panel)
        folds = make_folds(n_obs=data.n_obs, offset=5, **BLOCKS)

        for name, factory in baseline_factories(config).items():
            result = run_walkforward(factory, data, folds, verbose=False)

            assert result.predictions["model"].unique().tolist() == [name]
            assert len(result.predictions) == len(folds) * TEST * N_ASSETS
            assert np.isfinite(result.predictions["y_pred"]).all()

    def test_models_with_something_to_report_expose_diagnostics(self, config, var_panel):
        data = as_data(var_panel)
        fold = make_folds(n_obs=data.n_obs, offset=5, **BLOCKS)[0]
        train, val = data.segment(fold.train), data.segment(fold.val)

        for factory in (baseline_factories(config)[name] for name in ("mean", "ar", "var-bic", "var-p5")):
            model = factory()
            model.fit(train, val)
            assert isinstance(model, SupportsDiagnostics)
            assert model.diagnostics()

        assert not isinstance(ZeroForecaster(), SupportsDiagnostics)


class TestNaiveForecasters:
    def test_zero_forecasts_exactly_zero(self, white_noise):
        data = as_data(white_noise)
        segment = data.segment(np.arange(10, 30))
        model = ZeroForecaster()
        model.fit(segment, segment)

        predicted = model.predict(segment)

        assert predicted.shape == (20, N_ASSETS)
        assert (predicted == 0.0).all()

    def test_historical_mean_is_the_train_mean_and_nothing_else(self, white_noise):
        data = as_data(white_noise)
        train, later = data.segment(np.arange(10, 160)), data.segment(np.arange(160, 180))
        model = HistoricalMeanForecaster()
        model.fit(train, later)

        predicted = model.predict(later)

        np.testing.assert_allclose(predicted, np.tile(white_noise[10:160].mean(axis=0), (20, 1)))
        # One row is the whole story: the forecast does not vary with the origin.
        assert len(np.unique(predicted, axis=0)) == 1

    def test_historical_mean_predict_before_fit_raises(self, white_noise):
        with pytest.raises(RuntimeError, match="before fit"):
            HistoricalMeanForecaster().predict(as_data(white_noise).segment(np.arange(10, 30)))


class TestPerAssetAR:
    def test_recovers_a_known_autoregression(self, config, ar_panel):
        data = as_data(ar_panel)
        train, later = data.segment(np.arange(10, 300)), data.segment(np.arange(300, 320))
        model = PerAssetARForecaster(config)
        model.fit(train, later)

        assert (model.orders_ >= 1).all()
        np.testing.assert_allclose(model.coef_[:, 0], PHI, atol=0.1)

        # The forecast is the fitted recursion applied to the segment's own lags.
        order = int(model.orders_.max())
        expected = model.const_ + np.einsum(
            "nka,ak->na", later.lags[:, -order:, :][:, ::-1, :], model.coef_[:, :order]
        )
        np.testing.assert_allclose(model.predict(later), expected)

    def test_order_zero_on_white_noise_forecasts_the_intercept(self, config, white_noise):
        data = as_data(white_noise)
        train, later = data.segment(np.arange(10, 300)), data.segment(np.arange(300, 320))
        model = PerAssetARForecaster(config)
        model.fit(train, later)

        predicted = model.predict(later)

        assert (model.orders_ == 0).all()
        assert model.diagnostics()["ar_zero_order_share"] == 1.0
        assert np.isfinite(predicted).all()
        np.testing.assert_allclose(predicted, np.tile(model.const_, (20, 1)))

    def test_rejects_a_segment_shallower_than_the_selected_order(self, config, ar_panel):
        model = PerAssetARForecaster(config)
        deep = as_data(ar_panel, lookback=5)
        model.fit(deep.segment(np.arange(10, 300)), deep.segment(np.arange(300, 320)))

        shallow = as_data(ar_panel, lookback=1).segment(np.arange(300, 320))
        if int(model.orders_.max()) > 1:
            with pytest.raises(ValueError, match="exceeds the 1 lags"):
                model.predict(shallow)


class TestVAR:
    def test_matches_statsmodels_forecast_row_by_row(self, config, var_panel):
        """The contraction here and results.forecast() are the same algebra; if
        they disagree, the coefficient block has been reshaped the wrong way
        round, which no amount of plausible-looking output would reveal.
        """
        data = as_data(var_panel)
        train, later = data.segment(np.arange(10, 300)), data.segment(np.arange(300, 320))
        model = VARForecaster(config, lags=1)
        model.fit(train, later)

        reference_results = VAR(train.returns).fit(1)
        predicted = model.predict(later)

        for row, position in enumerate(later.positions):
            history = var_panel[position - reference_results.k_ar + 1 : position + 1]
            expected = reference_results.forecast(history, steps=1)[0]
            np.testing.assert_allclose(predicted[row], expected, atol=1e-12)

    def test_recovers_the_generating_coefficient_matrix(self, config, var_panel):
        data = as_data(var_panel)
        train = data.segment(np.arange(10, 390))
        model = VARForecaster(config, lags=1)
        model.fit(train, train)

        # coef_ is (lag, source, target); the generating matrix maps source to
        # target the other way round, hence the transpose.
        np.testing.assert_allclose(model.coef_[0].T, VAR_COEFFICIENTS, atol=0.08)

    def test_fixed_order_estimates_every_lag_it_was_given(self, config, var_panel):
        data = as_data(var_panel)
        train = data.segment(np.arange(10, 300))
        model = VARForecaster(config, lags=config.model.var.fixed_lag)
        model.fit(train, train)

        diagnostics = model.diagnostics()
        assert model.name == "var-p5"
        assert model.k_ar_ == 5
        assert diagnostics["n_params"] == N_ASSETS * (N_ASSETS * 5 + 1)
        assert diagnostics["obs_per_param"] == pytest.approx(290 * N_ASSETS / diagnostics["n_params"])

    def test_bic_selects_no_lag_on_white_noise_and_forecasts_the_intercept(self, config, white_noise):
        """The study's own data behaves this way on every fold, so the branch
        that handles it is the main path, not a safety net.
        """
        data = as_data(white_noise)
        train, later = data.segment(np.arange(10, 300)), data.segment(np.arange(300, 320))
        model = VARForecaster(config)
        model.fit(train, later)

        predicted = model.predict(later)

        assert model.name == "var-bic"
        assert model.k_ar_ == 0
        assert model.diagnostics()["n_params"] == N_ASSETS
        np.testing.assert_allclose(predicted, np.tile(model.const_, (20, 1)))

    def test_rejects_a_segment_shallower_than_the_fitted_order(self, config, var_panel):
        model = VARForecaster(config, lags=5)
        deep = as_data(var_panel, lookback=5)
        model.fit(deep.segment(np.arange(10, 300)), deep.segment(np.arange(300, 320)))

        shallow = as_data(var_panel, lookback=2).segment(np.arange(300, 320))
        with pytest.raises(ValueError, match="exceeds the 2 lags"):
            model.predict(shallow)


class TestBaselinesIgnoreTheFuture:
    @pytest.mark.parametrize("name", ["zero", "mean", "ar", "var-bic", "var-p5"])
    def test_forecasts_are_invariant_to_what_happens_after_the_test_block(
        self, config, var_panel, name
    ):
        """The S3.2 corruption test, applied to models that use history for real:
        replacing the panel beyond the last prediction origin must not move a
        single forecast.
        """
        factory = baseline_factories(config)[name]
        data = as_data(var_panel)
        fold = make_folds(n_obs=data.n_obs, offset=5, **BLOCKS)[0]
        cutoff = int(fold.test[-1])

        corrupted = var_panel.copy()
        rng = np.random.default_rng(21)
        corrupted[cutoff + 2 :] = rng.normal(5.0, 2.0, size=corrupted[cutoff + 2 :].shape)

        baseline = run_walkforward(factory, data, [fold], verbose=False)
        rerun = run_walkforward(factory, as_data(corrupted), [fold], verbose=False)

        pd.testing.assert_frame_equal(baseline.predictions, rerun.predictions)

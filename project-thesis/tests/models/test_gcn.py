"""Tests for the GCN of Section 6.3: the architecture (S4.1) and its training (S4.2).

The three architectural properties the plan names -- output shape, permutation
equivariance, spectrum of A_hat -- plus the ones without which those three would
pass on a model that is not a graph convolution at all. Two failure modes matter
here and neither is visible in a loss curve:

  - a layer that **ignores** A_hat still has the right shape and is still
    trivially equivariant, and would make the no-graph ablation compare a model
    against itself;
  - a per-node weight instead of the shared one still trains, still predicts,
    and breaks the equivariance the thesis proves in Chapter 2.

`test_graph_actually_propagates` and `test_permutation_equivariance` exist to
fail in those two cases, and were checked against both mutations.

The training half adds two of the same kind: a standardizer fitted on more than
the training split, and an early stopping that keeps the last weights rather than
the best. Both leave a model that trains, predicts and logs plausibly, so
`test_standardization_uses_the_train_split_only` and
`test_early_stopping_restores_the_best_weights` were checked against them too.

The adjacency used throughout comes from the pipeline's own chain
(mantegna_weights -> apply_threshold -> normalized_adjacency), not from an
invented matrix: the model is tested on the substrate it will actually be fed.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from cryptognn.config import Config, load_config
from cryptognn.evaluation.protocols import Forecaster, SupportsDiagnostics
from cryptognn.evaluation.walkforward import WalkforwardData, make_folds, run_walkforward
from cryptognn.graph.build import apply_threshold, mantegna_weights, normalized_adjacency
from cryptognn.models.gcn import (
    GCN2,
    GCNForecaster,
    GCNGridForecaster,
    GCNLayer,
    gcn_factories,
    seed_everything,
)
from cryptognn.paths import DEFAULT_CONFIG

N_ASSETS = 6
N_FEATURES = 8
BATCH = 5
TAU = 0.2145
SEED = 42

# Geometry of the synthetic walk-forward container of the training tests. Small
# enough that a fit is milliseconds, long enough for two folds.
WF_N_OBS = 260
WF_N_ASSETS = 4
WF_N_FEATURES = 3
WF_OFFSET = 5
BLOCKS = {"train": 120, "val": 40, "test": 40, "step": 40}


def tuned(config: Config, **overrides: float) -> Config:
    """The shipped config with a few GCN training settings replaced.

    The tests need shorter runs and a shorter patience than the study's 300
    epochs; overriding them here rather than hardcoding alternatives in the model
    keeps config/default.yaml the only place the *study's* numbers live.
    """
    return replace(config, model=replace(config.model, gcn=replace(config.model.gcn, **overrides)))


@pytest.fixture
def adjacency() -> np.ndarray:
    """A (BATCH, N, N) renormalized adjacency built the way the pipeline builds it.

    The correlations come from a factor model with a strong common component, so
    the thresholded graph is connected -- an isolated node would make
    normalized_adjacency() raise, and this fixture is not the place to exercise
    that path.
    """
    rng = np.random.default_rng(SEED)
    market = rng.standard_normal((BATCH, 200, 1))
    panels = 2.0 * market + rng.standard_normal((BATCH, 200, N_ASSETS))

    corr = np.stack([np.corrcoef(panel, rowvar=False) for panel in panels])
    weights = apply_threshold(corr, mantegna_weights(corr), TAU)
    return normalized_adjacency(weights)


@pytest.fixture
def features() -> np.ndarray:
    return np.random.default_rng(SEED + 1).standard_normal((BATCH, N_ASSETS, N_FEATURES))


def as_double(model: GCN2) -> GCN2:
    """The model in float64 and eval mode: no dropout, no float32 noise.

    Properties like equivariance hold exactly in the algebra, so the test should
    fail only when the algebra is wrong. In float32 the tolerance would have to
    be loose enough to hide a real asymmetry.
    """
    return model.double().eval()


class TestGCNLayer:
    def test_matches_the_explicit_algebra(self, adjacency, features):
        """The layer is A_hat @ (H W) + b and nothing else -- eq:gcn, literally.

        Computed here in NumPy from the layer's own parameters, so a transposed
        weight, a missing propagation or a bias added on the wrong side of the
        multiplication shows up as a mismatch rather than as plausible output.
        """
        seed_everything(SEED)
        layer = GCNLayer(N_FEATURES, 4).double()

        output = layer(torch.from_numpy(adjacency), torch.from_numpy(features))

        weight = layer.weight.detach().numpy()
        bias = layer.bias.detach().numpy()
        expected = adjacency @ (features @ weight) + bias
        np.testing.assert_allclose(output.detach().numpy(), expected, atol=1e-12)

    def test_rejects_mismatched_inputs(self, adjacency, features):
        layer = GCNLayer(N_FEATURES, 4).double()
        a_hat, h = torch.from_numpy(adjacency), torch.from_numpy(features)

        with pytest.raises(ValueError, match="Expected batched"):
            layer(a_hat[0], h[0])
        with pytest.raises(ValueError, match="does not match features"):
            layer(a_hat[:, :-1, :-1], h)
        with pytest.raises(ValueError, match=f"takes {N_FEATURES} input features"):
            layer(a_hat, h[..., :-1])


class TestGCN2:
    def test_output_shape(self, adjacency, features):
        """(B, N, F) -> (B, N): one forecast per asset, the shape the harness scores."""
        a_hat, h = torch.from_numpy(adjacency), torch.from_numpy(features)

        for hidden in (16, 32):
            for use_graph in (True, False):
                model = as_double(GCN2(N_FEATURES, hidden=hidden, dropout=0.2, use_graph=use_graph))
                assert model(a_hat, h).shape == (BATCH, N_ASSETS)

    def test_permutation_equivariance(self, adjacency, features):
        """f(P A P', P H) == P f(A, H): relabelling the assets relabels the forecasts.

        The experimental form of the property proved in Chapter 2, and the reason
        the layer's weight is shared across nodes. Giving each node its own weight
        makes this test fail, which is what it is for.
        """
        seed_everything(SEED)
        model = as_double(GCN2(N_FEATURES, hidden=16, dropout=0.5))
        permutation = np.random.default_rng(SEED).permutation(N_ASSETS)

        original = model(torch.from_numpy(adjacency), torch.from_numpy(features)).detach().numpy()
        permuted = model(
            torch.from_numpy(adjacency[:, permutation][:, :, permutation]),
            torch.from_numpy(features[:, permutation]),
        ).detach().numpy()

        np.testing.assert_allclose(permuted, original[:, permutation], atol=1e-10)

    def test_graph_actually_propagates(self, adjacency, features):
        """Perturbing one node's features moves its neighbours' forecasts.

        Without this, a forward() that dropped A_hat entirely would still have the
        right shape, still be equivariant, and would silently turn the whole study
        into a comparison of an MLP with itself.
        """
        model = as_double(GCN2(N_FEATURES, hidden=16, dropout=0.0))
        a_hat, h = torch.from_numpy(adjacency), torch.from_numpy(features)

        perturbed = features.copy()
        perturbed[:, 0, :] += 10.0

        baseline = model(a_hat, h).detach().numpy()
        moved = model(a_hat, torch.from_numpy(perturbed)).detach().numpy()

        neighbours = adjacency[0, 0] > 0
        neighbours[0] = False
        assert neighbours.any(), "fixture graph has an isolated node; the test would be vacuous"
        assert not np.allclose(moved[0, neighbours], baseline[0, neighbours])

    def test_rejects_invalid_hyperparameters(self):
        with pytest.raises(ValueError, match="must be positive"):
            GCN2(N_FEATURES, hidden=0, dropout=0.2)
        with pytest.raises(ValueError, match=r"dropout must lie in \[0, 1\)"):
            GCN2(N_FEATURES, hidden=16, dropout=1.0)


class TestNoGraphAblation:
    """The ablation isolates the graph, so it must differ in the graph alone."""

    def test_ignores_the_adjacency_it_is_given(self, adjacency, features):
        model = as_double(GCN2(N_FEATURES, hidden=16, dropout=0.0, use_graph=False))
        h = torch.from_numpy(features)

        rng = np.random.default_rng(SEED + 2)
        random_weights = np.abs(rng.standard_normal(adjacency.shape))
        random_weights = (random_weights + np.swapaxes(random_weights, -1, -2)) / 2.0
        np.einsum("...ii->...i", random_weights)[...] = 0.0
        unrelated = normalized_adjacency(random_weights)
        assert not np.allclose(unrelated, adjacency), "the two graphs must differ for the test to say anything"

        np.testing.assert_allclose(
            model(torch.from_numpy(adjacency), h).detach().numpy(),
            model(torch.from_numpy(unrelated), h).detach().numpy(),
            atol=1e-12,
        )

    def test_equals_the_graph_model_fed_the_identity(self, adjacency, features):
        """A_hat = I is what the flag substitutes, so the two must coincide exactly."""
        seed_everything(SEED)
        graph_model = as_double(GCN2(N_FEATURES, hidden=16, dropout=0.0, use_graph=True))
        seed_everything(SEED)
        ablated = as_double(GCN2(N_FEATURES, hidden=16, dropout=0.0, use_graph=False))

        identity = np.broadcast_to(np.eye(N_ASSETS), adjacency.shape).copy()
        h = torch.from_numpy(features)

        np.testing.assert_allclose(
            ablated(torch.from_numpy(adjacency), h).detach().numpy(),
            graph_model(torch.from_numpy(identity), h).detach().numpy(),
            atol=1e-12,
        )

    def test_has_the_same_capacity(self):
        """Equal parameter counts: the comparison is about the graph, not model size."""
        with_graph = GCN2(N_FEATURES, hidden=16, dropout=0.2, use_graph=True)
        without = GCN2(N_FEATURES, hidden=16, dropout=0.2, use_graph=False)

        assert with_graph.n_parameters() == without.n_parameters()
        # F*h + h  +  h*1 + 1, the shared weights of two layers.
        assert with_graph.n_parameters() == N_FEATURES * 16 + 16 + 16 + 1


class TestRenormalization:
    def test_renormalized_spectrum(self, adjacency):
        """Eigenvalues of A_hat inside [-1, 1] -- the point of the renormalization trick.

        It is what keeps a two-layer stack from amplifying or extinguishing the
        signal, so it is checked on the substrate the model consumes rather than
        trusted from the construction module.
        """
        eigenvalues = np.linalg.eigvalsh(adjacency)

        assert eigenvalues.min() >= -1.0 - 1e-9
        assert eigenvalues.max() <= 1.0 + 1e-9
        # The largest eigenvalue of a symmetrically normalized adjacency is 1.
        np.testing.assert_allclose(eigenvalues.max(axis=-1), 1.0, atol=1e-9)


class TestDeterminism:
    def test_dropout_is_training_only(self, adjacency, features):
        """eval() repeats itself, train() with dropout does not.

        Dropout left active at prediction time makes every forecast a sample
        rather than an estimate -- silent, and fatal to the five-seed averaging
        of S4.3.
        """
        seed_everything(SEED)
        model = GCN2(N_FEATURES, hidden=16, dropout=0.5).double()
        a_hat, h = torch.from_numpy(adjacency), torch.from_numpy(features)

        model.eval()
        np.testing.assert_allclose(model(a_hat, h).detach().numpy(), model(a_hat, h).detach().numpy())

        model.train()
        assert not np.allclose(model(a_hat, h).detach().numpy(), model(a_hat, h).detach().numpy())

    def test_same_seed_same_initialization(self):
        def initial_weights(seed: int) -> np.ndarray:
            seed_everything(seed)
            return GCN2(N_FEATURES, hidden=16, dropout=0.2).layer1.weight.detach().numpy().copy()

        np.testing.assert_array_equal(initial_weights(SEED), initial_weights(SEED))
        assert not np.allclose(initial_weights(SEED), initial_weights(SEED + 1))


# --------------------------------------------------------------------------
# S4.2 -- training
# --------------------------------------------------------------------------


@pytest.fixture
def config() -> Config:
    return load_config(DEFAULT_CONFIG)


def make_learnable_data(seed: int = 11) -> tuple[WalkforwardData, np.ndarray]:
    """A study container whose target the graph model can actually learn.

    The signal is `3 * (A_hat @ f0)`, the neighbourhood average of the first node
    feature: exactly the quantity one graph convolution computes, so a working
    GCN can recover it and a broken one cannot hide behind an unlearnable target.
    "The model runs" is a weaker claim than "the model learns", and only the
    second distinguishes a training loop from a random number generator.

    Returns the container and the raw feature tensor, which the look-ahead test
    needs in order to corrupt it.
    """
    rng = np.random.default_rng(seed)
    features = rng.standard_normal((WF_N_OBS, WF_N_ASSETS, WF_N_FEATURES))

    weights = np.abs(rng.standard_normal((WF_N_ASSETS, WF_N_ASSETS))) + 0.5
    weights = (weights + weights.T) / 2.0
    np.fill_diagonal(weights, 0.0)
    a_hat = np.broadcast_to(normalized_adjacency(weights), (WF_N_OBS, WF_N_ASSETS, WF_N_ASSETS)).copy()

    signal = 3.0 * np.einsum("ij,tj->ti", a_hat[0], features[:, :, 0])
    returns = np.empty((WF_N_OBS, WF_N_ASSETS))
    returns[0] = 0.1 * rng.standard_normal(WF_N_ASSETS)
    returns[1:] = signal[:-1] + 0.1 * rng.standard_normal((WF_N_OBS - 1, WF_N_ASSETS))

    data = WalkforwardData(
        dates=pd.date_range("2021-01-01", periods=WF_N_OBS, freq="D", tz="UTC"),
        assets=tuple(f"A{i}" for i in range(WF_N_ASSETS)),
        returns=returns,
        lookback=5,
        features=features,
        a_hat=a_hat,
    )
    return data, features


@pytest.fixture
def learnable_data() -> WalkforwardData:
    return make_learnable_data()[0]


@pytest.fixture
def folds(learnable_data):
    return make_folds(n_obs=learnable_data.n_obs, offset=WF_OFFSET, **BLOCKS)


def fitted(config: Config, data: WalkforwardData, fold, **kwargs) -> GCNForecaster:
    """A forecaster trained on one fold, with the fold's own splits."""
    model = GCNForecaster(config, hidden=kwargs.pop("hidden", 16), dropout=kwargs.pop("dropout", 0.0), **kwargs)
    model.fit(data.segment(fold.train), data.segment(fold.val))
    return model


class TestForecasterContract:
    def test_conforms_to_the_protocols(self, config):
        model = GCNForecaster(config, hidden=16, dropout=0.2)

        assert isinstance(model, Forecaster)
        assert isinstance(model, SupportsDiagnostics)
        assert model.name == "gcn"
        assert GCNForecaster(config, hidden=16, dropout=0.2, use_graph=False).name == "gcn-nograph"

    def test_runs_through_the_harness(self, config, learnable_data, folds):
        settings = tuned(config, epochs=40, patience=10)
        result = run_walkforward(
            lambda: GCNForecaster(settings, hidden=16, dropout=0.2), learnable_data, folds, verbose=False
        )

        assert len(result.predictions) == len(folds) * BLOCKS["test"] * WF_N_ASSETS
        assert np.isfinite(result.predictions["y_pred"]).all()
        assert result.predictions["model"].unique().tolist() == ["gcn"]
        assert len(result.diagnostics) == len(folds)
        for column in ("epochs_run", "best_epoch", "train_mse", "val_mse", "n_params", "seed", "use_graph"):
            assert column in result.diagnostics.columns

    def test_predict_before_fit_raises(self, config, learnable_data, folds):
        with pytest.raises(RuntimeError, match="before fit"):
            GCNForecaster(config, hidden=16, dropout=0.2).predict(learnable_data.segment(folds[0].test))

    def test_requires_features_and_a_graph(self, config, learnable_data, folds):
        bare = replace(learnable_data, features=None, a_hat=None)
        train, val = bare.segment(folds[0].train), bare.segment(folds[0].val)

        with pytest.raises(ValueError, match="features and a_hat"):
            GCNForecaster(config, hidden=16, dropout=0.2).fit(train, val)


class TestTraining:
    def test_learns_a_learnable_signal(self, config, learnable_data, folds):
        """The fitted model beats a forecast of zero by a wide margin.

        The bar is deliberately not "some improvement": on a target with this
        signal-to-noise ratio a working GCN cuts the RMSE several times over, so
        a marginal gain would mean something is wrong rather than that the test
        is strict.
        """
        fold = folds[0]
        model = fitted(tuned(config, epochs=400, patience=40), learnable_data, fold)
        test = learnable_data.segment(fold.test)

        predicted = model.predict(test.without_target())
        error = np.sqrt(np.mean((test.y - predicted) ** 2))
        zero_error = np.sqrt(np.mean(test.y**2))

        assert error < 0.5 * zero_error

    def test_early_stopping_restores_the_best_weights(self, config, learnable_data, folds):
        """The model returned is the one from the best epoch, not from the last.

        Keeping the final weights is the natural mistake, and it is invisible:
        the run still stops early, still logs a best epoch, and is merely worse.
        The check is that the validation error of the *returned* model equals the
        best value recorded, while the last epoch's was worse.
        """
        fold = folds[0]
        settings = tuned(config, epochs=300, patience=5)
        model = fitted(settings, learnable_data, fold, dropout=0.5)
        val = learnable_data.segment(fold.val)

        diagnostics = model.diagnostics()
        assert diagnostics["epochs_run"] < settings.model.gcn.epochs
        assert diagnostics["early_stopped"] == 1
        assert diagnostics["best_epoch"] < diagnostics["epochs_run"]

        restored = float(np.mean((val.y - model.predict(val.without_target())) ** 2))
        assert restored == pytest.approx(model.best_val_mse_, rel=1e-4)
        assert model.history_[-1]["val_mse"] > model.best_val_mse_

    def test_records_a_training_history(self, config, learnable_data, folds):
        model = fitted(tuned(config, epochs=30, patience=30), learnable_data, folds[0])

        assert len(model.history_) == model.epochs_run_ == 30
        assert model.history_[-1]["train_mse"] < model.history_[0]["train_mse"]
        assert model.diagnostics()["early_stopped"] == 0

    def test_ablation_differs_from_the_graph_model(self, config, learnable_data, folds):
        """Same seed, same hyperparameters: only the graph differs, and it shows.

        In S4.1 the two arms were shown to differ in one forward pass. Here they
        differ after training, which is the claim the ablation of Section 6.5
        actually makes.
        """
        fold = folds[0]
        settings = tuned(config, epochs=60, patience=20)
        test = learnable_data.segment(fold.test).without_target()

        with_graph = fitted(settings, learnable_data, fold, seed=SEED).predict(test)
        without = fitted(settings, learnable_data, fold, seed=SEED, use_graph=False).predict(test)

        assert not np.allclose(with_graph, without)


class TestForecasterDeterminism:
    def test_same_seed_same_predictions(self, config, learnable_data, folds):
        fold = folds[0]
        settings = tuned(config, epochs=25, patience=25)
        test = learnable_data.segment(fold.test).without_target()

        first = fitted(settings, learnable_data, fold, dropout=0.5, seed=3).predict(test)
        again = fitted(settings, learnable_data, fold, dropout=0.5, seed=3).predict(test)
        other = fitted(settings, learnable_data, fold, dropout=0.5, seed=4).predict(test)

        np.testing.assert_array_equal(first, again)
        assert not np.allclose(first, other)

    def test_fit_does_not_depend_on_what_was_fitted_before(self, config, learnable_data, folds):
        """Seeding inside fit(), not in the constructor: a fold's model is the same
        whether it is fitted first or after another fold, so a partial re-run
        reproduces the full run's numbers.
        """
        settings = tuned(config, epochs=25, patience=25)
        test = learnable_data.segment(folds[0].test).without_target()

        alone = fitted(settings, learnable_data, folds[0], dropout=0.5, seed=1).predict(test)

        model = GCNForecaster(settings, hidden=16, dropout=0.5, seed=1)
        model.fit(learnable_data.segment(folds[1].train), learnable_data.segment(folds[1].val))
        model.fit(learnable_data.segment(folds[0].train), learnable_data.segment(folds[0].val))

        np.testing.assert_array_equal(model.predict(test), alone)


class TestGCNIgnoresTheFuture:
    """Risk R2 applied to the one model Sprint 3 could not cover."""

    def test_standardization_uses_the_train_split_only(self, config, learnable_data, folds):
        """mu and sigma come from the training rows and from nothing else.

        Fitting on train plus validation is the variant Section 5.4 calls the
        most insidious: nothing crashes, the numbers merely improve. The
        assertion is against the train block's own statistics, so widening the
        fit by even one split moves them.
        """
        fold = folds[0]
        model = fitted(tuned(config, epochs=5, patience=5), learnable_data, fold)
        train = learnable_data.segment(fold.train)

        np.testing.assert_allclose(model.standardizer_.mean_, train.features.mean(axis=0))
        np.testing.assert_allclose(model.standardizer_.scale_, train.features.std(axis=0))

    def test_target_scale_comes_from_the_train_block_only(self, config, learnable_data, folds):
        """The scalar the target is divided by is a training statistic like any other.

        It is estimated from returns rather than features, so it is the one
        quantity the standardizer does not cover -- and taking it from the whole
        panel would leak the test period's volatility into the fit, the same
        mistake in a place nobody thinks to look.
        """
        fold = folds[0]
        model = fitted(tuned(config, epochs=5, patience=5), learnable_data, fold)

        assert model.target_scale_ == pytest.approx(float(np.std(learnable_data.segment(fold.train).y)))
        assert model.target_scale_ != pytest.approx(float(np.std(learnable_data.returns)))

    def test_predictions_come_back_in_return_units(self, config, learnable_data, folds):
        """The scale applied in fit() is undone in predict().

        Without the inverse the forecasts would be off by a constant factor --
        plausible-looking, correctly shaped, and wrong in every metric of
        Section 6.4. Checked against a model that has learned the signal, so the
        predictions have the target's own magnitude to be compared with.
        """
        fold = folds[0]
        model = fitted(tuned(config, epochs=400, patience=40), learnable_data, fold)
        test = learnable_data.segment(fold.test)

        predicted = model.predict(test.without_target())

        assert np.std(predicted) == pytest.approx(np.std(test.y), rel=0.35)

    def test_forecasts_are_invariant_to_the_future(self, config):
        """Replacing the panel beyond the last prediction origin moves no forecast.

        The corruption test of S3.2, which every baseline already passes, applied
        to the GCN: returns, features and graph are all overwritten past the
        fold's last origin, and the predictions must come back bit-identical.
        """
        data, features = make_learnable_data()
        fold = make_folds(n_obs=data.n_obs, offset=WF_OFFSET, **BLOCKS)[0]
        cutoff = int(fold.test[-1])
        settings = tuned(config, epochs=25, patience=25)

        rng = np.random.default_rng(77)
        corrupted_returns = data.returns.copy()
        corrupted_features = features.copy()
        corrupted_a_hat = data.a_hat.copy()
        corrupted_returns[cutoff + 2 :] = rng.normal(5.0, 2.0, corrupted_returns[cutoff + 2 :].shape)
        corrupted_features[cutoff + 1 :] = rng.normal(-3.0, 4.0, corrupted_features[cutoff + 1 :].shape)
        corrupted_a_hat[cutoff + 1 :] = np.eye(WF_N_ASSETS)
        corrupted = replace(
            data, returns=corrupted_returns, features=corrupted_features, a_hat=corrupted_a_hat
        )

        def factory():
            return GCNForecaster(settings, hidden=16, dropout=0.5, seed=SEED)

        baseline = run_walkforward(factory, data, [fold], verbose=False)
        rerun = run_walkforward(factory, corrupted, [fold], verbose=False)

        pd.testing.assert_frame_equal(baseline.predictions, rerun.predictions)


# --------------------------------------------------------------------------
# S4.3 -- the frozen grid
# --------------------------------------------------------------------------

# The grid the training tests run: the real shape (several widths x several
# dropout rates x several seeds) at a size that fits in a second.
GRID = {"epochs": 15, "patience": 15, "hidden": [4, 8], "dropout": [0.0, 0.5], "seeds": [0, 1]}


@pytest.fixture
def grid_config(config: Config) -> Config:
    return tuned(config, **GRID)


class TestGridForecaster:
    def test_conforms_to_the_protocols(self, grid_config):
        model = GCNGridForecaster(grid_config)

        assert isinstance(model, Forecaster)
        assert isinstance(model, SupportsDiagnostics)
        assert model.name == "gcn"
        assert GCNGridForecaster(grid_config, use_graph=False).name == "gcn-nograph"

    def test_fits_every_configuration_of_the_frozen_grid(self, grid_config, learnable_data, folds):
        fold = folds[0]
        model = GCNGridForecaster(grid_config)
        model.fit(learnable_data.segment(fold.train), learnable_data.segment(fold.val))

        assert len(model.grid_) == len(GRID["hidden"]) * len(GRID["dropout"])
        assert {(row["hidden"], row["dropout"]) for row in model.grid_} == {
            (h, d) for h in GRID["hidden"] for d in GRID["dropout"]
        }
        assert all(row["n_seeds"] == len(GRID["seeds"]) for row in model.grid_)
        assert model.diagnostics()["n_fits"] == model.n_fits == 8

    def test_selects_the_lowest_validation_error(self, grid_config, learnable_data, folds):
        """The configuration kept is the argmin of the recorded validation errors.

        Selecting on the training error instead would still produce a plausible
        run -- a configuration is chosen, the grid is logged, the fold completes.
        Recomputing the argmin from the record is what tells the two apart.
        """
        fold = folds[0]
        model = GCNGridForecaster(grid_config)
        model.fit(learnable_data.segment(fold.train), learnable_data.segment(fold.val))

        best = min(model.grid_, key=lambda row: row["val_mse"])
        diagnostics = model.diagnostics()

        assert model.selected_ == best
        assert diagnostics["selected_hidden"] == best["hidden"]
        assert diagnostics["selected_dropout"] == best["dropout"]
        assert diagnostics["val_mse"] == best["val_mse"]
        assert diagnostics[f"val_mse_h{best['hidden']}_d{best['dropout']}"] == best["val_mse"]

    def test_the_recorded_error_is_measured_on_the_validation_block(self, grid_config, learnable_data, folds):
        """`val_mse` is the ensemble's error on validation, recomputed from outside.

        `test_selects_the_lowest_validation_error` only holds the argmin
        consistent with whatever was recorded, so it cannot notice a selection
        scored on the training block instead -- the ranking stays self-consistent
        and the run still looks right. This measures the number independently and
        against the split it claims to come from, which is the assertion that
        fails when the wrong block is scored.
        """
        fold = folds[0]
        train, val = learnable_data.segment(fold.train), learnable_data.segment(fold.val)
        model = GCNGridForecaster(grid_config)
        model.fit(train, val)

        ensemble = np.mean([member.predict(val.without_target()) for member in model.selected_models_], axis=0)
        on_validation = float(np.mean((val.y - ensemble) ** 2))
        on_train = float(
            np.mean(
                (
                    train.y
                    - np.mean([member.predict(train.without_target()) for member in model.selected_models_], axis=0)
                )
                ** 2
            )
        )

        assert model.selected_["val_mse"] == pytest.approx(on_validation)
        # Guard against the two coinciding by accident, which would make the
        # assertion above pass for a model scored on the wrong block.
        assert on_validation != pytest.approx(on_train)

    def test_prediction_is_the_average_over_seeds(self, grid_config, learnable_data, folds):
        """The forecast is the mean of the selected configuration's seeds, exactly.

        Returning any single seed's forecast would be shaped identically and
        differ only in the third decimal, so the check is an exact comparison
        against the mean recomputed from the individual models.
        """
        fold = folds[0]
        model = GCNGridForecaster(grid_config)
        model.fit(learnable_data.segment(fold.train), learnable_data.segment(fold.val))
        test = learnable_data.segment(fold.test).without_target()

        members = [member.predict(test) for member in model.selected_models_]

        assert len(members) == len(GRID["seeds"])
        np.testing.assert_allclose(model.predict(test), np.mean(members, axis=0))
        # The seeds genuinely disagree, so averaging them is not a no-op.
        assert not np.allclose(members[0], members[1])

    def test_runs_through_the_harness(self, grid_config, learnable_data, folds):
        for name, factory in gcn_factories(grid_config).items():
            result = run_walkforward(factory, learnable_data, folds, verbose=False)

            assert result.predictions["model"].unique().tolist() == [name]
            assert len(result.predictions) == len(folds) * BLOCKS["test"] * WF_N_ASSETS
            assert np.isfinite(result.predictions["y_pred"]).all()
            for column in ("selected_hidden", "selected_dropout", "val_mse", "n_fits", "val_mse_h4_d0.0"):
                assert column in result.diagnostics.columns

    def test_is_reproducible(self, grid_config, learnable_data, folds):
        fold = folds[0]
        train, val = learnable_data.segment(fold.train), learnable_data.segment(fold.val)
        test = learnable_data.segment(fold.test).without_target()

        def run() -> np.ndarray:
            model = GCNGridForecaster(grid_config)
            model.fit(train, val)
            return model.predict(test)

        np.testing.assert_array_equal(run(), run())

    def test_selection_ignores_what_happens_after_the_fold(self, grid_config):
        """Corruption test of S3.2, applied to the selection as well as the fit.

        Selection is the one step of Sprint 4 that could reach forward without
        any individual model doing so, since it compares configurations rather
        than fitting them. Both the chosen configuration and the forecasts must
        be unmoved by anything past the fold's last prediction origin.
        """
        data, features = make_learnable_data()
        fold = make_folds(n_obs=data.n_obs, offset=WF_OFFSET, **BLOCKS)[0]
        cutoff = int(fold.test[-1])

        rng = np.random.default_rng(91)
        corrupted_returns = data.returns.copy()
        corrupted_features = features.copy()
        corrupted_a_hat = data.a_hat.copy()
        corrupted_returns[cutoff + 2 :] = rng.normal(4.0, 3.0, corrupted_returns[cutoff + 2 :].shape)
        corrupted_features[cutoff + 1 :] = rng.normal(-2.0, 5.0, corrupted_features[cutoff + 1 :].shape)
        corrupted_a_hat[cutoff + 1 :] = np.eye(WF_N_ASSETS)
        corrupted = replace(data, returns=corrupted_returns, features=corrupted_features, a_hat=corrupted_a_hat)

        def fit_on(source):
            model = GCNGridForecaster(grid_config)
            model.fit(source.segment(fold.train), source.segment(fold.val))
            return model

        original, rerun = fit_on(data), fit_on(corrupted)
        test = data.segment(fold.test).without_target()

        assert original.selected_ == rerun.selected_
        np.testing.assert_array_equal(original.predict(test), rerun.predict(test))

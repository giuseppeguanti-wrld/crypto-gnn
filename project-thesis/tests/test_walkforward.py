"""Tests for cryptognn.evaluation.walkforward: fold geometry, views, run loop.

Two layers, in the order they were written. The first checks that the harness
does what it says: folds with the geometry the study was planned around, a
Segment that shows a model the rows it claims to show, a run loop that refuses
input it cannot score honestly. The second, in TestNoLookAhead, checks the
stronger property the whole comparison rests on -- that no information dated
after a prediction's origin can reach it -- and is largely written by corruption:
alter the future, rebuild, and demand that the past be unchanged, byte for byte.

Those tests come before any result exists, deliberately: an error there
invalidates every number of Sections 6.4 and 6.5 and would be discovered far too
late to fix. All four named in CLAUDE.md are here; the two that hold features.py
to the rule arrived with it, one step later than the two that hold the harness.

The geometry case is run against config/default.yaml rather than against invented
numbers -- the claim worth protecting is that *the shipped configuration* yields
24 folds whose first test block opens before the Terra/Luna collapse, since that
is what makes Sections 6.5 and 6.6 comparable.
"""
from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from conftest import TAU, WF_GRAPH_OFFSET, WF_LOOKBACK, WF_N_ASSETS, WF_N_FEATURES, WF_N_OBS

from cryptognn.config import load_config
from cryptognn.evaluation.walkforward import (
    Fold,
    Segment,
    WalkforwardData,
    align_graph,
    make_folds,
    make_folds_from_config,
    run_walkforward,
)
from cryptognn.features import FoldStandardizer, build_node_features
from cryptognn.graph.build import apply_threshold, mantegna_weights, normalized_adjacency
from cryptognn.graph.correlation import rolling_correlation
from cryptognn.paths import DEFAULT_CONFIG

# The study's own numbers: 2006 log returns, the walk-forward block lengths of
# config/default.yaml, and the offset implied by the 60-day correlation window.
STUDY_N_OBS = 2006
STUDY_OFFSET = 59
STUDY_BLOCKS = {"train": 365, "val": 63, "test": 63, "step": 63}

# Small layout used for the container tests: 16 folds over the 120-row fixture.
SMALL_BLOCKS = {"train": 20, "val": 5, "test": 5, "step": 5}

# Correlation window of the graph-construction test. Shorter than the study's 60
# only so the 400-row fixture yields plenty of windows.
GRAPH_WINDOW = 30


def small_folds() -> list[Fold]:
    return make_folds(n_obs=WF_N_OBS, offset=WF_GRAPH_OFFSET, **SMALL_BLOCKS)


def build_graph_chain(returns: pd.DataFrame, window: int = GRAPH_WINDOW) -> np.ndarray:
    """The study's own graph pipeline, end to end, aligned on the return panel.

    Deliberately calls the production functions rather than a simplified stand-in:
    a look-ahead test that re-implements the thing it audits proves only that the
    re-implementation is clean.
    """
    corr, corr_index = rolling_correlation(returns, window)
    corr = corr.astype(np.float64)
    thresholded = apply_threshold(corr, mantegna_weights(corr), TAU)
    return align_graph(normalized_adjacency(thresholded), corr_index, returns.index)


class ConstantForecaster:
    """A forecaster with no content: it exists so the harness can be tested
    without a model, and records what it was handed so the loop's behaviour --
    a fresh instance per fold, validation always passed -- is observable.
    """

    name = "constant"

    def __init__(self, value: float = 0.25) -> None:
        self.value = value
        self.calls: list[tuple[int, int]] = []

    def fit(self, train, val) -> None:
        self.calls.append((len(train), len(val)))

    def predict(self, segment) -> np.ndarray:
        return np.full((len(segment), segment.n_assets), self.value)


class DiagnosticForecaster(ConstantForecaster):
    name = "diagnostic"

    def diagnostics(self) -> dict[str, float | int | str]:
        return {"lag_order": 3, "n_params": 42}


class LastLagForecaster(ConstantForecaster):
    """Forecasts from the inputs a legitimate model uses, and from nothing else.

    Reads the most recent lag and the first feature channel -- both dated at the
    prediction origin -- so that if the harness ever handed a model a row it
    should not have, this model's output would move.
    """

    name = "last-lag"

    def predict(self, segment) -> np.ndarray:
        recent = segment.lags[:, -1, :]
        return recent if segment.features is None else 0.5 * (recent + segment.features[:, :, 0])


class TargetPeekingForecaster(ConstantForecaster):
    """Returns the target it was asked to predict -- the leak in its purest form."""

    name = "peeking"

    def predict(self, segment) -> np.ndarray:
        return segment.y


class TestFold:
    def test_rejects_empty_block(self):
        with pytest.raises(ValueError, match="empty val"):
            Fold(index=0, train=np.arange(0, 10), val=np.array([], dtype=int), test=np.arange(10, 15))

    def test_rejects_non_contiguous_block(self):
        with pytest.raises(ValueError, match="not a contiguous range"):
            Fold(index=0, train=np.array([0, 1, 3]), val=np.arange(4, 6), test=np.arange(6, 8))

    def test_rejects_blocks_out_of_order(self):
        """The ordering test of S3.2 checks a property the constructor enforces:
        a fold with test before train cannot be built in the first place.
        """
        with pytest.raises(ValueError, match="out of order"):
            Fold(index=0, train=np.arange(10, 20), val=np.arange(20, 25), test=np.arange(5, 10))


class TestMakeFolds:
    def test_study_geometry(self):
        """The layout the whole of Sprints 3-5 is planned around."""
        folds = make_folds(n_obs=STUDY_N_OBS, offset=STUDY_OFFSET, **STUDY_BLOCKS)

        assert len(folds) == 24
        assert folds[0].train[0] == STUDY_OFFSET
        assert folds[0].test[0] == 487
        assert folds[-1].test[-1] == 1998
        assert [fold.index for fold in folds] == list(range(24))

    def test_blocks_are_contiguous_and_ordered(self):
        for fold in make_folds(n_obs=STUDY_N_OBS, offset=STUDY_OFFSET, **STUDY_BLOCKS):
            assert fold.sizes == (365, 63, 63)
            assert fold.train[-1] + 1 == fold.val[0]
            assert fold.val[-1] + 1 == fold.test[0]

    def test_folds_advance_by_step(self):
        folds = make_folds(n_obs=STUDY_N_OBS, offset=STUDY_OFFSET, **STUDY_BLOCKS)
        starts = np.array([fold.train[0] for fold in folds])

        assert np.all(np.diff(starts) == STUDY_BLOCKS["step"])
        # Consecutive test blocks tile the period without gap or overlap, so
        # every out-of-sample day is predicted exactly once.
        test_blocks = np.concatenate([fold.test for fold in folds])
        assert np.array_equal(test_blocks, np.arange(folds[0].test[0], folds[-1].test[-1] + 1))

    def test_no_position_before_offset_or_past_horizon(self):
        folds = make_folds(n_obs=STUDY_N_OBS, offset=STUDY_OFFSET, **STUDY_BLOCKS)

        assert min(fold.train[0] for fold in folds) == STUDY_OFFSET
        # The last origin still needs its target one row further on.
        assert max(fold.test[-1] for fold in folds) <= STUDY_N_OBS - 2

    def test_horizon_reserves_rows(self):
        """A longer horizon consumes usable positions and must cost folds."""
        one_day = make_folds(n_obs=WF_N_OBS, offset=WF_GRAPH_OFFSET, horizon=1, **SMALL_BLOCKS)
        ten_days = make_folds(n_obs=WF_N_OBS, offset=WF_GRAPH_OFFSET, horizon=10, **SMALL_BLOCKS)

        assert len(one_day) == 16
        assert len(ten_days) < len(one_day)
        assert ten_days[-1].test[-1] + 10 <= WF_N_OBS - 1

    def test_expanding_mode_grows_the_train_block(self):
        rolling = make_folds(n_obs=STUDY_N_OBS, offset=STUDY_OFFSET, mode="rolling", **STUDY_BLOCKS)
        expanding = make_folds(n_obs=STUDY_N_OBS, offset=STUDY_OFFSET, mode="expanding", **STUDY_BLOCKS)

        assert len(expanding) == len(rolling)
        train_lengths = [len(fold.train) for fold in expanding]
        assert train_lengths == sorted(train_lengths) and train_lengths[0] < train_lengths[-1]
        assert all(fold.train[0] == STUDY_OFFSET for fold in expanding)
        # Only the train block differs: the models are still scored on the same days.
        for rolling_fold, expanding_fold in zip(rolling, expanding, strict=True):
            assert np.array_equal(rolling_fold.test, expanding_fold.test)
            assert np.array_equal(rolling_fold.val, expanding_fold.val)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"mode": "sliding"}, "Unknown walk-forward mode"),
            ({"train": 0}, "train must be positive"),
            ({"step": -1}, "step must be positive"),
            ({"offset": -5}, "offset must be non-negative"),
            ({"n_obs": 100}, "No fold fits"),
        ],
    )
    def test_rejects_impossible_parameters(self, kwargs, match):
        arguments = {"n_obs": STUDY_N_OBS, "offset": STUDY_OFFSET, **STUDY_BLOCKS, **kwargs}
        with pytest.raises(ValueError, match=match):
            make_folds(**arguments)


class TestMakeFoldsFromConfig:
    def test_shipped_config_yields_the_planned_layout(self):
        """The offset is derived from graph.window, never written down twice."""
        config = load_config(DEFAULT_CONFIG)
        folds = make_folds_from_config(config, STUDY_N_OBS)

        assert len(folds) == 24
        assert folds[0].train[0] == config.graph.window - 1
        assert folds[0].sizes == (config.walkforward.train, config.walkforward.val, config.walkforward.test)


class TestAlignGraph:
    def test_places_each_graph_on_its_closing_date(self):
        dates = pd.date_range("2021-01-01", periods=20, freq="D", tz="UTC")
        corr_index = dates[5:]
        a_hat = np.arange(15 * 4).reshape(15, 2, 2).astype(float)

        aligned = align_graph(a_hat, corr_index, dates)

        assert aligned.shape == (20, 2, 2)
        assert np.isnan(aligned[:5]).all()
        np.testing.assert_array_equal(aligned[5:], a_hat)

    def test_matches_a_naive_index_against_a_tz_aware_panel(self):
        """The case the pipeline actually presents: returns.parquet keeps its
        UTC awareness, corr_index.npy loses it in the round trip through .npy.
        """
        dates = pd.date_range("2021-01-01", periods=20, freq="D", tz="UTC")
        naive_index = dates[5:].tz_localize(None)
        a_hat = np.arange(15 * 4).reshape(15, 2, 2).astype(float)

        aligned = align_graph(a_hat, naive_index, dates)

        assert np.isnan(aligned[:5]).all()
        np.testing.assert_array_equal(aligned[5:], a_hat)

    def test_rejects_length_mismatch(self):
        dates = pd.date_range("2021-01-01", periods=20, freq="D", tz="UTC")
        with pytest.raises(ValueError, match="matrices but corr_index"):
            align_graph(np.zeros((10, 2, 2)), dates[5:], dates)

    def test_rejects_dates_absent_from_the_panel(self):
        dates = pd.date_range("2021-01-01", periods=20, freq="D", tz="UTC")
        corr_index = pd.date_range("2022-01-01", periods=3, freq="D", tz="UTC")
        with pytest.raises(ValueError, match="absent from the return panel"):
            align_graph(np.zeros((3, 2, 2)), corr_index, dates)


class TestWalkforwardData:
    def test_rejects_misaligned_arrays(self, synthetic_walkforward_data):
        data = synthetic_walkforward_data
        with pytest.raises(ValueError, match="dates for"):
            WalkforwardData(dates=data.dates[:-1], assets=data.assets, returns=data.returns)
        with pytest.raises(ValueError, match="is not"):
            WalkforwardData(
                dates=data.dates,
                assets=data.assets,
                returns=data.returns,
                a_hat=np.zeros((WF_N_OBS, WF_N_ASSETS, WF_N_ASSETS + 1)),
            )


class TestSegment:
    def test_target_is_the_next_observation(self, synthetic_walkforward_data):
        positions = np.arange(30, 40)
        segment = synthetic_walkforward_data.segment(positions)

        np.testing.assert_array_equal(segment.y, synthetic_walkforward_data.returns[positions + 1])
        np.testing.assert_array_equal(segment.returns, synthetic_walkforward_data.returns[positions])
        assert segment.y[0, 0] == 31.0  # returns[t, j] == t + j/10, so position 30 targets 31
        assert segment.target_dates[0] == synthetic_walkforward_data.dates[31]
        assert segment.dates[0] == synthetic_walkforward_data.dates[30]

    def test_lags_end_at_the_origin_and_run_chronologically(self, synthetic_walkforward_data):
        positions = np.arange(30, 40)
        segment = synthetic_walkforward_data.segment(positions)

        assert segment.lags.shape == (len(positions), WF_LOOKBACK, WF_N_ASSETS)
        np.testing.assert_array_equal(segment.lags[:, -1, :], segment.returns)
        # Oldest first: position 30 with lookback 3 sees returns 28, 29, 30.
        np.testing.assert_array_equal(segment.lags[0, :, 0], np.array([28.0, 29.0, 30.0]))
        assert np.isfinite(segment.lags).all()

    def test_features_and_graph_are_sliced_on_the_same_axis(self, synthetic_walkforward_data):
        positions = np.arange(30, 40)
        segment = synthetic_walkforward_data.segment(positions)

        assert segment.features.shape == (len(positions), WF_N_ASSETS, WF_N_FEATURES)
        assert segment.a_hat.shape == (len(positions), WF_N_ASSETS, WF_N_ASSETS)
        np.testing.assert_array_equal(segment.features[:, :, 0], segment.returns)
        np.testing.assert_array_equal(segment.features[:, :, 2], segment.returns + 200.0)

    def test_optional_arrays_stay_absent(self, synthetic_walkforward_data):
        bare = WalkforwardData(
            dates=synthetic_walkforward_data.dates,
            assets=synthetic_walkforward_data.assets,
            returns=synthetic_walkforward_data.returns,
        )
        segment = bare.segment(np.arange(10, 20))

        assert segment.features is None and segment.a_hat is None
        assert segment.lags.shape == (10, 1, WF_N_ASSETS)  # default lookback

    def test_rejects_positions_without_a_target(self, synthetic_walkforward_data):
        with pytest.raises(ValueError, match="leave the panel"):
            synthetic_walkforward_data.segment(np.array([WF_N_OBS - 1]))

    def test_rejects_empty_positions(self, synthetic_walkforward_data):
        with pytest.raises(ValueError, match="empty position array"):
            synthetic_walkforward_data.segment(np.array([], dtype=int))


class TestRunWalkforward:
    def test_predictions_are_long_and_cover_every_test_day(self, synthetic_walkforward_data):
        folds = small_folds()
        result = run_walkforward(ConstantForecaster, synthetic_walkforward_data, folds, verbose=False)
        predictions = result.predictions

        assert list(predictions.columns) == ["fold", "date", "asset", "y_true", "y_pred", "model"]
        assert len(predictions) == len(folds) * SMALL_BLOCKS["test"] * WF_N_ASSETS
        assert predictions["model"].unique().tolist() == ["constant"]
        assert (predictions["y_pred"] == 0.25).all()
        assert predictions["fold"].nunique() == len(folds)

    def test_rows_carry_the_target_date_and_its_realized_return(self, synthetic_walkforward_data):
        folds = small_folds()
        result = run_walkforward(ConstantForecaster, synthetic_walkforward_data, folds, verbose=False)
        first = result.predictions.iloc[0]
        origin = folds[0].test[0]

        assert first["date"] == synthetic_walkforward_data.dates[origin + 1]
        assert first["asset"] == "A0"
        assert first["y_true"] == synthetic_walkforward_data.returns[origin + 1, 0]
        # Assets tile within a date rather than across dates.
        assert result.predictions["asset"].iloc[:WF_N_ASSETS].tolist() == list(synthetic_walkforward_data.assets)

    def test_one_fresh_model_per_fold_and_validation_always_passed(self, synthetic_walkforward_data):
        folds = small_folds()
        built: list[ConstantForecaster] = []

        def factory() -> ConstantForecaster:
            model = ConstantForecaster()
            built.append(model)
            return model

        run_walkforward(factory, synthetic_walkforward_data, folds, verbose=False)

        assert len(built) == len(folds)
        # Each instance was fitted exactly once, and saw a non-empty validation split.
        assert all(model.calls == [(SMALL_BLOCKS["train"], SMALL_BLOCKS["val"])] for model in built)

    def test_diagnostics_row_per_fold_with_optional_hook(self, synthetic_walkforward_data):
        folds = small_folds()
        plain = run_walkforward(ConstantForecaster, synthetic_walkforward_data, folds, verbose=False)
        hooked = run_walkforward(DiagnosticForecaster, synthetic_walkforward_data, folds, verbose=False)

        assert len(plain.diagnostics) == len(folds)
        assert "lag_order" not in plain.diagnostics.columns
        assert plain.diagnostics["n_train"].eq(SMALL_BLOCKS["train"]).all()
        assert (plain.diagnostics["fit_seconds"] >= 0).all()
        assert hooked.diagnostics["lag_order"].eq(3).all()
        assert hooked.diagnostics["n_params"].eq(42).all()

    def test_name_override_labels_the_run(self, synthetic_walkforward_data):
        result = run_walkforward(
            ConstantForecaster, synthetic_walkforward_data, small_folds(), name="zero", verbose=False
        )

        assert result.predictions["model"].unique().tolist() == ["zero"]
        assert result.diagnostics["model"].unique().tolist() == ["zero"]

    def test_rejects_a_fold_starting_before_the_first_graph(self, synthetic_walkforward_data):
        early = Fold(index=0, train=np.arange(0, 20), val=np.arange(20, 25), test=np.arange(25, 30))

        with pytest.raises(ValueError, match="before the first available graph"):
            run_walkforward(ConstantForecaster, synthetic_walkforward_data, [early], verbose=False)

    def test_rejects_non_finite_inputs(self, synthetic_walkforward_data):
        """An understated graph_offset must surface as a refusal to train, not
        as a model quietly fitted on the NaN padding of align_graph().
        """
        data = synthetic_walkforward_data
        unguarded = WalkforwardData(
            dates=data.dates,
            assets=data.assets,
            returns=data.returns,
            graph_offset=0,
            lookback=WF_LOOKBACK,
            features=data.features,
            a_hat=data.a_hat,
        )
        fold = Fold(index=0, train=np.arange(0, 20), val=np.arange(20, 25), test=np.arange(25, 30))

        with pytest.raises(ValueError, match="non-finite values in a_hat"):
            run_walkforward(ConstantForecaster, unguarded, [fold], verbose=False)

    def test_rejects_malformed_predictions(self, synthetic_walkforward_data):
        class MisshapenForecaster(ConstantForecaster):
            def predict(self, segment):
                return np.zeros((len(segment), segment.n_assets + 1))

        class NonFiniteForecaster(ConstantForecaster):
            def predict(self, segment):
                return np.full((len(segment), segment.n_assets), np.nan)

        folds = small_folds()[:1]
        with pytest.raises(ValueError, match="expected"):
            run_walkforward(MisshapenForecaster, synthetic_walkforward_data, folds, verbose=False)
        with pytest.raises(ValueError, match="non-finite predictions"):
            run_walkforward(NonFiniteForecaster, synthetic_walkforward_data, folds, verbose=False)


class TestNoLookAhead:
    """The property the entire predictive comparison rests on.

    Not "the code does what it says" -- that is the concern of the classes above
    -- but "no information dated after t can reach a forecast of r_{t+1}". Where
    possible it is checked by corruption: rebuild the inputs from a panel whose
    future has been replaced, and require what precedes t to be bit-identical.
    A test of that shape cannot pass by accident, and cannot be satisfied by a
    pipeline that merely looks careful.
    """

    def test_fold_ordering(self):
        """Within a fold, every training and validation day precedes every test day."""
        folds = make_folds_from_config(load_config(DEFAULT_CONFIG), STUDY_N_OBS)

        for fold in folds:
            assert fold.train.max() < fold.val.min()
            assert fold.val.max() < fold.test.min()
            assert max(fold.train.max(), fold.val.max()) < fold.test.min()
            train, val, test = set(fold.train), set(fold.val), set(fold.test)
            assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)

        for earlier, later in itertools.pairwise(folds):
            # No test day is ever scored twice, and the days a fold was tested on
            # do become training material for the next one -- that is walk-forward
            # working as intended, not a leak. With step == val == test they land
            # exactly on the following fold's validation block.
            assert earlier.test.max() < later.test.min()
            assert np.array_equal(earlier.test, later.val)
            assert set(earlier.val).issubset(set(later.train))

    def test_graph_precedes_target(self, correlated_returns):
        """The graph used to forecast r_{t+1} is built on [t-w+1, t] and never beyond.

        Three claims, the last of which is the one that matters: the correlation
        index dates each window by its close; a segment's adjacency at a position
        equals the graph rebuilt from that window alone; and replacing the panel's
        future leaves every graph up to t untouched.
        """
        window = GRAPH_WINDOW
        dates = correlated_returns.index
        values = correlated_returns.to_numpy()
        _, corr_index = rolling_correlation(correlated_returns, window)
        aligned = build_graph_chain(correlated_returns, window)

        assert corr_index.equals(dates[window - 1 :])
        assert np.isnan(aligned[: window - 1]).all()
        assert np.isfinite(aligned[window - 1 :]).all()

        # Rebuilt from that window alone: no rolling machinery, no shared state.
        for position in (window - 1, 150, len(dates) - 2):
            corr = np.corrcoef(values[position - window + 1 : position + 1].T)
            expected = normalized_adjacency(apply_threshold(corr, mantegna_weights(corr), TAU))
            np.testing.assert_allclose(aligned[position], expected, atol=1e-6)

        cutoff = 250
        rng = np.random.default_rng(999)
        corrupted = correlated_returns.copy()
        # A different distribution, not a rescaling: multiplying the future by a
        # constant would leave every correlation unchanged and the test vacuous.
        corrupted.iloc[cutoff + 1 :] = rng.standard_normal(corrupted.iloc[cutoff + 1 :].shape)
        rebuilt = build_graph_chain(corrupted, window)

        np.testing.assert_array_equal(rebuilt[: cutoff + 1], aligned[: cutoff + 1])
        assert not np.allclose(rebuilt[cutoff + 1], aligned[cutoff + 1])

    def test_segment_inputs_ignore_the_future(self, synthetic_walkforward_data):
        """What a model receives at position t is invariant to the panel's future.

        The half of the feature-leak check the harness can make on its own: the
        node features of S3.3 reach a model through exactly these arrays, so a
        feature dated after t would have to survive this slicing to do harm.
        """
        data = synthetic_walkforward_data
        cutoff = 80
        positions = np.arange(60, cutoff + 1)

        # Every input array is corrupted, not just the returns: a segment that
        # read features or graphs one row ahead would otherwise pick up nothing,
        # since both containers would carry the same untouched arrays.
        corrupted = replace(
            data,
            returns=data.returns.copy(),
            features=data.features.copy(),
            a_hat=data.a_hat.copy(),
        )
        corrupted.returns[cutoff + 1 :] = -999.0
        corrupted.features[cutoff + 1 :] = -999.0
        corrupted.a_hat[cutoff + 1 :] = -999.0

        original = data.segment(positions)
        rebuilt = corrupted.segment(positions)

        np.testing.assert_array_equal(rebuilt.returns, original.returns)
        np.testing.assert_array_equal(rebuilt.lags, original.lags)
        np.testing.assert_array_equal(rebuilt.features, original.features)
        np.testing.assert_array_equal(rebuilt.a_hat, original.a_hat)
        # The corruption is real and lands where it should: on the target of the
        # last origin, whose realized return is the first corrupted row.
        assert (rebuilt.y[-1] == -999.0).all()
        np.testing.assert_array_equal(rebuilt.y[:-1], original.y[:-1])

    def test_predict_cannot_read_the_target(self, synthetic_walkforward_data):
        """The target is withheld at prediction time, and reaching for it fails loudly."""
        segment = synthetic_walkforward_data.segment(np.arange(30, 40))
        masked = segment.without_target()

        assert np.isnan(masked.y).all()
        assert np.isfinite(segment.y).all()  # the original is left intact for scoring
        np.testing.assert_array_equal(masked.lags, segment.lags)
        assert isinstance(masked, Segment)

        with pytest.raises(ValueError, match="non-finite predictions"):
            run_walkforward(TargetPeekingForecaster, synthetic_walkforward_data, small_folds()[:1], verbose=False)

    def test_standardizer_train_only(self, correlated_returns, synthetic_volumes):
        """mu and sigma come from the training rows and from nowhere else.

        The insidious variant of look-ahead named in Section 5.4: standardizing
        on the full sample leaks the test period's scale into training, and
        nothing about the run looks wrong afterwards -- the numbers merely
        improve. Checked by corruption, and then by the converse, so that the
        first half cannot pass for a reason as dull as the corruption not landing.
        """
        config = load_config(DEFAULT_CONFIG)
        features = build_node_features(correlated_returns, synthetic_volumes, config)
        train, later = slice(30, 200), slice(200, 260)

        fitted = FoldStandardizer().fit(features[train])
        standardized = fitted.transform(features[train])

        rng = np.random.default_rng(11)
        corrupted = features.copy()
        corrupted[later] = rng.normal(50.0, 20.0, size=corrupted[later].shape)

        refitted = FoldStandardizer().fit(corrupted[train])
        np.testing.assert_array_equal(refitted.mean_, fitted.mean_)
        np.testing.assert_array_equal(refitted.scale_, fitted.scale_)
        np.testing.assert_array_equal(refitted.transform(corrupted[train]), standardized)

        # The later block is rescaled with the train statistics, not re-centred on
        # its own: a transform() that recomputed mu and sigma per block would leak
        # the test period's scale into the test period's inputs, and would show up
        # here as a block sitting exactly on zero.
        transformed_later = fitted.transform(features[later])
        np.testing.assert_allclose(transformed_later, (features[later] - fitted.mean_) / fitted.scale_)
        assert not np.allclose(transformed_later.mean(axis=0), 0.0, atol=1e-6)

        # The converse: the corrupted rows do move the statistics of a
        # standardizer that is allowed to see them. Without this the test above
        # would also pass on a standardizer that ignored its input entirely.
        leaking = FoldStandardizer().fit(corrupted[30:260])
        assert not np.allclose(leaking.mean_, fitted.mean_)

    def test_no_target_leak_in_features(self, correlated_returns, synthetic_volumes):
        """No feature at t contains r_{t+1}: altering the targets leaves X alone.

        The corruption test PLANNING asks for, run on the real feature tensor
        rather than on the container arrays: everything from t+1 onward is
        replaced -- returns and volumes both -- and every feature row up to t
        must come back bit-identical.
        """
        config = load_config(DEFAULT_CONFIG)
        cutoff = 250
        baseline = build_node_features(correlated_returns, synthetic_volumes, config)

        rng = np.random.default_rng(13)
        future = slice(cutoff + 1, None)
        returns = correlated_returns.copy()
        volumes = synthetic_volumes.copy()
        returns.iloc[future] = rng.standard_normal(returns.iloc[future].shape)
        volumes.iloc[future] *= 1000.0

        rebuilt = build_node_features(returns, volumes, config)

        np.testing.assert_array_equal(rebuilt[: cutoff + 1], baseline[: cutoff + 1])
        assert not np.allclose(rebuilt[cutoff + 1], baseline[cutoff + 1])

    def test_predictions_invariant_to_the_future(self, synthetic_walkforward_data):
        """End to end: a fold's forecasts do not move when the panel's future does.

        Covers the whole loop -- fold slicing, segment construction, the model
        call -- rather than any one function, which is where an off-by-one in the
        slicing would hide from the checks above.
        """
        data = synthetic_walkforward_data
        fold = small_folds()[0]
        cutoff = int(fold.test[-1])

        corrupted = replace(
            data,
            returns=data.returns.copy(),
            features=data.features.copy(),
            a_hat=data.a_hat.copy(),
        )
        corrupted.returns[cutoff + 2 :] = -999.0  # cutoff + 1 is the last target, still scored
        corrupted.features[cutoff + 1 :] = -999.0
        corrupted.a_hat[cutoff + 1 :] = -999.0

        baseline = run_walkforward(LastLagForecaster, data, [fold], verbose=False)
        rerun = run_walkforward(LastLagForecaster, corrupted, [fold], verbose=False)

        pd.testing.assert_frame_equal(baseline.predictions, rerun.predictions)
        assert baseline.predictions["y_pred"].nunique() > 1  # the model is not a constant

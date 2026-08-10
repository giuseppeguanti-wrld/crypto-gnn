"""Tests for cryptognn.features: causal node features and per-fold standardization.

Every channel is checked against the obvious direct computation over the window
it claims to use -- a loop or a slice, not a second vectorized implementation.
The vectorization exists for speed and style; any disagreement with the naive
version is a bug in it, never a different definition.

The look-ahead properties of this module (features invariant to the future, mu
and sigma fitted on train alone) are held to in TestNoLookAhead of
test_walkforward.py, where the study's other anti-leak tests live.

build_study_data() has no test here: it reads data/processed/, and the suite
must run on a fresh clone before any script has been executed. It is exercised
end to end from the command line instead.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from cryptognn.config import load_config
from cryptognn.features import FoldStandardizer, build_node_features, feature_names
from cryptognn.paths import DEFAULT_CONFIG

# Longest window of the shipped config: max(vol_windows) = 20, so 19 warm-up rows.
WARMUP = 19


@pytest.fixture
def config():
    return load_config(DEFAULT_CONFIG)


@pytest.fixture
def config_without_volume(config):
    return dataclasses.replace(
        config, features=dataclasses.replace(config.features, use_volume=False)
    )


class TestBuildNodeFeatures:
    def test_shape_and_channel_names(self, correlated_returns, synthetic_volumes, config):
        features = build_node_features(correlated_returns, synthetic_volumes, config)

        assert features.shape == (*correlated_returns.shape, 8)
        assert feature_names(config) == [
            "r_lag0",
            "r_lag1",
            "r_lag2",
            "r_lag3",
            "r_lag4",
            "rv_5",
            "rv_20",
            "logvol_z_20",
        ]

    def test_lagged_return_channels_are_causal(self, correlated_returns, synthetic_volumes, config):
        """Channel k at row t is r_{t-k}; channel 0 is r_t itself, known at the close."""
        features = build_node_features(correlated_returns, synthetic_volumes, config)
        values = correlated_returns.to_numpy()

        for position in (WARMUP, 100, len(values) - 1):
            for lag in range(config.features.lags):
                np.testing.assert_allclose(features[position, :, lag], values[position - lag])

    def test_realized_volatility_matches_the_direct_computation(
        self, correlated_returns, synthetic_volumes, config
    ):
        features = build_node_features(correlated_returns, synthetic_volumes, config)
        values = correlated_returns.to_numpy()
        offset = config.features.lags

        for channel, window in enumerate(config.features.vol_windows):
            for position in (100, 250):
                expected = np.sqrt(np.mean(values[position - window + 1 : position + 1] ** 2, axis=0))
                np.testing.assert_allclose(features[position, :, offset + channel], expected)

    def test_volume_zscore_matches_the_direct_computation(
        self, correlated_returns, synthetic_volumes, config
    ):
        features = build_node_features(correlated_returns, synthetic_volumes, config)
        log_volume = np.log(synthetic_volumes.to_numpy())
        window = max(config.features.vol_windows)

        for position in (100, 250):
            history = log_volume[position - window + 1 : position + 1]
            expected = (log_volume[position] - history.mean(axis=0)) / history.std(axis=0)
            np.testing.assert_allclose(features[position, :, -1], expected)

    def test_volume_zscore_is_blind_to_the_unit_of_account(
        self, correlated_returns, synthetic_volumes, config
    ):
        """Rescaling an asset's volume must not move its channel: taking logs
        turns a unit change into an additive constant, which the z-score removes.
        This is what makes base volume usable across assets that trade in
        billions of coins and assets that trade in thousands.
        """
        baseline = build_node_features(correlated_returns, synthetic_volumes, config)
        rescaled = synthetic_volumes.copy()
        rescaled.iloc[:, 0] *= 1e6

        features = build_node_features(correlated_returns, rescaled, config)

        np.testing.assert_allclose(features[WARMUP:], baseline[WARMUP:], atol=1e-10)

    def test_warm_up_rows_are_nan_and_nothing_else_is(
        self, correlated_returns, synthetic_volumes, config
    ):
        """The incomplete history stays missing rather than being back-filled: a
        value invented there could only come from the future.
        """
        features = build_node_features(correlated_returns, synthetic_volumes, config)

        assert np.isnan(features[:WARMUP]).any(axis=(1, 2)).all()
        assert np.isfinite(features[WARMUP:]).all()

    def test_volume_channel_is_optional(self, correlated_returns, config_without_volume):
        features = build_node_features(correlated_returns, None, config_without_volume)

        assert features.shape == (*correlated_returns.shape, 7)
        assert feature_names(config_without_volume) == [
            "r_lag0", "r_lag1", "r_lag2", "r_lag3", "r_lag4", "rv_5", "rv_20",
        ]

    @staticmethod
    def _with_a_negative_bar(volumes):
        corrupted = volumes.copy()
        corrupted.iloc[3, 0] = -1.0
        return corrupted

    @pytest.mark.parametrize(
        ("mutate", "match"),
        [
            (lambda volumes: None, "no volume panel was given"),
            (lambda volumes: volumes.iloc[:-5], "misses 5 return dates"),
            (lambda volumes: TestBuildNodeFeatures._with_a_negative_bar(volumes), "non-positive"),
        ],
    )
    def test_rejects_an_unusable_volume_panel(
        self, correlated_returns, synthetic_volumes, config, mutate, match
    ):
        with pytest.raises(ValueError, match=match):
            build_node_features(correlated_returns, mutate(synthetic_volumes), config)


class TestFoldStandardizer:
    @pytest.fixture
    def train_block(self, correlated_returns, synthetic_volumes, config):
        return build_node_features(correlated_returns, synthetic_volumes, config)[WARMUP:200]

    def test_standardizes_each_asset_and_channel_separately(self, train_block):
        standardizer = FoldStandardizer()
        transformed = standardizer.fit_transform(train_block)

        assert standardizer.mean_.shape == train_block.shape[1:]
        np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-10)
        np.testing.assert_allclose(transformed.std(axis=0), 1.0, atol=1e-10)

    def test_constant_channel_standardizes_to_zero_instead_of_amplified_noise(self, train_block):
        """The std of a constant array is ~1e-16, not 0. Dividing that residue by
        itself would produce an O(1) value out of nothing -- silently, since it
        is neither NaN nor obviously wrong.
        """
        block = train_block.copy()
        block[:, 0, 3] = 0.7

        transformed = FoldStandardizer().fit_transform(block)

        assert np.isfinite(transformed).all()
        np.testing.assert_allclose(transformed[:, 0, 3], 0.0, atol=1e-12)

    def test_transform_before_fit_raises(self, train_block):
        with pytest.raises(RuntimeError, match="before fit"):
            FoldStandardizer().transform(train_block)

    def test_rejects_a_block_of_the_wrong_shape(self, train_block):
        standardizer = FoldStandardizer().fit(train_block)
        with pytest.raises(ValueError, match="does not match the fitted"):
            standardizer.transform(train_block[:, :-1])

    def test_rejects_a_train_block_reaching_into_the_warm_up(
        self, correlated_returns, synthetic_volumes, config
    ):
        features = build_node_features(correlated_returns, synthetic_volumes, config)
        with pytest.raises(ValueError, match="Non-finite values in the training block"):
            FoldStandardizer().fit(features[:100])

    def test_transform_segment_touches_features_only(
        self, synthetic_walkforward_data, train_block
    ):
        segment = synthetic_walkforward_data.segment(np.arange(30, 40))
        standardizer = FoldStandardizer().fit(segment.features)

        standardized = standardizer.transform_segment(segment)

        np.testing.assert_allclose(standardized.features, standardizer.transform(segment.features))
        np.testing.assert_array_equal(standardized.returns, segment.returns)
        np.testing.assert_array_equal(standardized.lags, segment.lags)
        np.testing.assert_array_equal(standardized.y, segment.y)
        np.testing.assert_array_equal(standardized.a_hat, segment.a_hat)

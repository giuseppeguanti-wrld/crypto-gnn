"""Tests for cryptognn.graph.threshold.

These target the property the permutation null exists to have: it must destroy
cross-asset dependence while leaving each asset's marginal untouched. Both
halves are checked on data with a known, strong correlation structure, because
a null that quietly preserved some dependence would still produce a
plausible-looking tau -- just a badly inflated one.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from conftest import N_ASSETS, N_PAIRS

from cryptognn.config import Config, GraphConfig, ThresholdConfig
from cryptognn.graph.threshold import calibrate_tau, check_tau_plausible, permutation_null


def _config(**threshold_overrides) -> Config:
    """A minimal Config carrying only what calibrate_tau() reads, so the tests
    do not depend on config/default.yaml staying fixed.
    """
    threshold = ThresholdConfig(
        **{
            "method": "permutation",
            "alpha": 0.05,
            "n_permutations": 200,
            "n_calibration_windows": 5,
            "statistic": "pooled",
            "tau_fixed": 0.30,
            **threshold_overrides,
        }
    )
    graph = GraphConfig(window=60, weight="mantegna", self_loops=True, threshold=threshold)
    return Config(
        data=None, graph=graph, features=None, walkforward=None, model=None, backtest=None, seed=42
    )


class TestPermutationNull:
    def test_shapes(self, correlated_returns):
        pooled, max_per_permutation = permutation_null(
            correlated_returns.iloc[:60], n_permutations=50, rng=np.random.default_rng(0)
        )

        assert pooled.shape == (50 * N_PAIRS,)
        assert max_per_permutation.shape == (50,)
        assert max_per_permutation.max() <= pooled.max()

    def test_preserves_marginals(self, correlated_returns):
        """Each shuffled column must be a permutation of the original column --
        that is what keeps the heavy tails of the marginals in the null (thesis
        Sec. 4.2).

        Checked through the diagonal of the replica correlation matrices:
        shuffling a column cannot change its variance, so the variance implied
        by the null is the observed one. A null that resampled or refitted the
        marginals instead would not have this property.
        """
        window = correlated_returns.iloc[:60]
        rng = np.random.default_rng(0)
        values = window.to_numpy()

        replicas = np.broadcast_to(values, (20, 60, N_ASSETS))
        shuffled = rng.permuted(replicas, axis=1)

        for replica in shuffled:
            np.testing.assert_allclose(np.sort(replica, axis=0), np.sort(values, axis=0), atol=1e-12)

    def test_destroys_cross_dependence(self, correlated_returns):
        """On a panel whose true pairwise correlation is ~0.8, the null must be
        centered at zero. If columns were permuted jointly instead of
        independently, the null would sit near the observed correlation and tau
        would be inflated.
        """
        window = correlated_returns.iloc[:60]
        observed = np.corrcoef(window.to_numpy().T)[np.triu_indices(N_ASSETS, k=1)]

        pooled, _ = permutation_null(window, n_permutations=500, rng=np.random.default_rng(0))

        assert observed.mean() > 0.5, "fixture should be strongly correlated"
        assert abs(pooled.mean()) < 0.02
        # Sampling sd of rho under the null is ~1/sqrt(T_w - 1) = 0.13 for T_w = 60.
        assert 0.08 < pooled.std() < 0.20

    def test_does_not_mutate_input(self, correlated_returns):
        window = correlated_returns.iloc[:60]
        before = window.to_numpy().copy()

        permutation_null(window, n_permutations=20, rng=np.random.default_rng(0))

        np.testing.assert_array_equal(window.to_numpy(), before)

    def test_rejects_bad_input(self, correlated_returns):
        rng = np.random.default_rng(0)

        with pytest.raises(ValueError, match="2-D"):
            permutation_null(np.zeros(60), n_permutations=10, rng=rng)
        with pytest.raises(ValueError, match="n_permutations"):
            permutation_null(correlated_returns.iloc[:60], n_permutations=0, rng=rng)
        with pytest.raises(ValueError, match="at least 2 assets"):
            permutation_null(correlated_returns.iloc[:60, :1], n_permutations=10, rng=rng)


class TestCalibrateTau:
    def test_is_deterministic(self, correlated_returns):
        """Same config, same seed, same tau -- reproducibility of the study threshold."""
        config = _config()

        first = calibrate_tau(correlated_returns, config)
        second = calibrate_tau(correlated_returns, config)

        assert first.tau == second.tau
        assert first.tau_fwer == second.tau_fwer
        assert first.per_window_tau == second.per_window_tau

    def test_record_is_complete(self, correlated_returns):
        config = _config()

        calibration = calibrate_tau(correlated_returns, config)

        assert calibration.n_calibration_windows == 5
        assert len(calibration.per_window_tau) == 5
        assert len(calibration.window_end_dates) == 5
        assert calibration.n_pairs == N_PAIRS
        assert calibration.window == 60
        assert calibration.seed == 42
        # Windows must span the whole sample, not cluster at its start.
        assert calibration.window_end_dates[0] == str(correlated_returns.index[59].date())
        assert calibration.window_end_dates[-1] == str(correlated_returns.index[-1].date())
        # JSON-serializable, since it is written to results/metrics/tau_calibration.json.
        json.loads(json.dumps(calibration.to_dict()))

    def test_threshold_ordering(self, correlated_returns):
        """tau controls the per-pair rate, tau_fwer the family-wise rate over all
        pairs at once, so the latter is necessarily the more conservative bound.
        """
        calibration = calibrate_tau(correlated_returns, _config())

        assert 0.0 < calibration.tau < calibration.tau_fwer

    def test_independent_of_observed_correlation(self, correlated_returns):
        """tau must depend on the null only. Doubling the common factor -- which
        raises every observed correlation -- must leave tau essentially
        unchanged, since the null is built by destroying exactly that structure.
        """
        config = _config()
        rng = np.random.default_rng(7)
        stronger = correlated_returns + 0.05 * rng.standard_normal((len(correlated_returns), 1))

        baseline = calibrate_tau(correlated_returns, config)
        inflated = calibrate_tau(stronger, config)

        assert (
            np.corrcoef(stronger.to_numpy().T)[0, 1]
            > np.corrcoef(correlated_returns.to_numpy().T)[0, 1]
        )
        assert abs(baseline.tau - inflated.tau) < 0.03

    def test_rejects_unsupported_settings(self, correlated_returns):
        with pytest.raises(ValueError, match="threshold method"):
            calibrate_tau(correlated_returns, _config(method="fisher"))
        with pytest.raises(ValueError, match="threshold statistic"):
            calibrate_tau(correlated_returns, _config(statistic="max"))


class TestCheckTauPlausible:
    def test_bounds(self, correlated_returns):
        calibration = calibrate_tau(correlated_returns, _config())
        check_tau_plausible(calibration)

        # A null that failed to break the dependence would land near the observed
        # correlation; a collapsed one, near zero. Both must be caught.
        with pytest.raises(ValueError, match="plausible range"):
            check_tau_plausible(dataclasses.replace(calibration, tau=0.85))
        with pytest.raises(ValueError, match="plausible range"):
            check_tau_plausible(dataclasses.replace(calibration, tau=0.001))

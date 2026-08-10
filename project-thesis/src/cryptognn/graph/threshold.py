"""Calibration of the correlation threshold tau for the crypto-gnn study.

Decides which correlations are strong enough to become edges of the graph. The
threshold is derived from a permutation null rather than picked by hand: each
return series is shuffled independently inside the window, which destroys the
cross-asset dependence while preserving each marginal -- fat tails included --
and the resulting distribution of "correlations that arise by chance alone"
sets tau at its (1 - alpha) quantile.

Preserving the marginals is the whole point. The textbook Fisher/Gaussian test
for rho assumes finite fourth moments; thesis Section 4.2 (correlation pitfalls)
argues that assumption is exactly what daily crypto returns violate. Resampling
the observed values sidesteps it instead of working around it.

Note this calibrates the *edge* threshold only. Marchenko-Pastur, which the
thesis invokes for the noise floor, governs the *spectrum* of the matrix
(collective structure), not any individual rho_ij; it is used separately in
graph.metrics.eigs_outside_mp() for Section 6.6.

Exports:
  - permutation_null(): null distribution of pairwise correlations for one window
  - calibrate_tau(): single study-wide tau plus two robustness thresholds
  - check_tau_plausible(): asserts the calibrated tau lands in its expected range
  - TauCalibration: frozen result record, JSON-serializable via to_dict()

Integration: called by scripts/02_build_graphs.py before graph construction;
  the result is saved to results/metrics/tau_calibration.json and the tau it
  carries is consumed by cryptognn.graph.build.apply_threshold().
Why a single tau for the whole study: a threshold recalibrated per window would
  make graph density incomparable across periods -- density would then measure
  the moving threshold as much as the market. Section 6.6 reads density as a
  crisis signal, which requires the yardstick to stay fixed.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cryptognn.config import Config
from cryptognn.graph.correlation import correlation_from_windows


@dataclass(frozen=True)
class TauCalibration:
    """Outcome of the tau calibration, with everything needed to reproduce it.

    Three thresholds are reported together so Section 6.2 can state how much the
    conclusions depend on the choice, instead of resting on a single number:

      - tau: the study threshold. Per-pair control at level alpha.
      - tau_fwer: family-wise control across all n_pairs simultaneously (quantile
        of the per-permutation maximum). Necessarily higher, hence a sparser and
        more conservative graph.
      - tau_fixed: a round, calibration-free value from config, to show the
        topology is not an artifact of the calibration procedure itself.
    """

    tau: float
    tau_fwer: float
    tau_fixed: float
    alpha: float
    n_permutations: int
    n_calibration_windows: int
    window: int
    n_pairs: int
    statistic: str
    seed: int
    window_end_dates: list[str]
    per_window_tau: list[float]
    per_window_tau_fwer: list[float]
    # Resulting graph density (mean/min/max/sd across windows) under each of the
    # three thresholds, filled in by scripts/02_build_graphs.py via
    # dataclasses.replace() once the graphs exist -- calibrate_tau() never sees
    # the correlation tensor, only the returns. Optional so the calibration
    # stands on its own, but in practice always written: without it, the fact
    # that the calibrated tau yields a near-complete graph on this universe
    # (density ~0.97) would be recorded nowhere, and Section 6.2 would present
    # density as a crisis signal without stating how little room it has to move.
    density: dict[str, dict[str, float]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TauCalibration:
        """Rebuild a calibration from its dict form, the inverse of to_dict().

        Lets cryptognn.artifacts.load_tau() hand back a typed record rather than
        a bare dict, so callers write `calibration.tau` -- which a type checker
        can verify -- instead of `calibration["tau"]`, which it cannot.
        """
        return cls(**raw)

    def to_json(self, path: str | Path) -> None:
        """Write the calibration record to `path` as indented JSON."""
        with Path(path).open("w") as f:
            json.dump(self.to_dict(), f, indent=2)


def permutation_null(
    returns_window: np.ndarray | pd.DataFrame,
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Null distribution of pairwise correlations for a single window of returns.

    Each of the `n_permutations` replicas shuffles every asset column
    independently along time. Independent per column is what makes this a null:
    a shared permutation would reorder the panel as a block and leave the
    cross-asset dependence intact. Shuffling values rather than drawing from a
    fitted distribution keeps each asset's marginal exactly as observed, so the
    heavy tails that inflate the sampling variance of rho are inherited by the
    null instead of being assumed away.

    `returns_window` is (T_w, N): T_w observations for N assets.

    Returns:
      pooled: (n_permutations * n_pairs,) every off-diagonal correlation from
        every replica, signed. Its (1 - alpha) quantile controls the per-pair
        false-edge rate.
      max_per_permutation: (n_permutations,) the largest correlation within each
        replica. Its (1 - alpha) quantile controls the family-wise rate across
        all n_pairs at once.

    Both are signed, not absolute: edges are thresholded on signed rho (see
    graph.build.apply_threshold), so the null must be one-sided in the same
    direction to mean anything.
    """
    values = np.asarray(returns_window, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"returns_window must be 2-D (T_w, N), got shape {values.shape}")
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1, got {n_permutations}")

    n_obs, n_assets = values.shape
    if n_assets < 2:
        raise ValueError(f"need at least 2 assets to form a pair, got {n_assets}")

    # (B, T_w, N) view over the same buffer; permuted() copies rather than
    # writing through it, so the caller's window is never mutated.
    replicas = np.broadcast_to(values, (n_permutations, n_obs, n_assets))
    shuffled = rng.permuted(replicas, axis=1)

    corr = correlation_from_windows(shuffled.transpose(0, 2, 1))  # (B, N, N)

    rows, cols = np.triu_indices(n_assets, k=1)
    pairs = corr[:, rows, cols]  # (B, n_pairs)

    return pairs.reshape(-1), pairs.max(axis=1)


def calibrate_tau(returns: pd.DataFrame, config: Config) -> TauCalibration:
    """Calibrate the study-wide correlation threshold on a permutation null.

    Runs the null on `config.graph.threshold.n_calibration_windows` windows
    spread evenly across the sample, takes the (1 - alpha) quantile within each,
    and reports the **median** across windows. Spreading the windows keeps the
    threshold from being set by whichever regime happens to sit at the start of
    the period; the median keeps a single turbulent window from dragging it.

    The windows are `config.graph.window` observations long -- the same length
    used to build the graph, since the sampling variance of rho depends on it,
    and a threshold calibrated at another length would not transfer.

    Determinism: a single Generator seeded from `config.seed` is threaded
    through the windows in order, so re-running reproduces tau exactly.
    """
    threshold_config = config.graph.threshold
    if threshold_config.method != "permutation":
        raise ValueError(
            f"unsupported threshold method {threshold_config.method!r}; only 'permutation' is implemented"
        )
    if threshold_config.statistic != "pooled":
        raise ValueError(
            f"unsupported threshold statistic {threshold_config.statistic!r}; only 'pooled' is implemented"
        )

    window = config.graph.window
    n_windows = threshold_config.n_calibration_windows
    alpha = threshold_config.alpha
    n_obs = len(returns)

    if n_obs < window:
        raise ValueError(f"need at least window={window} observations to calibrate, got {n_obs}")

    # Equispaced *end* positions, first window flush with the start of the sample
    # and last flush with its end, so the calibration spans the whole period.
    starts = np.linspace(0, n_obs - window, n_windows).round().astype(int)

    rng = np.random.default_rng(config.seed)
    quantile = 1.0 - alpha

    per_window_tau: list[float] = []
    per_window_tau_fwer: list[float] = []
    window_end_dates: list[str] = []

    for start in starts:
        window_returns = returns.iloc[start : start + window]
        pooled, max_per_permutation = permutation_null(
            window_returns, threshold_config.n_permutations, rng
        )
        per_window_tau.append(float(np.quantile(pooled, quantile)))
        per_window_tau_fwer.append(float(np.quantile(max_per_permutation, quantile)))
        window_end_dates.append(str(returns.index[start + window - 1].date()))

    n_assets = returns.shape[1]

    return TauCalibration(
        tau=float(np.median(per_window_tau)),
        tau_fwer=float(np.median(per_window_tau_fwer)),
        tau_fixed=float(threshold_config.tau_fixed),
        alpha=alpha,
        n_permutations=threshold_config.n_permutations,
        n_calibration_windows=n_windows,
        window=window,
        n_pairs=n_assets * (n_assets - 1) // 2,
        statistic=threshold_config.statistic,
        seed=config.seed,
        window_end_dates=window_end_dates,
        per_window_tau=per_window_tau,
        per_window_tau_fwer=per_window_tau_fwer,
    )


def check_tau_plausible(
    calibration: TauCalibration,
    *,
    lower_bound: float = 0.10,
    upper_bound: float = 0.40,
) -> None:
    """Assert the calibrated tau lands where the null implies it should, raising
    a ValueError otherwise -- a violation means the permutation is broken, not
    that the market is unusual.

    Under the null with T_w = 60 the sampling standard deviation of rho is about
    1/sqrt(T_w - 1) ~ 0.13, putting the 95th percentile near 1.645 * 0.13 ~ 0.21;
    heavy tails widen it somewhat, so roughly 0.20-0.28 is expected. The default
    bounds are deliberately loose around that range: they catch the failure
    modes, not a merely surprising value.

      - tau at or above `upper_bound` is the signature of a permutation that
        failed to break the dependence (e.g. one shared permutation applied to
        every column, which leaves the real correlations intact).
      - tau at or below `lower_bound` points the other way: a null whose
        correlations are all near zero, e.g. windows collapsed to a constant.

    The bounds apply to tau only. tau_fwer is a maximum over n_pairs and is
    expected to sit well above them by construction.
    """
    if not lower_bound < calibration.tau < upper_bound:
        raise ValueError(
            f"calibrated tau = {calibration.tau:.4f} outside the plausible range "
            f"({lower_bound}, {upper_bound}) for window={calibration.window} -- "
            "suspect the permutation null, not the data"
        )

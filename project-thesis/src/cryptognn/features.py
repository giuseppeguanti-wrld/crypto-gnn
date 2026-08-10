"""Node features and per-fold standardization for the crypto-gnn study.

Turns the validated return and volume panels into the (T, N, F) tensor a model
reads at each prediction origin, and assembles that tensor, the raw returns and
the aligned graph into the container the walk-forward harness consumes.

The whole module obeys one rule, and exists mostly to make that rule checkable in
one place: **row t describes the close of day t and nothing later**. Every window
is causal and closed at t, exactly like the correlation window behind a_hat[t].
The first rows, which have no history to speak of, stay NaN rather than being
back-filled -- a fabricated value there is a look-ahead in disguise, since it
would be built from the only data available, which is the future.

Exports:
  - build_node_features(): (T, N, 8) causal feature tensor
  - feature_names(): the channel labels, for tables and figures
  - FoldStandardizer: mean/scale fitted on a fold's train split alone
  - build_study_data(): artifacts on disk -> WalkforwardData

Integration: build_study_data() is the single assembly point used by
  scripts/04_run_baselines.py and 05_run_gcn.py; FoldStandardizer is applied by
  the models that need it (the GCN of Sprint 4), not by the harness.
Why the standardizer is per fold: fitting mu and sigma on the whole sample leaks
  the test period's scale into training. Section 5.4 of the thesis calls that the
  most insidious variant of look-ahead precisely because the result still looks
  plausible -- nothing crashes, the numbers merely improve.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from cryptognn.artifacts import load_a_hat, load_corr_index, load_returns, load_volumes
from cryptognn.config import Config
from cryptognn.evaluation.walkforward import Segment, WalkforwardData, align_graph
from cryptognn.windows import causal_windows

# A spread below this fraction of a channel's own level counts as no spread.
# `sd > 0` is not enough: the standard deviation of a constant array comes out at
# ~1e-16 rather than exactly 0, because the mean it subtracts is itself rounded.
# Dividing that residue by itself amplifies pure noise to O(1) -- a constant
# channel would standardize to -1 or 3.7 instead of 0, and nothing would warn.
_SCALE_FLOOR = 1e-12


def _safe_scale(spread: np.ndarray, level: np.ndarray) -> np.ndarray:
    """`spread`, with negligible values replaced by 1 so division leaves them alone.

    Scaled by the channel's own level, since what counts as negligible for log
    volume around 20 is not what counts for a daily return around 0.001. Where
    the spread is negligible the numerator is too, so the standardized value
    comes out at ~0 -- the right reading for a quantity that has not moved.
    """
    return np.where(spread > _SCALE_FLOOR * np.maximum(np.abs(level), 1.0), spread, 1.0)


def feature_names(config: Config) -> list[str]:
    """Channel labels, in the order build_node_features() stacks them."""
    names = [f"r_lag{k}" for k in range(config.features.lags)]
    names += [f"rv_{window}" for window in config.features.vol_windows]
    if config.features.use_volume:
        names.append(f"logvol_z_{max(config.features.vol_windows)}")
    return names


def build_node_features(
    returns: pd.DataFrame,
    volumes: pd.DataFrame | None,
    config: Config,
) -> np.ndarray:
    """The (T, N, F) node feature tensor, every channel causal and closed at t.

    Channels, in order (F = 8 with the shipped config):
      0..4  lagged returns r_t, r_{t-1}, ..., r_{t-4}. r_t is included: it is
            known at the close of day t, which is when the forecast is made.
      5..6  realized volatility sqrt(mean(r^2)) over 5 and 20 days. Squared
            returns rather than a variance around a mean: at daily frequency the
            mean is indistinguishable from zero and estimating it only adds noise.
      7     twenty-day z-score of log volume. Logs first, so the z-score is
            invariant to each asset's unit of account; the window is the longest
            of vol_windows, so one config entry governs both.

    `volumes` may be None when config.features.use_volume is False, which yields
    F = 7. It is otherwise required and must cover every date of `returns`: the
    volume panel is one row longer (it keeps the first day, which the returns
    lose to differencing), and is aligned here.

    The first max(window) - 1 rows are NaN. Folds start at position 59 (the
    graph offset), well past them, and run_walkforward() rejects any fold that
    would reach into them.
    """
    values = returns.to_numpy(dtype=np.float64)
    lags = config.features.lags
    vol_windows = list(config.features.vol_windows)

    lag_windows = causal_windows(values, lags)
    # Column -1 of a causal window is r_t, column -1-k is r_{t-k}.
    channels = [lag_windows[:, -1 - k, :] for k in range(lags)]
    channels += [np.sqrt(np.mean(causal_windows(values, window) ** 2, axis=1)) for window in vol_windows]

    if config.features.use_volume:
        if volumes is None:
            raise ValueError("config.features.use_volume is True but no volume panel was given")
        missing = returns.index.difference(volumes.index)
        if len(missing) > 0:
            raise ValueError(f"Volume panel misses {len(missing)} return dates, first {missing[0]}")

        aligned = volumes.loc[returns.index, list(returns.columns)].to_numpy(dtype=np.float64)
        if (aligned <= 0).any():
            raise ValueError("Volume panel contains non-positive values; log volume is undefined there")

        log_volume = np.log(aligned)
        history = causal_windows(log_volume, max(vol_windows))
        mean = history.mean(axis=1)
        channels.append((log_volume - mean) / _safe_scale(history.std(axis=1), mean))

    features = np.stack(channels, axis=-1)
    expected = lags + len(vol_windows) + int(config.features.use_volume)
    if features.shape != (*values.shape, expected):
        raise ValueError(f"Built {features.shape} features, expected {(*values.shape, expected)}")
    return features


class FoldStandardizer:
    """Per-asset, per-channel standardization fitted on one fold's train split.

    mu and sigma have shape (N, F): each asset's each channel gets its own,
    because the assets differ by an order of magnitude in volatility and a shared
    scale would let the most volatile of them dominate the shared weights of the
    GCN. Both are estimated on the training rows only, then applied unchanged to
    validation and test -- the point of the class, and what
    test_standardizer_train_only holds it to.

    Deliberately not an sklearn transformer: sklearn is not a dependency of this
    study, and the only behaviour needed is two arrays and a subtraction.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> FoldStandardizer:
        """Estimate mu and sigma over the sample axis of a (n, N, F) train block."""
        if x.ndim != 3:
            raise ValueError(f"Expected a (n, N, F) block, got shape {x.shape}")
        if not np.isfinite(x).all():
            raise ValueError(
                "Non-finite values in the training block: the fold most likely reaches into "
                "the feature warm-up rows, whose history is incomplete"
            )

        self.mean_ = x.mean(axis=0)
        # A channel that never moves within a fold standardizes to ~0 rather than
        # to noise amplified to O(1); see _safe_scale.
        self.scale_ = _safe_scale(x.std(axis=0), self.mean_)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("FoldStandardizer.transform() called before fit()")
        if x.shape[1:] != self.mean_.shape:
            raise ValueError(f"Block of shape {x.shape} does not match the fitted {self.mean_.shape}")
        return (x - self.mean_) / self.scale_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)

    def transform_segment(self, segment: Segment) -> Segment:
        """The same segment with standardized features; every other array is untouched.

        Returns and lags stay raw on purpose: they are the input of the
        autoregressive baselines, which are estimated on returns in their own
        units, and of the target, which must remain comparable across models.
        """
        if segment.features is None:
            raise ValueError("Segment carries no features to standardize")
        return replace(segment, features=self.transform(segment.features))


def build_study_data(config: Config, fwer: bool = False) -> WalkforwardData:
    """Assemble the study's artifacts into the container the harness consumes.

    The single place where files on disk become model input: returns, node
    features and the graph, all indexed by the same 2006 panel rows, so a fold
    position means one thing throughout. scripts/04 and 05 call this instead of
    repeating the sequence, which is how the two stay comparable.

    `fwer=True` selects the pre-registered sparser graph variant (A_hat_fwer),
    decided in Sprint 2 before any predictive result was seen.
    """
    returns = load_returns()
    volumes = load_volumes() if config.features.use_volume else None

    return WalkforwardData(
        dates=returns.index,
        assets=tuple(returns.columns),
        returns=returns.to_numpy(dtype=np.float64),
        graph_offset=config.graph.window - 1,
        # Deep enough for whichever consumer needs the longest history: the
        # feature lags, or the lag order the AR/VAR baselines may select.
        lookback=max(config.features.lags, config.model.var.max_lag, config.model.ar.max_lag),
        features=build_node_features(returns, volumes, config),
        a_hat=align_graph(load_a_hat(fwer=fwer), load_corr_index(), returns.index),
    )

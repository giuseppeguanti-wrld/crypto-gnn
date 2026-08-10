"""Configuration management for the crypto-gnn study.

Provides a frozen, type-safe configuration object built from config/default.yaml.
All study parameters (universe, period, graph window, model hyperparameters, backtest costs, etc.)
flow through this module. Acts as the single source of truth: no magic numbers in code.

Exports:
  - load_config(path): Parse YAML and construct the nested Config dataclass
  - config_hash(path): SHA-1 of normalized YAML for reproducibility manifest
  - Config: Top-level configuration dataclass with nested sections (data, graph, model, etc.)

Integration: Imported by scripts, models, and evaluation code to read parameters.
Why it exists: Eliminates hardcoded values, makes hyperparameter sweeps and ablations trivial,
  and ensures run_manifest.json can independently verify what configuration was used.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataConfig:
    source: str
    quote: str
    interval: str
    start: date
    end: date
    symbols: list[str]


@dataclass(frozen=True)
class ThresholdConfig:
    method: str
    alpha: float
    n_permutations: int
    n_calibration_windows: int
    statistic: str
    tau_fixed: float


@dataclass(frozen=True)
class GraphConfig:
    window: int
    weight: str
    self_loops: bool
    threshold: ThresholdConfig


@dataclass(frozen=True)
class FeaturesConfig:
    lags: int
    vol_windows: list[int]
    use_volume: bool


@dataclass(frozen=True)
class WalkforwardConfig:
    train: int
    val: int
    test: int
    step: int
    mode: str


@dataclass(frozen=True)
class GCNConfig:
    hidden: list[int]
    dropout: list[float]
    lr: float
    weight_decay: float
    epochs: int
    patience: int
    seeds: list[int]


@dataclass(frozen=True)
class VARConfig:
    max_lag: int
    ic: str
    fixed_lag: int


@dataclass(frozen=True)
class ARConfig:
    max_lag: int
    ic: str


@dataclass(frozen=True)
class ModelConfig:
    gcn: GCNConfig
    var: VARConfig
    ar: ARConfig


@dataclass(frozen=True)
class BacktestConfig:
    cost_bps: int


@dataclass(frozen=True)
class Config:
    data: DataConfig
    graph: GraphConfig
    features: FeaturesConfig
    walkforward: WalkforwardConfig
    model: ModelConfig
    backtest: BacktestConfig
    seed: int


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)

    return Config(
        data=DataConfig(**raw["data"]),
        graph=GraphConfig(
            window=raw["graph"]["window"],
            weight=raw["graph"]["weight"],
            self_loops=raw["graph"]["self_loops"],
            threshold=ThresholdConfig(**raw["graph"]["threshold"]),
        ),
        features=FeaturesConfig(**raw["features"]),
        walkforward=WalkforwardConfig(**raw["walkforward"]),
        model=ModelConfig(
            gcn=GCNConfig(**raw["model"]["gcn"]),
            var=VARConfig(**raw["model"]["var"]),
            ar=ARConfig(**raw["model"]["ar"]),
        ),
        backtest=BacktestConfig(**raw["backtest"]),
        seed=raw["seed"],
    )


def config_hash(path: str | Path) -> str:
    """SHA-1 of the YAML content, normalized (sorted keys) so key order
    and formatting differences do not change the hash. Rereads the file
    independently of load_config() so it can be used on its own when
    writing run_manifest.json.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    normalized = json.dumps(raw, sort_keys=True, default=str)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()

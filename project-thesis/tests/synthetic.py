"""Shape and parameter constants of the shared synthetic fixtures.

Kept apart from conftest.py deliberately. conftest is a pytest mechanism: pytest
loads it, and its job is to *provide fixtures by injection*. Importing values out
of it -- which several test modules used to do -- worked only because pytest's
default import mode happened to put tests/ on sys.path, and broke the moment the
suite was organized into subpackages. A module that other modules import from
should be a module, declared as such.

The constants live here rather than being duplicated per test file because the
fixtures in conftest.py are built from them and several tests assert against the
resulting shapes: a panel of N_ASSETS columns must yield N_PAIRS correlations,
and a fold of a WF_N_OBS-row container must not reach past its end. Two copies of
those numbers would let a fixture and its assertions disagree.

Importable from any test module because pyproject.toml adds tests/ to
`pythonpath`.
"""
from __future__ import annotations

# Synthetic return panel of the graph fixtures.
N_ASSETS = 6
N_PAIRS = N_ASSETS * (N_ASSETS - 1) // 2

# The threshold calibrated on the study data, used across the construction and
# metric tests as a realistic value rather than a round invented one.
TAU = 0.2145

# Shape of the synthetic walk-forward container. Small enough to reason about,
# large enough to hold several folds of a few observations each.
WF_N_OBS = 120
WF_N_ASSETS = 4
WF_N_FEATURES = 3
WF_GRAPH_OFFSET = 10
WF_LOOKBACK = 3

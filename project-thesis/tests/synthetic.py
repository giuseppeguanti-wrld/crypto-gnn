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

# The seven models the chapter compares, in the order the results are reported,
# plus the backtest's reference row. Shared by the table and summary fixtures
# because both build frames keyed on them and a mismatch between the two would
# look like a missing model rather than like a fixture disagreeing with itself.
MODELS = ("zero", "mean", "ar", "var-bic", "var-p5", "gcn", "gcn-nograph")
GCN_ARMS = ("gcn", "gcn-nograph")
BACKTEST_REFERENCE = "buy-and-hold"

# The study's own universe rather than a shortened list: the observations-per-
# parameter column of the model table is a function of how many assets there are,
# so a shorter list would make the dimensionality claim come out at a different
# number than the one the chapter argues from.
SYMBOLS = (
    "BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE", "DOT",
    "AVAX", "LINK", "LTC", "BCH", "XLM", "TRX", "ETC",
)
# Only the first three carry descriptive statistics: the table's contract is one
# row per described asset, which a subset exercises and a full list would not.
DESCRIBED = SYMBOLS[:3]

# Panel length that yields the study's 24 folds under the config fixture's
# walk-forward settings, so the fold-level fixtures have the real fold count.
# The two are asserted to agree in tests/test_summary.py rather than trusted:
# if make_folds ever laid them out differently, every per-fold fixture built
# from N_FOLDS would silently stop matching the folds it describes.
N_OBS = 2006
N_FOLDS = 24

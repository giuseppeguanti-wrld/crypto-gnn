"""Fixtures shared by the drawing, composition and style tests.

Every module in tests/viz/ renders with the study's own rcParams and must leave
no figure open behind it, so the style fixture is autouse here rather than
repeated in each file -- which is what it was before this package existed.

Nothing here reads data/ or results/: the suite must run on a fresh clone,
before any script has been executed.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

from cryptognn.viz.style import apply_style

matplotlib.use("Agg")  # headless rendering; must be selected before pyplot loads

import matplotlib.pyplot as plt

N_ASSETS = 6


@pytest.fixture(autouse=True)
def _style():
    """The study's rcParams, and a clean figure registry after every test.

    Without the teardown a test that forgets to close its figure would leak into
    the next one's `plt.get_fignums()` check, which is precisely the assertion
    the drawing contract rests on.
    """
    apply_style(usetex=False)
    yield
    plt.close("all")


@pytest.fixture
def block_corr() -> np.ndarray:
    """Two clearly separated blocks: assets 0-2 and 3-5."""
    corr = np.full((N_ASSETS, N_ASSETS), 0.1)
    corr[:3, :3] = 0.9
    corr[3:, 3:] = 0.8
    np.fill_diagonal(corr, 1.0)
    return corr


@pytest.fixture
def topology_frame() -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=300, freq="D", name="date")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "mean_correlation": rng.uniform(0.3, 0.9, len(index)),
            "graph_density": rng.uniform(0.8, 1.0, len(index)),
            "graph_density_fwer": rng.uniform(0.4, 1.0, len(index)),
        },
        index=index,
    )

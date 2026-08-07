"""Filesystem path constants for the crypto-gnn study.

Centralizes every directory the pipeline reads from or writes to, resolved relative
to the project root (project-thesis/), regardless of the current working directory
the scripts are invoked from.

Exports:
  - ROOT: project-thesis/ root directory
  - DATA_RAW, DATA_PROCESSED: raw klines and processed prices/returns/correlations
  - RESULTS, RESULTS_METRICS, RESULTS_FIGURES, RESULTS_TABLES: pipeline outputs
  - FIGURES: PDF figures destination in ../latex-thesis/figures/
  - ensure_dirs(): idempotent creation of all directories above

Integration: Imported by scripts/*.py and any module reading/writing to disk, so paths
  are never hardcoded or duplicated across the codebase.
Why it exists: Keeps directory layout in one place; moving or renaming a directory
  means editing this file only, not grepping the whole repo for string literals.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_PROCESSED = DATA / "processed"

RESULTS = ROOT / "results"
RESULTS_METRICS = RESULTS / "metrics"
RESULTS_FIGURES = RESULTS / "figures"
RESULTS_TABLES = RESULTS / "tables"

FIGURES = ROOT.parent / "latex-thesis" / "figures"

CONFIG = ROOT / "config"


def ensure_dirs() -> None:
    """Create every directory referenced above if it does not already exist.

    Safe to call multiple times (mkdir with exist_ok=True): scripts call this at
    startup instead of assuming the tree from PLANNING.md Section 2 is already in place.
    """
    for directory in (
        DATA_RAW,
        DATA_PROCESSED,
        RESULTS_METRICS,
        RESULTS_FIGURES,
        RESULTS_TABLES,
    ):
        directory.mkdir(parents=True, exist_ok=True)

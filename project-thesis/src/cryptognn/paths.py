"""Filesystem path constants for the crypto-gnn study.

Centralizes every directory the pipeline reads from or writes to, resolved
relative to the project root (project-thesis/), regardless of the working
directory the scripts are invoked from. That independence is the point: an entry
point should behave the same whether it is run from project-thesis/, from the
repository root, or from an IDE with an unrelated working directory.

Exports:
  - ROOT: project-thesis/ root directory
  - DATA_RAW, DATA_PROCESSED: raw klines and processed prices/returns/correlations
  - RESULTS, RESULTS_METRICS, RESULTS_FIGURES, RESULTS_TABLES: pipeline outputs
  - FIGURES: PDF figures destination in ../latex-thesis/figures/
  - CONFIG, DEFAULT_CONFIG, DEFAULT_EVENTS: configuration directory and its files
  - ensure_dirs(): idempotent creation of the output directories

Integration: imported by scripts/*.py and any module reading or writing to disk,
  so paths are never hardcoded or duplicated across the codebase.
Why it exists: keeps the directory layout in one place; moving or renaming a
  directory means editing this file only, not grepping the repo for string
  literals. DEFAULT_CONFIG and DEFAULT_EVENTS are here rather than in each
  script's argparse defaults for the same reason -- a default spelled
  "config/default.yaml" is relative to the working directory, and silently
  breaks the moment a script is launched from anywhere else.
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
DEFAULT_CONFIG = CONFIG / "default.yaml"
DEFAULT_EVENTS = CONFIG / "events.yaml"

# Files and directories that must exist directly under a genuine project root.
# Checked rather than assumed -- see _verify_layout().
_LAYOUT_MARKERS = ("pyproject.toml", "config", "src/cryptognn")


def _verify_layout(root: Path) -> None:
    """Raise unless `root` really is the project-thesis/ directory.

    ROOT is derived from this file's location, three levels up from
    src/cryptognn/paths.py. That holds for a source tree and for an editable
    install (which is how this project is installed), but not for a regular
    `pip install .`: the package would then live in site-packages/, and ROOT
    would point at some unrelated directory three levels above it.

    The failure that would cause is worse than a crash. Nothing here reads a
    file at import time, so the pipeline would run to completion and quietly
    create data/ and results/ in the wrong place, writing every artifact there.
    That is discovered only when someone goes looking for the results. Checking
    the layout converts a silent misplacement into an immediate, explicit error.
    """
    missing = [marker for marker in _LAYOUT_MARKERS if not (root / marker).exists()]
    if missing:
        raise RuntimeError(
            f"cryptognn.paths resolved the project root to {root}, which does not look like "
            f"the project-thesis/ directory: missing {', '.join(missing)}.\n"
            "This normally means the package was installed non-editable, so it lives in "
            "site-packages/ and its paths point outside the project. Reinstall with:\n"
            "    uv pip install -e . --no-deps"
        )


_verify_layout(ROOT)


def ensure_dirs() -> None:
    """Create every output directory referenced above if it does not already exist.

    Safe to call multiple times (mkdir with exist_ok=True): scripts call this at
    startup instead of assuming the tree is already in place. Only output
    directories are created -- config/ and src/ are inputs, and their absence is
    a layout error caught by _verify_layout(), not something to paper over by
    creating an empty directory.
    """
    for directory in (
        DATA_RAW,
        DATA_PROCESSED,
        RESULTS_METRICS,
        RESULTS_FIGURES,
        RESULTS_TABLES,
    ):
        directory.mkdir(parents=True, exist_ok=True)

"""crypto-gnn: graph neural networks on dynamic cryptocurrency correlation graphs.

Experimental code for Chapter 6 (case study) of the thesis. The package is
installed in editable mode (`pip install -e .`), so every entry point in
scripts/ and every test imports it as `cryptognn.*` with no path manipulation.

The version is read from the installed distribution metadata rather than
duplicated here, so pyproject.toml stays the single source of truth; it is
recorded in results/run_manifest.json to tie a set of results to the code that
produced them.
"""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cryptognn")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]

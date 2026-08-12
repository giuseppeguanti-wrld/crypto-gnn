"""Reading and writing the pipeline's on-disk artifacts.

Every file the pipeline produces or consumes passes through here, so a filename,
a serialization format, and the command that produces it are each written down
once. The scripts of scripts/01-08 use it, and so will the Streamlit explorer,
which must read exactly the artifacts the thesis figures were drawn from.

Three properties this centralization buys:

  - **Save and load cannot drift.** Each artifact has a matching pair, sharing
    one path constant; renaming a file is a single edit rather than a grep.
  - **One conversion, not several.** Arrays are stored float32 (size) and used
    float64 (numerics), and the correlation index is a DatetimeIndex on the way
    in and out. Before this module the two consumers of corr_index converted it
    in two different ways.
  - **A missing artifact is a catchable error.** MissingArtifactError derives
    from FileNotFoundError, hence from Exception, so a GUI can catch it and
    render the remedy. Raising SystemExit here -- which derives from
    BaseException and slips past `except Exception` -- would let a missing file
    tear down a running app instead of producing a message.

Exports:
  - MissingArtifactError: carries the missing path and the command that makes it
  - path constants and corr_path(window)
  - load_/save_ pairs for prices, returns, volumes, correlations, graphs, tau,
    the stylized-fact tables, topology, the event study, the walk-forward
    predictions, diagnostics, accuracy summary and Diebold-Mariano matrix, the
    backtest summary and equity curves, the generated LaTeX tables, the
    write-up summary and the run manifest
  - publish_to_thesis(): copy the finished figures and tables into ../latex-thesis/

Integration: used by scripts/02 through 08, and by app/streamlit_app.py.
"""
from __future__ import annotations

import json
import shutil
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from cryptognn import paths
from cryptognn.graph.threshold import TauCalibration

# The command that regenerates each artifact, quoted back to the user when one
# is missing. Kept next to the paths so the two cannot fall out of step.
COMMAND_DOWNLOAD = "python scripts/01_download_data.py"
COMMAND_BUILD = "python scripts/02_build_graphs.py"
COMMAND_TOPOLOGY = "python scripts/03_topology_analysis.py"
COMMAND_BASELINES = "python scripts/04_run_baselines.py"
COMMAND_GCN = "python scripts/05_run_gcn.py"
COMMAND_BACKTEST = "python scripts/06_run_backtest.py"
COMMAND_FIGURES = "python scripts/07_make_figures.py --usetex"
COMMAND_TABLES = "python scripts/08_make_tables.py"


def _predictions_command(name: str) -> str:
    """Which script produces the prediction artifacts of one run group.

    The walk-forward artifacts are parametrized by group name, but the command
    that makes them is not: a missing predictions_gcn.parquet has to point at
    script 05, not at 04, or MissingArtifactError hands the user a command that
    completes successfully and leaves the file still missing.
    """
    return COMMAND_BASELINES if name == "baselines" else COMMAND_GCN


class MissingArtifactError(FileNotFoundError):
    """An artifact the pipeline expects has not been produced yet.

    Carries both the path and the command that creates it, so every consumer --
    a script printing to a terminal, an app rendering an error panel -- can tell
    the user what to run without hardcoding pipeline knowledge of its own.
    """

    def __init__(self, path: Path, command: str) -> None:
        self.path = path
        self.command = command
        super().__init__(f"Missing artifact: {path}\nRun it first with:\n    {command}")


def corr_path(window: int) -> Path:
    """Path of the rolling correlation tensor for a given window length.

    The only artifact whose name carries a parameter: the window is part of the
    filename so several window lengths can coexist for the sensitivity analysis
    of Sprint 6 without overwriting one another.
    """
    return paths.DATA_PROCESSED / f"corr_{window}.npy"


def _processed(name: str) -> Path:
    return paths.DATA_PROCESSED / name


def _metrics(name: str) -> Path:
    return paths.RESULTS_METRICS / name


def _require(path: Path, command: str) -> Path:
    if not path.exists():
        raise MissingArtifactError(path, command)
    return path


# --------------------------------------------------------------------------
# Prices and returns (scripts/02)
# --------------------------------------------------------------------------


def save_prices(prices: pd.DataFrame) -> Path:
    path = _processed("prices.parquet")
    prices.to_parquet(path)
    return path


def load_prices() -> pd.DataFrame:
    return pd.read_parquet(_require(_processed("prices.parquet"), COMMAND_BUILD))


def save_returns(returns: pd.DataFrame) -> Path:
    path = _processed("returns.parquet")
    returns.to_parquet(path)
    return path


def load_returns() -> pd.DataFrame:
    return pd.read_parquet(_require(_processed("returns.parquet"), COMMAND_BUILD))


def save_volumes(volumes: pd.DataFrame) -> Path:
    path = _processed("volumes.parquet")
    volumes.to_parquet(path)
    return path


def load_volumes() -> pd.DataFrame:
    """Traded volume on the price index -- the input of the node feature of Section 6.3.

    Kept apart from prices rather than stored as one multi-column frame: the two
    are read by different consumers (prices by nothing downstream, volume by the
    feature builder), and a single wide frame would make every reader carry a
    column selection.
    """
    return pd.read_parquet(_require(_processed("volumes.parquet"), COMMAND_BUILD))


# --------------------------------------------------------------------------
# Rolling correlations (scripts/02)
# --------------------------------------------------------------------------


def save_corr(corr: np.ndarray, index: pd.DatetimeIndex, window: int) -> Path:
    """Store the correlation tensor and the closing date of each window.

    The tensor goes to disk as float32: at (T, N, N) it is the study's largest
    artifact and the extra precision buys nothing that survives the next
    eigendecomposition. Timestamps are written naive (tz stripped) because .npy
    has no timezone concept; load_corr() restores a DatetimeIndex.
    """
    np.save(corr_path(window), corr.astype(np.float32))
    naive = pd.DatetimeIndex(index).tz_localize(None) if index.tz is not None else pd.DatetimeIndex(index)
    np.save(_processed("corr_index.npy"), naive.to_numpy())
    return corr_path(window)


def load_corr_index() -> pd.DatetimeIndex:
    """The closing date of every correlation window, as a DatetimeIndex.

    Available on its own because a caller aligning the graph to the return panel
    needs the dates and not the tensor: cryptognn.features.build_study_data()
    would otherwise read 1.7 MB of correlations to look at their labels.
    """
    return pd.DatetimeIndex(np.load(_require(_processed("corr_index.npy"), COMMAND_BUILD)), name="date")


def load_corr(window: int) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """The correlation tensor as float64, with its dates as a DatetimeIndex.

    Both conversions happen here and only here. Callers previously did them
    inline and disagreed: one script kept corr_index as a raw array and applied
    pd.to_datetime() further down, the other wrapped it at load time.
    """
    corr = np.load(_require(corr_path(window), COMMAND_BUILD)).astype(np.float64)
    return corr, load_corr_index()


# --------------------------------------------------------------------------
# Graphs (scripts/02)
# --------------------------------------------------------------------------


def save_graphs(
    w_full: np.ndarray,
    w_thresh: np.ndarray,
    a_hat: np.ndarray,
    a_hat_fwer: np.ndarray,
) -> list[Path]:
    """Store the four graph tensors as float32, in one call so none is forgotten."""
    written = []
    for name, array in (
        ("W_full", w_full),
        ("W_thresh", w_thresh),
        ("A_hat", a_hat),
        ("A_hat_fwer", a_hat_fwer),
    ):
        path = _processed(f"{name}.npy")
        np.save(path, array.astype(np.float32))
        written.append(path)
    return written


def _load_graph(name: str) -> np.ndarray:
    return np.load(_require(_processed(f"{name}.npy"), COMMAND_BUILD)).astype(np.float64)


def load_w_full() -> np.ndarray:
    """Complete Mantegna weight graph -- the substrate of the spectral metrics."""
    return _load_graph("W_full")


def load_w_thresh() -> np.ndarray:
    """Thresholded graph -- the substrate of graph density."""
    return _load_graph("W_thresh")


def load_a_hat(fwer: bool = False) -> np.ndarray:
    """Renormalized adjacency, the GCN substrate.

    `fwer=True` returns the pre-registered robustness variant built at the
    family-wise threshold, which is sparser than the calibrated one.
    """
    return _load_graph("A_hat_fwer" if fwer else "A_hat")


# --------------------------------------------------------------------------
# Threshold calibration (scripts/02)
# --------------------------------------------------------------------------


def save_tau(calibration: TauCalibration) -> Path:
    path = _metrics("tau_calibration.json")
    calibration.to_json(path)
    return path


def load_tau() -> TauCalibration:
    """The calibration as a typed record, not a dict: callers write
    `calibration.tau`, which a type checker can verify.
    """
    with _require(_metrics("tau_calibration.json"), COMMAND_BUILD).open() as f:
        return TauCalibration.from_dict(json.load(f))


# --------------------------------------------------------------------------
# Topology and event study (scripts/03)
# --------------------------------------------------------------------------


def save_topology(topology: pd.DataFrame) -> Path:
    path = _metrics("topology.parquet")
    topology.to_parquet(path)
    return path


def load_topology() -> pd.DataFrame:
    return pd.read_parquet(_require(_metrics("topology.parquet"), COMMAND_TOPOLOGY))


# --------------------------------------------------------------------------
# Walk-forward predictions (scripts/04, and 05 from Sprint 4)
# --------------------------------------------------------------------------


def save_predictions(predictions: pd.DataFrame, name: str = "baselines") -> Path:
    """The long predictions table of one run group (baselines, gcn).

    Parametrized by name rather than split into two functions: the baselines and
    the GCN produce the identical schema -- (fold, date, asset, y_true, y_pred,
    model) -- and Sprint 4 concatenates the two files to build one comparison.
    """
    path = _metrics(f"predictions_{name}.parquet")
    predictions.to_parquet(path)
    return path


def load_predictions(name: str = "baselines") -> pd.DataFrame:
    return pd.read_parquet(_require(_metrics(f"predictions_{name}.parquet"), _predictions_command(name)))


def save_run_diagnostics(diagnostics: pd.DataFrame, name: str = "baselines") -> Path:
    path = _metrics(f"diagnostics_{name}.parquet")
    diagnostics.to_parquet(path)
    return path


def load_run_diagnostics(name: str = "baselines") -> pd.DataFrame:
    """Per-fold, per-model record of what each fit decided -- the VAR's selected
    lag order and parameter count among it, which Section 6.4 reports directly.
    """
    return pd.read_parquet(_require(_metrics(f"diagnostics_{name}.parquet"), _predictions_command(name)))


def save_summary(summary: pd.DataFrame, name: str = "baselines") -> Path:
    """One accuracy row per model: the body of the comparison table."""
    path = _metrics(f"summary_{name}.parquet")
    summary.to_parquet(path)
    return path


def load_summary(name: str = "baselines") -> pd.DataFrame:
    return pd.read_parquet(_require(_metrics(f"summary_{name}.parquet"), _predictions_command(name)))


def save_dm_matrix(matrix: pd.DataFrame, name: str = "baselines") -> Path:
    path = _metrics(f"dm_{name}.parquet")
    matrix.to_parquet(path)
    return path


def load_dm_matrix(name: str = "baselines") -> pd.DataFrame:
    """Diebold-Mariano statistic and p-value for every ordered pair of models.

    Here rather than written straight from the script, unlike the QA tables of
    scripts/02, because these two are *read* again: Sprint 4 merges them with
    the GCN's own results, and Sprint 5 turns them into the LaTeX tables. An
    artifact with a second reader is exactly what this module is for.
    """
    return pd.read_parquet(_require(_metrics(f"dm_{name}.parquet"), _predictions_command(name)))


# --------------------------------------------------------------------------
# Backtest (scripts/06)
# --------------------------------------------------------------------------

# Parametrized by name like the predictions family, and for the same reason: the
# study reports one group ("all", every model at every cost level), but a
# sensitivity run over a second threshold or a second window would want its own
# file rather than to overwrite this one.


def save_backtest(summary: pd.DataFrame, name: str = "all") -> Path:
    """One row per (model, cost level): Sharpe, drawdown, cumulative return."""
    path = _metrics(f"backtest_{name}.parquet")
    summary.to_parquet(path)
    return path


def load_backtest(name: str = "all") -> pd.DataFrame:
    return pd.read_parquet(_require(_metrics(f"backtest_{name}.parquet"), COMMAND_BACKTEST))


def save_backtest_curves(curves: pd.DataFrame, name: str = "all") -> Path:
    path = _metrics(f"backtest_curves_{name}.parquet")
    curves.to_parquet(path)
    return path


def load_backtest_curves(name: str = "all") -> pd.DataFrame:
    """The daily equity series behind the summary, one block per model and cost.

    Stored beside the summary rather than recomputed by whoever draws the
    figure: fig_equity_curves.pdf has to plot the same curves the table's Sharpe
    was computed from, and the only way to guarantee that is for both to come
    out of the same run.
    """
    return pd.read_parquet(_require(_metrics(f"backtest_curves_{name}.parquet"), COMMAND_BACKTEST))


# --------------------------------------------------------------------------
# Tables, publication and the run manifest (scripts/08)
# --------------------------------------------------------------------------


def save_table(name: str, body: str) -> Path:
    """Write one generated LaTeX table into results/tables/.

    UTF-8 with newline="\\n" explicitly: the .tex files are copied into a
    document that is compiled on whatever machine has MiKTeX, and letting
    Windows write CRLF would make every regeneration a whole-file diff.
    """
    path = paths.RESULTS_TABLES / f"{name}.tex"
    path.write_text(body, encoding="utf-8", newline="\n")
    return path


def load_table(name: str) -> str:
    return _require(paths.RESULTS_TABLES / f"{name}.tex", COMMAND_TABLES).read_text(encoding="utf-8")


def save_summary_markdown(body: str) -> Path:
    """Write results/summary.md, the write-up's index of every number.

    Named apart from save_summary(), which is the accuracy parquet: they are two
    different artifacts and the collision would be between a table of numbers and
    a document about them.

    Same UTF-8 and newline="\\n" as save_table(), for a further reason. The file
    is regenerated on every run of scripts/08, and it has to come out byte-identical
    when nothing upstream changed -- otherwise `git status` reports a modification
    that is only a line ending, and the run manifest records a dirty tree that is
    not dirty. That is also why nothing in the document is stamped with the time
    it was produced: provenance belongs in the manifest.
    """
    path = paths.RESULTS / "summary.md"
    path.write_text(body, encoding="utf-8", newline="\n")
    return path


def load_summary_markdown() -> str:
    return _require(paths.RESULTS / "summary.md", COMMAND_TABLES).read_text(encoding="utf-8")


def publish_to_thesis(figures: Sequence[str], tables: Sequence[str]) -> list[Path]:
    """Copy the finished figures and tables into ../latex-thesis/.

    The seam between the two sub-projects, crossed in exactly one direction and
    from exactly one place. Everything is verified present before anything is
    copied: a thesis that compiles with seven figures out of eight is worse than
    one that refuses to compile, because the missing one is discovered by a
    reader rather than by the build.
    """
    sources = [(paths.RESULTS_FIGURES / f"{name}.pdf", paths.FIGURES) for name in figures]
    sources += [(paths.RESULTS_TABLES / f"{name}.tex", paths.TABLES) for name in tables]

    missing = [str(source) for source, _ in sources if not source.exists()]
    if missing:
        raise MissingArtifactError(Path(missing[0]), f"{COMMAND_FIGURES} and {COMMAND_TABLES}")

    written = []
    for source, destination in sources:
        destination.mkdir(parents=True, exist_ok=True)
        target = destination / source.name
        shutil.copyfile(source, target)
        written.append(target)
    return written


def save_manifest(manifest: dict) -> Path:
    """Record of what produced the results, written last because it describes them all."""
    path = paths.RESULTS / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return path


def load_manifest() -> dict:
    with _require(paths.RESULTS / "run_manifest.json", COMMAND_TABLES).open(encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Stylized facts (scripts/02)
# --------------------------------------------------------------------------

# Written by scripts/02 as quality assurance, read by Sprint 5 for the universe
# table of Section 6.1. They arrived here late: Sprint 3 left them writing
# themselves with a bare to_parquet on the grounds that nothing read them back,
# which was true then and stops being true the moment tab_universe.tex exists.
_QA_TABLES = {
    "descriptive": "descriptive.parquet",
    "acf_returns": "acf_returns.parquet",
    "acf_abs_returns": "acf_abs_returns.parquet",
    "ljung_box_returns": "ljung_box_returns.parquet",
    "ljung_box_abs_returns": "ljung_box_abs_returns.parquet",
}


def save_stylized_facts(
    descriptive: pd.DataFrame,
    acf_returns: pd.DataFrame,
    acf_abs_returns: pd.DataFrame,
    ljung_box_returns: pd.DataFrame,
    ljung_box_abs_returns: pd.DataFrame,
) -> list[Path]:
    """Store the five quality-assurance tables in one call, so none is forgotten.

    Grouped like save_graphs() rather than split into five functions: they are
    produced together by one step of scripts/02 and describe one thing -- the
    return panel's univariate behaviour. A caller that saved four of them would
    leave the fifth silently stale.
    """
    tables = (descriptive, acf_returns, acf_abs_returns, ljung_box_returns, ljung_box_abs_returns)
    written = []
    for filename, table in zip(_QA_TABLES.values(), tables, strict=True):
        path = _metrics(filename)
        table.to_parquet(path)
        written.append(path)
    return written


def load_descriptive() -> pd.DataFrame:
    """Per-asset mean, annualized volatility, skewness and excess kurtosis.

    The body of tab_universe.tex in Section 6.1, which is why this pair exists.
    """
    return pd.read_parquet(_require(_metrics(_QA_TABLES["descriptive"]), COMMAND_BUILD))


def load_acf(absolute: bool = False) -> pd.DataFrame:
    """Autocorrelation of r_t up to 30 lags, or of |r_t| with `absolute=True`.

    One function with a flag rather than two: the two tables have an identical
    schema and are always read as a pair, since the finding is the contrast
    between them -- ACF(r) near zero, ACF(|r|) positive and slow to decay.
    """
    key = "acf_abs_returns" if absolute else "acf_returns"
    return pd.read_parquet(_require(_metrics(_QA_TABLES[key]), COMMAND_BUILD))


def load_ljung_box(absolute: bool = False) -> pd.DataFrame:
    """Ljung-Box statistic and p-value at 30 lags, for r_t or for |r_t|."""
    key = "ljung_box_abs_returns" if absolute else "ljung_box_returns"
    return pd.read_parquet(_require(_metrics(_QA_TABLES[key]), COMMAND_BUILD))


def save_event_study(study: pd.DataFrame) -> Path:
    path = _metrics("event_study.parquet")
    study.to_parquet(path)
    return path


def load_event_study() -> pd.DataFrame:
    return pd.read_parquet(_require(_metrics("event_study.parquet"), COMMAND_TOPOLOGY))

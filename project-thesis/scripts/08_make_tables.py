"""Entry point for Sprint 5 (S5.3 and S5.4): the handover to the thesis.

Last script of the pipeline. It turns the artifacts into the five LaTeX tables
Chapter 6 includes and into results/summary.md, copies the tables and the figures
into ../latex-thesis/, and writes the manifest that records which commit and which
configuration produced them.

After this runs, writing Chapter 6 is `\\input{tables/tab_results_main}`,
`\\includegraphics{figures/fig_equity_curves}` and reading one Markdown file for
everything that goes in the prose. That is the point of the sprint: no number in
the chapter should require rerunning anything to obtain, and none should be typed
by hand -- a transcribed figure disagrees with its source the first time anything
upstream changes, and nobody notices until a reader adds up a column.

The summary lives here rather than in a ninth script for two reasons. The
pipeline's last step should be the one that describes all the others, and the
manifest has to be written after everything it hashes. Splitting them would put
the manifest in the middle.

Recomputes nothing of the study. Every table, and every number in the summary, is
a formatting of a parquet file that scripts 02 through 06 already produced.

Integration: eighth script in the pipeline (scripts/01-08). Consumes
results/metrics/*.{parquet,json}, data/processed/returns.parquet and
results/figures/*.pdf; produces results/tables/*.tex, results/summary.md,
results/run_manifest.json, and the copies under ../latex-thesis/.

Usage:
    python scripts/08_make_tables.py
    python scripts/08_make_tables.py --no-publish     # stop before copying
"""
from __future__ import annotations

import datetime as dt
import hashlib
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from cryptognn.artifacts import (
    load_acf,
    load_backtest,
    load_descriptive,
    load_dm_matrix,
    load_event_study,
    load_ljung_box,
    load_returns,
    load_run_diagnostics,
    load_summary,
    load_tau,
    load_topology,
    publish_to_thesis,
    save_manifest,
    save_summary_markdown,
    save_table,
)
from cryptognn.cli import build_parser, run
from cryptognn.config import config_hash, load_config
from cryptognn.evaluation.walkforward import make_folds_from_config
from cryptognn.paths import FIGURES, RESULTS, RESULTS_FIGURES, RESULTS_METRICS, TABLES, ensure_dirs
from cryptognn.summary import SECTION_TITLES, build_summary, check_summary
from cryptognn.tables import TABLE_NAMES, build_all
from cryptognn.viz.figures import FIGURE_NAMES

# Recorded in the manifest rather than a full pip freeze: these are the packages
# whose version could change a number, and a list of 39 transitive pins would
# bury them. requirements.txt remains the exact lockfile.
TRACKED_PACKAGES = ("numpy", "pandas", "scipy", "statsmodels", "torch", "networkx", "matplotlib", "pyarrow")


def main() -> None:
    parser = build_parser("Generate the LaTeX tables and publish the results to the thesis.")
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Write results/tables/ but do not copy anything into ../latex-thesis/.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    ensure_dirs()

    descriptive = load_descriptive()
    calibration = load_tau()
    diagnostics_baselines = load_run_diagnostics("baselines")
    diagnostics_gcn = load_run_diagnostics("gcn")
    overall = load_summary("all")
    dm = load_dm_matrix("all")
    backtest = load_backtest()

    tables = build_all(
        descriptive=descriptive,
        calibration=calibration,
        baselines=diagnostics_baselines,
        gcn=diagnostics_gcn,
        summary=overall,
        dm=dm,
        backtest=backtest,
        config=config,
    )
    if set(tables) != set(TABLE_NAMES):
        raise ValueError(f"Built {sorted(tables)}, expected {sorted(TABLE_NAMES)}")

    print(f"{len(tables)} tables:")
    for name in TABLE_NAMES:
        body = tables[name]
        check(name, body)
        save_table(name, body)
        print(f"  {name + '.tex':24s} {len(body.splitlines()):3d} lines, {_row_count(body)} rows")

    dates = load_returns().index
    summary = build_summary(
        config=config,
        config_sha1=config_hash(args.config),
        descriptive=descriptive,
        acf=load_acf(),
        acf_abs=load_acf(absolute=True),
        ljung_box=load_ljung_box(),
        ljung_box_abs=load_ljung_box(absolute=True),
        calibration=calibration,
        topology=load_topology(),
        event_study=load_event_study(),
        diagnostics_baselines=diagnostics_baselines,
        diagnostics_gcn=diagnostics_gcn,
        overall=overall,
        by_asset=load_summary("all_by_asset"),
        by_fold=load_summary("all_by_fold"),
        dm=dm,
        backtest=backtest,
        folds=make_folds_from_config(config, len(dates)),
        dates=dates,
    )
    check_summary(summary)
    path = save_summary_markdown(summary)
    print(f"\n  saved {path.name}: {len(summary.splitlines())} lines, {len(SECTION_TITLES)} sections")

    if args.no_publish:
        print(f"\n  wrote results/tables/ only (--no-publish); {len(tables)} files")
    else:
        written = publish_to_thesis(FIGURE_NAMES, TABLE_NAMES)
        print(f"\nPublished to ../latex-thesis/: {len(FIGURE_NAMES)} figures, {len(TABLE_NAMES)} tables")
        print(f"  {FIGURES}\n  {TABLES}")
        if len(written) != len(FIGURE_NAMES) + len(TABLE_NAMES):
            raise ValueError(f"Copied {len(written)} files, expected {len(FIGURE_NAMES) + len(TABLE_NAMES)}")

    path = save_manifest(build_manifest(args.config))
    print(f"\n  saved {path.name}")


# A cell holding one of these is a formatting failure that compiles perfectly
# and reaches the printed page. Matched against whole cells rather than searched
# for in the source: "nan" is a substring of "Binance", which the caption of
# tab_universe legitimately contains.
NON_NUMBERS = frozenset({"nan", "NaN", "$nan$", "inf", "-inf", "$inf$", "None", ""})


def check(name: str, body: str) -> None:
    """Reject a table that would not compile, or that prints a non-number.

    Cheap, and it catches the two failures that are expensive later: an
    unbalanced environment stops the LaTeX build with an error pointing at the
    wrong line, and a stray "nan" does not stop it at all.
    """
    for opening, closing in (("\\begin{table}", "\\end{table}"), ("\\begin{tabular}", "\\end{tabular}")):
        if body.count(opening) != 1 or body.count(closing) != 1:
            raise ValueError(f"{name}: unbalanced {opening}/{closing}")

    for line in _data_rows(body):
        offending = [cell.strip() for cell in line.split("&") if cell.strip() in NON_NUMBERS]
        if offending:
            raise ValueError(f"{name}: cell {offending[0]!r} would be typeset as written, in row {line.strip()!r}")


def _data_rows(body: str) -> list[str]:
    """The body rows of a table: below the header rule, above the bottom one.

    The header is excluded deliberately rather than incidentally. A grouped
    header carries an empty leading cell -- the model column spans both levels --
    and an empty cell is exactly what this wants to reject everywhere else.
    """
    lines = body.splitlines()
    start = lines.index("\\midrule") + 1 if "\\midrule" in lines else 0
    end = lines.index("\\bottomrule") if "\\bottomrule" in lines else len(lines)
    return [line for line in lines[start:end] if line.rstrip().endswith("\\\\")]


def _row_count(body: str) -> int:
    return len(_data_rows(body))


def build_manifest(config_path: Path) -> dict:
    """Provenance: which commit, which configuration, which files.

    PLANNING asks for step durations here. They are not recoverable after the
    fact by a script that did not run the pipeline, and instrumenting the other
    seven to time themselves is a different change. Digests do the job durations
    would not have done anyway: the acceptance criterion is reproducing the run
    months later, and what that needs is the ability to tell whether a file has
    changed. The modification times still give the order the steps ran in.
    """
    artifacts = sorted(RESULTS_METRICS.glob("*")) + sorted(RESULTS_FIGURES.glob("*.pdf"))
    return {
        "generated_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "git": _git_state(),
        "config": {"path": str(config_path), "sha1": config_hash(config_path)},
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": {name: _version(name) for name in TRACKED_PACKAGES},
        },
        "artifacts": {
            str(path.relative_to(RESULTS.parent)): {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
                "modified": dt.datetime.fromtimestamp(path.stat().st_mtime, dt.UTC).isoformat(timespec="seconds"),
            }
            for path in artifacts
            if path.is_file()
        },
    }


def _git_state() -> dict[str, str | bool | None]:
    """Commit, branch and whether the tree was dirty when the results were made.

    `dirty` matters more than the commit does: a hash pinned while uncommitted
    edits were in the working tree does not identify the code that ran, and
    recording the hash alone would imply it does.
    """
    def ask(*command: str) -> str | None:
        try:
            return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()
        except (subprocess.CalledProcessError, OSError):
            return None

    status = ask("git", "status", "--porcelain")
    return {
        "commit": ask("git", "rev-parse", "HEAD"),
        "branch": ask("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    run(main)

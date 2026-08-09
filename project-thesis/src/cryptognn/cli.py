"""Shared command-line plumbing for the pipeline entry points.

The scripts of scripts/01-07 all take the same configuration arguments and all
need to fail the same way when an artifact is missing. Keeping that here means
the four of them cannot drift into four slightly different spellings of the
same flag, and that the study's defaults live with the paths they come from.

Exports:
  - build_parser(): an ArgumentParser carrying the common arguments
  - run(): invoke a script's main(), turning a missing artifact into a message

Integration: imported by every scripts/*.py.
Why run() exists: MissingArtifactError is a library-level error, deliberately a
  plain Exception so a GUI can catch and render it. A command-line program,
  though, should print one clear line and exit non-zero rather than show a
  traceback. run() is where that translation happens -- at the process boundary,
  once, instead of inside the library where it would dictate how every consumer
  must fail.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from cryptognn import paths
from cryptognn.artifacts import MissingArtifactError


def build_parser(description: str, *, events: bool = False) -> argparse.ArgumentParser:
    """An ArgumentParser with the arguments every entry point shares.

    `--config` always, `--events` when the script reads the crisis dates. The
    defaults are absolute paths from cryptognn.paths, so a script behaves the
    same whatever directory it is launched from; a path passed explicitly on the
    command line still resolves against the working directory, which is what a
    caller expects of an argument they typed.

    The returned parser is meant to be extended with each script's own flags
    (--force, --corr-only, --usetex, --outdir).
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config",
        type=Path,
        default=paths.DEFAULT_CONFIG,
        help="Path to the study config YAML (default: config/default.yaml).",
    )
    if events:
        parser.add_argument(
            "--events",
            type=Path,
            default=paths.DEFAULT_EVENTS,
            help="Path to the crisis events YAML (default: config/events.yaml).",
        )
    return parser


def run(main: Callable[[], None]) -> None:
    """Run a script's main(), reporting a missing artifact instead of a traceback.

    A missing artifact is an ordinary state of a multi-step pipeline -- step 3
    run before step 2 -- not a bug, so it deserves the command that fixes it
    rather than a stack trace. Any other exception propagates untouched: those
    are bugs, and their tracebacks are the useful part.
    """
    try:
        main()
    except MissingArtifactError as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from None

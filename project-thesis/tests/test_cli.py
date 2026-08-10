"""Tests for cryptognn.cli: the argument defaults and the process boundary.

Two properties, both of which fail silently rather than loudly if broken.

The defaults must be **absolute**. A default spelled "config/default.yaml" works
from project-thesis/ and resolves to nothing from anywhere else, so a script
launched from the repository root or from an IDE would die on a missing file --
or worse, pick up a different file that happens to sit there. That is the reason
cryptognn.paths exists, and nothing checked that the parsers actually use it.

And `run()` is the only place where a library-level error becomes an exit code.
MissingArtifactError is deliberately an ordinary Exception so a Streamlit app can
catch it; a command line, though, wants one line and a non-zero status, not a
traceback. Everything else must still propagate, because everything else is a bug
and its traceback is the useful part.
"""
from __future__ import annotations

import pytest

from cryptognn import paths
from cryptognn.artifacts import MissingArtifactError
from cryptognn.cli import build_parser, run


class TestBuildParser:
    def test_config_default_is_the_absolute_study_config(self):
        args = build_parser("test").parse_args([])

        assert args.config == paths.DEFAULT_CONFIG
        assert args.config.is_absolute()

    def test_events_argument_is_opt_in(self):
        """Only the scripts that read crisis dates carry --events; adding it
        everywhere would advertise an option most entry points ignore.
        """
        assert not hasattr(build_parser("test").parse_args([]), "events")

        args = build_parser("test", events=True).parse_args([])
        assert args.events == paths.DEFAULT_EVENTS
        assert args.events.is_absolute()

    def test_an_explicit_path_overrides_the_default(self, tmp_path):
        alternative = tmp_path / "alt.yaml"

        args = build_parser("test").parse_args(["--config", str(alternative)])

        assert args.config == alternative

    def test_parser_is_extensible_by_the_calling_script(self):
        """scripts/02, 05 and 06 add their own flags to the returned parser."""
        parser = build_parser("test")
        parser.add_argument("--corr-only", action="store_true")

        assert parser.parse_args(["--corr-only"]).corr_only is True


class TestRun:
    def test_a_missing_artifact_becomes_exit_1_and_names_the_command(self, capsys):
        def main() -> None:
            raise MissingArtifactError(paths.DATA_PROCESSED / "returns.parquet", "python scripts/02_build_graphs.py")

        with pytest.raises(SystemExit) as exit_info:
            run(main)

        assert exit_info.value.code == 1
        error_output = capsys.readouterr().err
        assert "python scripts/02_build_graphs.py" in error_output
        assert "returns.parquet" in error_output

    def test_any_other_exception_keeps_its_traceback(self):
        """A bug must not be flattened into a tidy message: the traceback is what
        makes it fixable.
        """
        def main() -> None:
            raise ZeroDivisionError("a real bug")

        with pytest.raises(ZeroDivisionError, match="a real bug"):
            run(main)

    def test_a_successful_main_returns_quietly(self, capsys):
        calls = []

        run(lambda: calls.append(1))

        assert calls == [1]
        assert capsys.readouterr().err == ""

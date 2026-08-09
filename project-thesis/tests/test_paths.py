"""Tests for the path constants (cryptognn.paths).

The property these protect is that an entry point behaves identically whichever
directory it is launched from. Every constant must therefore be absolute, and
the project root must be verified rather than assumed -- a wrong root does not
crash, it silently relocates the whole pipeline's output.
"""
from __future__ import annotations

import pytest

from cryptognn import paths


def test_root_is_the_project_directory():
    """ROOT is derived from this package's location; it must land on the
    directory that actually holds the project, identified by its pyproject.
    """
    assert paths.ROOT.is_absolute()
    assert (paths.ROOT / "pyproject.toml").is_file()
    assert (paths.ROOT / "src" / "cryptognn").is_dir()
    assert paths.ROOT.name == "project-thesis"


@pytest.mark.parametrize(
    "name",
    [
        "ROOT",
        "DATA",
        "DATA_RAW",
        "DATA_PROCESSED",
        "RESULTS",
        "RESULTS_METRICS",
        "RESULTS_FIGURES",
        "RESULTS_TABLES",
        "FIGURES",
        "CONFIG",
        "DEFAULT_CONFIG",
        "DEFAULT_EVENTS",
    ],
)
def test_every_path_constant_is_absolute(name):
    """A relative constant would resolve against the working directory, which is
    exactly the bug these constants exist to prevent.
    """
    assert getattr(paths, name).is_absolute(), name


def test_default_config_files_exist():
    """The argparse defaults of the scripts point here, so a missing file would
    break every entry point at once.
    """
    assert paths.DEFAULT_CONFIG.is_file()
    assert paths.DEFAULT_EVENTS.is_file()
    assert paths.DEFAULT_CONFIG.parent == paths.CONFIG
    assert paths.DEFAULT_EVENTS.parent == paths.CONFIG


def test_figures_points_outside_the_package():
    """The thesis figures live in the sibling LaTeX project, not under results/."""
    assert paths.FIGURES == paths.ROOT.parent / "latex-thesis" / "figures"


def test_verify_layout_accepts_a_well_formed_root(tmp_path):
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "config").mkdir()
    (tmp_path / "src" / "cryptognn").mkdir(parents=True)

    paths._verify_layout(tmp_path)  # must not raise


def test_verify_layout_rejects_a_wrong_root(tmp_path):
    """The case this guard exists for: a non-editable install puts the package in
    site-packages/, ROOT lands on an unrelated directory, and the pipeline would
    otherwise create data/ and results/ there and write every artifact to a place
    nobody looks. The message must name the likely cause, not just the missing
    directory -- whoever reads it is looking at a traceback.
    """
    with pytest.raises(RuntimeError) as error:
        paths._verify_layout(tmp_path)

    message = str(error.value)
    assert "pyproject.toml" in message
    assert "non-editable" in message
    assert "pip install -e ." in message


def test_verify_layout_rejects_a_partial_root(tmp_path):
    """Some markers present is not enough: a directory that merely contains a
    config/ folder is not this project.
    """
    (tmp_path / "config").mkdir()

    with pytest.raises(RuntimeError, match="pyproject.toml"):
        paths._verify_layout(tmp_path)


def test_ensure_dirs_is_idempotent(tmp_path, monkeypatch):
    """Scripts call it at every startup, so a second call must be a no-op.

    Exercised against a temporary root so the test never depends on -- or
    creates -- the real data/ and results/ trees.
    """
    targets = ["DATA_RAW", "DATA_PROCESSED", "RESULTS_METRICS", "RESULTS_FIGURES", "RESULTS_TABLES"]
    for name in targets:
        monkeypatch.setattr(paths, name, tmp_path / name.lower())

    paths.ensure_dirs()
    paths.ensure_dirs()

    for name in targets:
        assert getattr(paths, name).is_dir()


def test_ensure_dirs_does_not_create_input_directories(tmp_path, monkeypatch):
    """config/ and src/ are inputs: their absence is a layout error to report,
    not something to hide by creating an empty directory.
    """
    monkeypatch.setattr(paths, "CONFIG", tmp_path / "config")
    for name in ["DATA_RAW", "DATA_PROCESSED", "RESULTS_METRICS", "RESULTS_FIGURES", "RESULTS_TABLES"]:
        monkeypatch.setattr(paths, name, tmp_path / name.lower())

    paths.ensure_dirs()

    assert not (tmp_path / "config").exists()

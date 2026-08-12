"""Tests for the artifact I/O layer (cryptognn.artifacts).

Two properties matter here. The first is that every save/load pair round-trips:
these functions are the only place the pipeline's file formats are decided, so a
mismatch between the two halves would corrupt results silently rather than
raise. The second is that a missing artifact produces a *catchable* error --
the Streamlit explorer of Sprint 6 has to intercept it and render the remedy,
which the previous SystemExit (a BaseException) made impossible.

Everything runs against tmp_path with the path constants redirected, so no test
reads or writes the real data/ and results/ trees.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptognn import artifacts, paths
from cryptognn.graph.threshold import TauCalibration

WINDOW = 60
N_ASSETS = 4
N_WINDOWS = 12


@pytest.fixture(autouse=True)
def _isolated_tree(tmp_path, monkeypatch):
    """Redirect the artifact directories into tmp_path.

    artifacts.py reads `paths.DATA_PROCESSED` through the module rather than
    importing the constant, which is what makes this redirection possible at
    all -- a `from ... import DATA_PROCESSED` would bind the real path at import
    time and ignore the patch.
    """
    directories = {
        "DATA_PROCESSED": tmp_path / "processed",
        "RESULTS_METRICS": tmp_path / "metrics",
        "RESULTS": tmp_path / "results",
        "RESULTS_FIGURES": tmp_path / "results" / "figures",
        "RESULTS_TABLES": tmp_path / "results" / "tables",
        # The publication targets, redirected so a test never writes into the
        # real ../latex-thesis/ tree.
        "FIGURES": tmp_path / "thesis" / "figures",
        "TABLES": tmp_path / "thesis" / "tables",
    }
    for name, directory in directories.items():
        directory.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(paths, name, directory)
    return tmp_path


@pytest.fixture
def corr_tensor() -> tuple[np.ndarray, pd.DatetimeIndex]:
    rng = np.random.default_rng(0)
    corr = rng.uniform(-1, 1, (N_WINDOWS, N_ASSETS, N_ASSETS))
    corr = (corr + corr.transpose(0, 2, 1)) / 2
    np.einsum("kii->ki", corr)[:] = 1.0
    index = pd.date_range("2021-01-01", periods=N_WINDOWS, freq="D", name="date")
    return corr, index


@pytest.fixture
def calibration() -> TauCalibration:
    return TauCalibration(
        tau=0.2145,
        tau_fwer=0.4311,
        tau_fixed=0.30,
        alpha=0.05,
        n_permutations=500,
        n_calibration_windows=24,
        window=WINDOW,
        n_pairs=105,
        statistic="pooled",
        seed=42,
        window_end_dates=["2021-03-02", "2026-06-30"],
        per_window_tau=[0.21, 0.22],
        per_window_tau_fwer=[0.43, 0.44],
        density={"tau": {"mean": 0.97, "min": 0.73, "max": 1.0, "sd": 0.05}},
    )


# --------------------------------------------------------------------------
# Round trips
# --------------------------------------------------------------------------


def test_prices_returns_and_volumes_round_trip():
    frame = pd.DataFrame(
        {"BTC": [1.0, 2.0, 3.0], "ETH": [4.0, 5.0, 6.0]},
        index=pd.date_range("2021-01-01", periods=3, freq="D", name="date"),
    )

    artifacts.save_prices(frame)
    artifacts.save_returns(frame * 0.1)
    artifacts.save_volumes(frame * 1000.0)

    # check_freq=False: Parquet stores timestamps, not the inferred `freq`
    # attribute pandas attaches to a date_range. The pipeline's own indices are
    # built from arrays and carry no freq either, so nothing depends on it.
    pd.testing.assert_frame_equal(artifacts.load_prices(), frame, check_freq=False)
    pd.testing.assert_frame_equal(artifacts.load_returns(), frame * 0.1, check_freq=False)
    pd.testing.assert_frame_equal(artifacts.load_volumes(), frame * 1000.0, check_freq=False)


def test_corr_round_trip(corr_tensor):
    corr, index = corr_tensor

    artifacts.save_corr(corr, index, WINDOW)
    loaded, loaded_index = artifacts.load_corr(WINDOW)

    assert loaded.shape == corr.shape
    np.testing.assert_allclose(loaded, corr, atol=1e-6)  # float32 on disk
    pd.testing.assert_index_equal(loaded_index, index)


def test_corr_is_float32_on_disk_and_float64_in_memory(corr_tensor):
    """Size on disk, precision in memory. Both halves of that trade live in this
    module now, instead of being re-applied by hand in every consumer.
    """
    corr, index = corr_tensor

    artifacts.save_corr(corr, index, WINDOW)

    assert np.load(artifacts.corr_path(WINDOW)).dtype == np.float32
    assert artifacts.load_corr(WINDOW)[0].dtype == np.float64


def test_corr_index_is_always_a_datetime_index(corr_tensor):
    """The contract that removes the divergence this module was written for:
    two consumers previously converted corr_index in two different ways.
    """
    corr, index = corr_tensor

    artifacts.save_corr(corr, index, WINDOW)
    _, loaded_index = artifacts.load_corr(WINDOW)

    assert isinstance(loaded_index, pd.DatetimeIndex)
    assert loaded_index.name == "date"


def test_corr_index_survives_a_timezone_aware_input(corr_tensor):
    """The pipeline's index is UTC-aware; .npy has no timezone concept, so the
    saver strips it. The dates themselves must not shift.
    """
    corr, index = corr_tensor
    aware = index.tz_localize("UTC")

    artifacts.save_corr(corr, aware, WINDOW)
    _, loaded_index = artifacts.load_corr(WINDOW)

    assert loaded_index.tz is None
    assert list(loaded_index.date) == list(aware.date)


def test_corr_path_carries_the_window():
    """Several window lengths must be able to coexist for a sensitivity study."""
    assert artifacts.corr_path(60) != artifacts.corr_path(90)
    assert artifacts.corr_path(60).name == "corr_60.npy"


def test_graphs_round_trip():
    rng = np.random.default_rng(1)
    tensors = [rng.uniform(0, 1, (N_WINDOWS, N_ASSETS, N_ASSETS)) for _ in range(4)]

    artifacts.save_graphs(*tensors)

    loaded = [
        artifacts.load_w_full(),
        artifacts.load_w_thresh(),
        artifacts.load_a_hat(),
        artifacts.load_a_hat(fwer=True),
    ]
    for original, result in zip(tensors, loaded, strict=True):
        np.testing.assert_allclose(result, original, atol=1e-6)
        assert result.dtype == np.float64


def test_a_hat_fwer_is_a_distinct_artifact():
    """The pre-registered robustness variant must not overwrite the study one."""
    base = np.ones((2, N_ASSETS, N_ASSETS))

    artifacts.save_graphs(base, base, base, base * 0.5)

    assert not np.allclose(artifacts.load_a_hat(), artifacts.load_a_hat(fwer=True))


def test_tau_round_trips_as_a_typed_record(calibration):
    artifacts.save_tau(calibration)

    loaded = artifacts.load_tau()

    assert isinstance(loaded, TauCalibration)
    assert loaded == calibration
    assert loaded.tau == pytest.approx(0.2145)
    assert loaded.density["tau"]["mean"] == pytest.approx(0.97)


def test_tau_calibration_from_dict_is_the_inverse_of_to_dict(calibration):
    assert TauCalibration.from_dict(calibration.to_dict()) == calibration


def test_topology_and_event_study_round_trip():
    topology = pd.DataFrame(
        {"mean_correlation": [0.1, 0.2], "graph_density": [0.9, 1.0]},
        index=pd.date_range("2021-01-01", periods=2, freq="D", name="date"),
    )
    study = pd.DataFrame({"event_key": ["a", "b"], "value": [1.0, 2.0]})

    artifacts.save_topology(topology)
    artifacts.save_event_study(study)

    pd.testing.assert_frame_equal(artifacts.load_topology(), topology, check_freq=False)
    pd.testing.assert_frame_equal(artifacts.load_event_study(), study)


def test_stylized_facts_round_trip():
    """The five quality-assurance tables of scripts/02.

    They wrote themselves with a bare to_parquet until Sprint 5's universe table
    gave them a second reader, which is the condition this module exists for: an
    artifact read in one place and written in another is where a filename drifts.
    """
    assets = ["BTC", "ETH"]
    descriptive = pd.DataFrame({"mean": [1e-4, 2e-4], "excess_kurtosis": [5.1, 6.2]}, index=assets)
    lags = pd.RangeIndex(3, name="lag")
    acf_returns = pd.DataFrame({"BTC": [1.0, 0.01, -0.02], "ETH": [1.0, 0.03, 0.0]}, index=lags)
    acf_abs = pd.DataFrame({"BTC": [1.0, 0.31, 0.28], "ETH": [1.0, 0.29, 0.25]}, index=lags)
    ljung_box = pd.DataFrame({"lb_stat": [12.3, 9.8], "lb_pvalue": [0.42, 0.61]}, index=assets)
    ljung_box_abs = pd.DataFrame({"lb_stat": [980.1, 850.4], "lb_pvalue": [0.0, 0.0]}, index=assets)

    written = artifacts.save_stylized_facts(descriptive, acf_returns, acf_abs, ljung_box, ljung_box_abs)

    assert [path.name for path in written] == [
        "descriptive.parquet",
        "acf_returns.parquet",
        "acf_abs_returns.parquet",
        "ljung_box_returns.parquet",
        "ljung_box_abs_returns.parquet",
    ]
    pd.testing.assert_frame_equal(artifacts.load_descriptive(), descriptive)
    pd.testing.assert_frame_equal(artifacts.load_acf(), acf_returns)
    pd.testing.assert_frame_equal(artifacts.load_acf(absolute=True), acf_abs)
    pd.testing.assert_frame_equal(artifacts.load_ljung_box(), ljung_box)
    pd.testing.assert_frame_equal(artifacts.load_ljung_box(absolute=True), ljung_box_abs)


def test_stylized_facts_are_saved_or_none_are():
    """Five tables, one call: a caller cannot save four and leave the fifth stale."""
    frame = pd.DataFrame({"x": [1.0]})

    with pytest.raises(TypeError):
        artifacts.save_stylized_facts(frame, frame, frame, frame)  # type: ignore[call-arg]


def test_walkforward_outputs_round_trip():
    """The four tables scripts/04 produces, each read again by Sprint 4 or 5."""
    predictions = pd.DataFrame(
        {
            "fold": [0, 0],
            "date": pd.date_range("2022-05-05", periods=2, freq="D", tz="UTC"),
            "asset": ["BTC", "ETH"],
            "y_true": [0.01, -0.02],
            "y_pred": [0.0, 0.0],
            "model": ["zero", "zero"],
        }
    )
    diagnostics = pd.DataFrame({"fold": [0], "model": ["zero"], "n_train": [365]})
    summary = pd.DataFrame({"model": ["zero"], "rmse": [0.041453]})
    dm = pd.DataFrame({"model_a": ["mean"], "model_b": ["zero"], "statistic": [2.7782]})

    artifacts.save_predictions(predictions)
    artifacts.save_run_diagnostics(diagnostics)
    artifacts.save_summary(summary)
    artifacts.save_dm_matrix(dm)

    pd.testing.assert_frame_equal(artifacts.load_predictions(), predictions)
    pd.testing.assert_frame_equal(artifacts.load_run_diagnostics(), diagnostics)
    pd.testing.assert_frame_equal(artifacts.load_summary(), summary)
    pd.testing.assert_frame_equal(artifacts.load_dm_matrix(), dm)


def test_walkforward_outputs_are_named_per_run_group():
    """The GCN of Sprint 4 writes the same schema under its own name, so the two
    run groups coexist instead of overwriting each other.
    """
    summary = pd.DataFrame({"model": ["gcn"], "rmse": [0.042]})

    artifacts.save_summary(summary, name="gcn")

    assert (paths.RESULTS_METRICS / "summary_gcn.parquet").exists()
    pd.testing.assert_frame_equal(artifacts.load_summary(name="gcn"), summary)


def test_backtest_outputs_round_trip():
    """The two tables scripts/06 produces, read again by scripts 07 and 08.

    Saved as a pair because the figure and the table have to agree: the equity
    curve drawn in Section 6.5 must be the one the Sharpe beside it was computed
    from, and the only guarantee of that is a single run writing both.
    """
    summary = pd.DataFrame({"model": ["zero"], "cost_bps": [10.0], "sharpe": [float("nan")]})
    curves = pd.DataFrame(
        {
            "model": ["zero", "zero"],
            "cost_bps": [10.0, 10.0],
            "date": pd.date_range("2022-05-05", periods=2, freq="D", tz="UTC"),
            "net_return": [0.0, 0.0],
            "equity": [1.0, 1.0],
        }
    )

    artifacts.save_backtest(summary)
    artifacts.save_backtest_curves(curves)

    pd.testing.assert_frame_equal(artifacts.load_backtest(), summary)
    pd.testing.assert_frame_equal(artifacts.load_backtest_curves(), curves)


# --------------------------------------------------------------------------
# Tables, publication and the manifest (scripts/08)
# --------------------------------------------------------------------------


def test_tables_round_trip():
    body = "\\begin{table}[H]\n\\end{table}\n"

    artifacts.save_table("tab_universe", body)

    assert (paths.RESULTS_TABLES / "tab_universe.tex").exists()
    assert artifacts.load_table("tab_universe") == body


def test_tables_are_written_with_unix_newlines():
    """The .tex files are copied into a document compiled on whatever machine
    has MiKTeX. Letting Windows write CRLF makes every regeneration a whole-file
    diff, which hides the one line that actually changed.
    """
    artifacts.save_table("tab_x", "\\toprule\n\\bottomrule\n")

    assert b"\r\n" not in (paths.RESULTS_TABLES / "tab_x.tex").read_bytes()


def test_summary_markdown_round_trips():
    body = "# Sintesi\n\n| a | b |\n"

    artifacts.save_summary_markdown(body)

    assert (paths.RESULTS / "summary.md").exists()
    assert artifacts.load_summary_markdown() == body


def test_summary_markdown_is_written_with_unix_newlines():
    """summary.md is rewritten on every run of scripts/08 and has to come out
    byte-identical when nothing upstream changed. CRLF would make each rerun a
    whole-file diff, and the manifest would then record a dirty tree that is not.
    """
    artifacts.save_summary_markdown("# Sintesi\n\ntesto\n")

    assert b"\r\n" not in (paths.RESULTS / "summary.md").read_bytes()


def test_publish_copies_figures_and_tables_into_the_thesis():
    (paths.RESULTS_FIGURES / "fig_a.pdf").write_bytes(b"%PDF-1.4")
    artifacts.save_table("tab_a", "\\toprule\n")

    written = artifacts.publish_to_thesis(["fig_a"], ["tab_a"])

    assert (paths.FIGURES / "fig_a.pdf").read_bytes() == b"%PDF-1.4"
    assert (paths.TABLES / "tab_a.tex").exists()
    assert len(written) == 2


def test_publish_refuses_a_partial_set():
    """A thesis that compiles with seven figures out of eight is worse than one
    that refuses to compile: the missing one is found by a reader, not a build.
    """
    (paths.RESULTS_FIGURES / "fig_a.pdf").write_bytes(b"%PDF-1.4")

    with pytest.raises(artifacts.MissingArtifactError, match="fig_b"):
        artifacts.publish_to_thesis(["fig_a", "fig_b"], [])

    assert not list(paths.FIGURES.iterdir()), "nothing may be copied when the set is incomplete"


def test_manifest_round_trips_as_json():
    manifest = {"git": {"commit": "abc123", "dirty": False}, "artifacts": {"a.parquet": {"sha256": "0" * 64}}}

    artifacts.save_manifest(manifest)

    assert artifacts.load_manifest() == manifest


# --------------------------------------------------------------------------
# Missing artifacts
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("loader", "command"),
    [
        (lambda: artifacts.load_returns(), artifacts.COMMAND_BUILD),
        (lambda: artifacts.load_prices(), artifacts.COMMAND_BUILD),
        (lambda: artifacts.load_volumes(), artifacts.COMMAND_BUILD),
        (lambda: artifacts.load_corr(WINDOW), artifacts.COMMAND_BUILD),
        (lambda: artifacts.load_corr_index(), artifacts.COMMAND_BUILD),
        (lambda: artifacts.load_w_full(), artifacts.COMMAND_BUILD),
        (lambda: artifacts.load_w_thresh(), artifacts.COMMAND_BUILD),
        (lambda: artifacts.load_a_hat(), artifacts.COMMAND_BUILD),
        (lambda: artifacts.load_tau(), artifacts.COMMAND_BUILD),
        (lambda: artifacts.load_topology(), artifacts.COMMAND_TOPOLOGY),
        (lambda: artifacts.load_descriptive(), artifacts.COMMAND_BUILD),
        (lambda: artifacts.load_acf(), artifacts.COMMAND_BUILD),
        (lambda: artifacts.load_acf(absolute=True), artifacts.COMMAND_BUILD),
        (lambda: artifacts.load_ljung_box(), artifacts.COMMAND_BUILD),
        (lambda: artifacts.load_ljung_box(absolute=True), artifacts.COMMAND_BUILD),
        (lambda: artifacts.load_event_study(), artifacts.COMMAND_TOPOLOGY),
        (lambda: artifacts.load_predictions(), artifacts.COMMAND_BASELINES),
        (lambda: artifacts.load_run_diagnostics(), artifacts.COMMAND_BASELINES),
        (lambda: artifacts.load_summary(), artifacts.COMMAND_BASELINES),
        (lambda: artifacts.load_dm_matrix(), artifacts.COMMAND_BASELINES),
        # The walk-forward artifacts are parametrized by run group but the script
        # that makes them is not: everything other than the baselines comes from
        # script 05, and pointing a user at 04 would have them run a command that
        # succeeds and leaves the file still missing.
        (lambda: artifacts.load_predictions(name="gcn"), artifacts.COMMAND_GCN),
        (lambda: artifacts.load_run_diagnostics(name="gcn"), artifacts.COMMAND_GCN),
        (lambda: artifacts.load_summary(name="all"), artifacts.COMMAND_GCN),
        (lambda: artifacts.load_summary(name="all_by_fold"), artifacts.COMMAND_GCN),
        (lambda: artifacts.load_dm_matrix(name="all"), artifacts.COMMAND_GCN),
        (lambda: artifacts.load_backtest(), artifacts.COMMAND_BACKTEST),
        (lambda: artifacts.load_backtest_curves(), artifacts.COMMAND_BACKTEST),
        (lambda: artifacts.load_table("tab_universe"), artifacts.COMMAND_TABLES),
        (lambda: artifacts.load_summary_markdown(), artifacts.COMMAND_TABLES),
        (lambda: artifacts.load_manifest(), artifacts.COMMAND_TABLES),
    ],
)
def test_missing_artifact_names_the_command_that_makes_it(loader, command):
    with pytest.raises(artifacts.MissingArtifactError) as error:
        loader()

    assert error.value.command == command
    assert command in str(error.value)
    assert str(error.value.path) in str(error.value)


def test_missing_artifact_is_catchable_as_an_ordinary_exception():
    """The Sprint 6 requirement, and the reason this is not a SystemExit.

    SystemExit derives from BaseException, so `except Exception` lets it through
    and a running Streamlit app would be torn down by a missing file instead of
    showing the command that produces it.
    """
    try:
        artifacts.load_topology()
    except Exception as error:  # noqa: BLE001 -- deliberately broad: that is the property tested
        assert isinstance(error, artifacts.MissingArtifactError)
        assert isinstance(error, FileNotFoundError)
    else:
        pytest.fail("expected MissingArtifactError")

    assert not issubclass(artifacts.MissingArtifactError, SystemExit)


def test_missing_artifact_reports_the_redirected_path(_isolated_tree):
    """The error must name the path actually looked at, not a stale constant."""
    with pytest.raises(artifacts.MissingArtifactError) as error:
        artifacts.load_topology()

    assert error.value.path.is_relative_to(_isolated_tree)

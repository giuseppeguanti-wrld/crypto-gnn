"""Smoke tests for app/streamlit_app.py, run headless via streamlit.testing.v1.

Deliberately not isolated with tmp_path the way tests/test_artifacts.py
redirects paths.DATA_PROCESSED etc.: these tests read the project's real
data/ and results/ trees, the same ones the app itself reads. Doing the full
isolation (synthetic corr/topology/tau artifacts written through
artifacts.save_*) is a real improvement but a costlier one, and Sprint 6 has
no fixed day in PLANNING.md -- recorded here as a deliberate deferral rather
than done silently or expanded into now.
"""
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from cryptognn import artifacts
from cryptognn.paths import ROOT

APP_PATH = ROOT / "app" / "streamlit_app.py"


@pytest.fixture(autouse=True)
def _clear_streamlit_cache():
    # st.cache_data/cache_resource are process-global: a response cached by one
    # test would survive into the next test's monkeypatch and mask the guard on
    # a missing artifact.
    st.cache_data.clear()
    st.cache_resource.clear()
    yield


def test_app_runs_without_exception():
    at = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    assert not at.exception
    assert not at.error


def test_app_shows_error_on_missing_artifact(monkeypatch):
    def _raise(*_args, **_kwargs):
        raise artifacts.MissingArtifactError(Path("data/processed/returns.parquet"), artifacts.COMMAND_BUILD)

    monkeypatch.setattr(artifacts, "load_returns", _raise)
    at = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not at.exception
    assert at.error
    assert artifacts.COMMAND_BUILD in at.error[0].value


def test_default_selection_shows_five_metrics_three_images_and_download():
    at = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    assert not at.exception
    assert not at.error
    assert at.selectbox[0].options == ["60"]  # only the window actually on disk today
    assert at.radio[0].value == "Calibrata"
    assert len(at.metric) == 5
    assert len(at.image) == 3  # col_graph + col_heat + the S6.4 topology strip
    assert len(at.download_button) == 1


def test_manual_threshold_reveals_slider():
    at = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    at.radio[0].set_value("Manuale").run(timeout=30)
    assert not at.exception
    assert len(at.slider) == 1


def test_compare_two_dates_reveals_second_slider():
    at = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    at.checkbox[1].set_value(True).run(timeout=30)  # 0 = Layout fisso, 1 = Confronta due date
    assert not at.exception
    assert len(at.select_slider) == 2


def test_compare_two_dates_still_renders_three_images():
    at = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    at.checkbox[1].set_value(True).run(timeout=30)  # Confronta due date
    assert not at.exception
    assert not at.error
    # graph + heatmap each merge both compared panels into one figure, plus the
    # topology strip -- unaffected by the compare toggle.
    assert len(at.image) == 3


def test_topology_strip_renders_with_event_markers():
    at = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    assert not at.exception
    assert not at.error
    # Exercises load_events()/draw_metric_series() end to end: config/events.yaml
    # parses and the extra panel renders without disturbing the other two images.
    assert len(at.image) == 3


def test_manual_threshold_value_persists_across_radio_toggle():
    at = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    at.radio[0].set_value("Manuale").run(timeout=30)
    default_value = at.slider[0].value
    # Away from the edges of [0.0, 0.9] regardless of where tau.tau happens to land.
    custom_value = round(default_value + 0.1, 2) if default_value < 0.8 else round(default_value - 0.1, 2)
    at.slider[0].set_value(custom_value).run(timeout=30)

    # S6.5: the slider only exists while "Manuale" is selected -- without the
    # session_state key it lost this value the moment the radio moved away.
    at.radio[0].set_value("Calibrata").run(timeout=30)
    at.radio[0].set_value("Manuale").run(timeout=30)

    assert not at.exception
    assert at.slider[0].value == pytest.approx(custom_value)


def test_layout_toggle_off_does_not_crash():
    at = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    at.checkbox[0].set_value(False).run(timeout=30)  # Layout fisso spento
    assert not at.exception
    assert not at.error

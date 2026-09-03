"""Tests for the crisis event study (cryptognn.events).

The central test is test_step_series_shows_the_window_lag: on a series that
jumps at a known date, the reading at offset 0 is still the pre-event value
because the metric's 60-day window has only just reached the event. That lag is
the reason the offsets extend to +/-60, and asserting it directly keeps the
rationale from decaying into an unexplained constant.
"""
from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

from cryptognn.events import (
    DEFAULT_OFFSETS,
    Event,
    event_study,
    events_without_citation,
    load_events,
)

EVENT_DATE = pd.Timestamp("2022-06-01")
BEFORE, AFTER = 1.0, 3.0


@pytest.fixture
def events_file(tmp_path):
    path = tmp_path / "events.yaml"
    path.write_text(
        "events:\n"
        "  - key: first\n"
        "    date: 2022-06-01\n"
        "    label: Primo\n"
        "    description: Un evento documentato\n"
        "    citation: someref2023\n"
        "  - key: second\n"
        "    date: 2022-11-08\n"
        "    label: Secondo\n"
        "    citation: ''\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def step_topology() -> pd.DataFrame:
    """A metric that is flat at 1.0, then flat at 3.0 from the event date on.

    The step lets every reading be checked against a value known in closed form,
    and makes the percentile degenerate to 0 before and 1 after -- so a
    percentile that lands anywhere else is a bug, not sampling noise.
    """
    index = pd.date_range("2021-01-01", "2023-12-31", freq="D", name="date")
    values = np.where(index < EVENT_DATE, BEFORE, AFTER)
    return pd.DataFrame({"metric_a": values, "metric_b": values * 2.0}, index=index)


def test_load_events_parses_fields(events_file):
    events = load_events(events_file)

    assert [event.key for event in events] == ["first", "second"]
    assert events[0].date == datetime.date(2022, 6, 1)
    assert events[0].label == "Primo"
    assert events[0].description == "Un evento documentato"
    assert events[0].citation == "someref2023"
    # An absent description is an empty string, not None.
    assert events[1].description == ""


def test_load_events_allows_empty_citation(events_file):
    """An uncited event must load: the pipeline reports the debt, it does not
    refuse to run over it.
    """
    events = load_events(events_file)

    assert events[1].citation == ""


def test_load_events_rejects_duplicate_key(tmp_path):
    path = tmp_path / "events.yaml"
    path.write_text(
        "events:\n"
        "  - {key: same, date: 2022-06-01, label: A, citation: x}\n"
        "  - {key: same, date: 2022-11-08, label: B, citation: y}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate event key"):
        load_events(path)


def test_load_events_rejects_malformed_date(tmp_path):
    """A quoted date parses as a string, which would silently break the offset
    arithmetic; it must be rejected where it is written, not where it is used.
    """
    path = tmp_path / "events.yaml"
    path.write_text(
        "events:\n  - {key: a, date: '01/06/2022', label: A, citation: x}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="expected an unquoted ISO date"):
        load_events(path)


def test_load_events_rejects_empty_file(tmp_path):
    path = tmp_path / "events.yaml"
    path.write_text("events:\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No events declared"):
        load_events(path)


def test_events_without_citation_finds_the_debt(events_file):
    events = load_events(events_file)

    uncited = events_without_citation(events)

    assert [event.key for event in uncited] == ["second"]


def test_event_study_reads_known_values(step_topology):
    """On a step series, each offset must return the value on its side of the
    jump, and the percentile must be 0 before and 1 after.
    """
    events = [Event(key="e", date=EVENT_DATE.date(), label="E")]

    study = event_study(step_topology, events)
    metric_a = study[study["metric"] == "metric_a"].set_index("offset_days")

    assert metric_a.loc[-60, "value"] == BEFORE
    assert metric_a.loc[-30, "value"] == BEFORE
    assert metric_a.loc[0, "value"] == AFTER  # the step is at the event date itself
    assert metric_a.loc[60, "value"] == AFTER
    assert metric_a.loc[-60, "percentile"] == pytest.approx(0.0)
    assert metric_a.loc[60, "percentile"] == pytest.approx(
        float((step_topology["metric_a"] < AFTER).mean())
    )


def test_step_series_shows_the_window_lag():
    """The reason the offsets reach +/-60.

    Emulating a rolling 60-day metric -- one that only reaches its post-event
    level 60 days after the shock -- the reading at offset 0 is still entirely
    pre-event, and only +60 sees the full move. An event study stopping at +30
    would report roughly half the change.
    """
    index = pd.date_range("2021-01-01", "2023-12-31", freq="D", name="date")
    days_since = (index - EVENT_DATE).days.to_numpy()
    # Linear ramp over 60 days: the window's gradual absorption of the event.
    ramp = np.clip(days_since / 60.0, 0.0, 1.0)
    topology = pd.DataFrame({"metric": BEFORE + (AFTER - BEFORE) * ramp}, index=index)
    events = [Event(key="e", date=EVENT_DATE.date(), label="E")]

    study = event_study(topology, events).set_index("offset_days")

    assert study.loc[0, "value"] == pytest.approx(BEFORE)
    assert study.loc[30, "value"] == pytest.approx(BEFORE + (AFTER - BEFORE) * 0.5)
    assert study.loc[60, "value"] == pytest.approx(AFTER)
    # The full change is 200%; the inner -30 -> +30 horizon sees only half of it.
    assert study.loc[60, "pct_change_clean"] == pytest.approx(200.0)
    assert study.loc[60, "pct_change_local"] == pytest.approx(100.0)


def test_event_study_percentage_changes(step_topology):
    events = [Event(key="e", date=EVENT_DATE.date(), label="E")]

    study = event_study(step_topology, events)

    # 1.0 -> 3.0 is +200%, for both metrics: the second is the first doubled,
    # and a percentage change is invariant to that scaling.
    for metric in ("metric_a", "metric_b"):
        rows = study[study["metric"] == metric]
        assert rows["pct_change_clean"].unique() == pytest.approx([200.0])


def test_event_study_offset_outside_sample_is_nan():
    """An offset past the end of the sample yields NaN, never the nearest
    available date: substituting a reading from months away would be worse than
    an honest gap.
    """
    index = pd.date_range("2022-05-01", "2022-06-15", freq="D", name="date")
    topology = pd.DataFrame({"metric": np.arange(len(index), dtype=float)}, index=index)
    events = [Event(key="e", date=EVENT_DATE.date(), label="E")]

    study = event_study(topology, events).set_index("offset_days")

    assert pd.isna(study.loc[60, "value"])
    assert pd.isna(study.loc[60, "date_used"])
    assert pd.isna(study.loc[-60, "value"])
    # The in-range offsets are still read normally.
    assert not pd.isna(study.loc[0, "value"])
    assert study.loc[0, "date_used"] == EVENT_DATE


def test_event_study_records_the_date_actually_used(step_topology):
    """date_used makes the alignment auditable rather than implicit."""
    events = [Event(key="e", date=EVENT_DATE.date(), label="E")]

    study = event_study(step_topology, events).set_index("offset_days")

    assert study.loc[0, "date_used"].iloc[0] == EVENT_DATE
    assert study.loc[-30, "date_used"].iloc[0] == EVENT_DATE - pd.Timedelta(days=30)
    assert study.loc[60, "date_used"].iloc[0] == EVENT_DATE + pd.Timedelta(days=60)


def test_event_study_shape_and_columns(step_topology):
    """One row per (event, metric, offset), with every reported column present."""
    events = [
        Event(key="a", date=EVENT_DATE.date(), label="A"),
        Event(key="b", date=datetime.date(2022, 11, 8), label="B"),
    ]

    study = event_study(step_topology, events)

    assert len(study) == len(events) * step_topology.shape[1] * len(DEFAULT_OFFSETS)
    assert set(study.columns) == {
        "event_key",
        "event_date",
        "label",
        "metric",
        "offset_days",
        "date_used",
        "value",
        "percentile",
        "pct_change_clean",
        "pct_change_local",
    }
    assert set(study["offset_days"]) == set(DEFAULT_OFFSETS)


def test_event_study_rejects_degenerate_input(step_topology):
    with pytest.raises(ValueError, match="No events"):
        event_study(step_topology, [])
    with pytest.raises(ValueError, match="at least two offsets"):
        event_study(step_topology, [Event(key="e", date=EVENT_DATE.date(), label="E")], offsets=(0,))


def test_event_study_pct_change_local_is_nan_without_enough_offsets(step_topology):
    """With fewer than 4 offsets there is no inner pair distinct from the outer
    one; pct_change_local must say so honestly, not silently copy pct_change_clean.
    """
    events = [Event(key="e", date=EVENT_DATE.date(), label="E")]

    study = event_study(step_topology, events, offsets=(-60, 0, 60))

    assert study["pct_change_local"].isna().all()
    assert not study["pct_change_clean"].isna().any()

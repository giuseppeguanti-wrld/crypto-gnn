"""Crisis event study over the topological metric series (thesis Section 6.6).

Section 6.6 asks whether the correlation graph compacts during market stress.
The metric series of graph.metrics answer "does it move?"; this module answers
"does it move *when the market breaks*?", by reading each series around
documented crisis dates and placing those readings in the context of the whole
sample.

Two devices make readings comparable across metrics measured on different
scales (mean correlation lives in [0, 1], the combinatorial Fiedler value in
[3.7, 10.4]): the percentile of each value within the full historical
distribution, and the percentage change between offsets.

The offsets are the crux. A metric at date t is computed on the returns window
[t-59, t], so the value *on the event date* contains the event only on its last
day and is otherwise pre-event data. Offsets reach +/-60 so the study can
compare two windows that share no observation at all: -60 is entirely before the
event, +60 entirely after. Measured on this study's data, the 2021 crash reads
at the 23rd percentile of mean correlation at offset 0 but the 99th at +60 --
stopping at +30, as a naive reading would, halves the measured effect for a
reason that is an artifact of the rolling window rather than a property of the
market.

Exports:
  - Event: frozen record of one crisis date (key, date, label, description, citation)
  - load_events(): parse config/events.yaml
  - events_without_citation(): the bibliographic debt, reported by the pipeline
  - event_study(): long-format table of values, percentiles and changes

Integration: called by scripts/03_topology_analysis.py after topology.parquet is
  written; produces results/metrics/event_study.parquet, which feeds the event
  markers of the Section 6.6 figures (S2.5) and the Streamlit explorer.
Why events live in a config file: the dates are study parameters like any other,
  and each must be documentable with a citation -- keeping them in YAML with a
  citation field makes an undocumented event visible instead of letting it hide
  as a literal in plotting code.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# -60 and +60 are the only pair of offsets whose 60-day windows do not overlap,
# which is what makes their comparison a genuine before/after rather than a
# comparison of two windows sharing half their data.
DEFAULT_OFFSETS: tuple[int, ...] = (-60, -30, 0, 30, 60)


@dataclass(frozen=True)
class Event:
    """One documented crisis date.

    `citation` is a BibTeX key from latex-thesis/bibliography/references.bib, or
    an empty string when no source has been recorded yet. Empty is allowed so
    the pipeline can run and report the gap, not so the gap can be ignored: see
    events_without_citation().
    """

    key: str
    date: date
    label: str
    description: str = ""
    citation: str = ""


def load_events(path: str | Path) -> list[Event]:
    """Parse the crisis events from a YAML file, in the order declared.

    Raises on a duplicate key or a malformed date rather than silently dropping
    or reordering an event: the events end up as vertical markers on the figures
    of Section 6.6, where a missing one is far harder to notice than a crash
    here.
    """
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    entries = (raw or {}).get("events")
    if not entries:
        raise ValueError(f"No events declared in {path}")

    events = []
    seen: set[str] = set()
    for entry in entries:
        key = entry["key"]
        if key in seen:
            raise ValueError(f"Duplicate event key {key!r} in {path}")
        seen.add(key)

        # yaml.safe_load already yields a datetime.date for an unquoted ISO date;
        # anything else is a formatting mistake worth naming explicitly.
        event_date = entry["date"]
        if not isinstance(event_date, date):
            # ValueError, not TypeError: this is a malformed *configuration file*,
            # not a caller passing the wrong type, and every other config error in
            # the study raises ValueError (calibrate_tau, normalized_adjacency).
            raise ValueError(  # noqa: TRY004
                f"Event {key!r} has date {event_date!r}: expected an unquoted ISO date (YYYY-MM-DD)"
            )

        events.append(
            Event(
                key=key,
                date=event_date,
                label=entry["label"],
                description=entry.get("description", "") or "",
                citation=entry.get("citation", "") or "",
            )
        )

    return events


def events_without_citation(events: list[Event]) -> list[Event]:
    """The events still lacking a bibliographic source.

    Only events documentable with a citation belong in the study. This surfaces
    the outstanding ones so the pipeline can report them on every run, keeping
    the debt visible until Section 6.6 is written.
    """
    return [event for event in events if not event.citation]


def event_study(
    topology: pd.DataFrame,
    events: list[Event],
    offsets: tuple[int, ...] = DEFAULT_OFFSETS,
) -> pd.DataFrame:
    """Read every topological metric around every event, in long format.

    `topology` is the (T, M) frame written by scripts/03_topology_analysis.py,
    indexed by the window's closing date.

    Returns one row per (event_key, metric, offset_days) with:
      date_used        the index date actually read (NaT if the offset falls
                       outside the sample)
      value            the metric there
      percentile       share of the full sample below that value, in [0, 1] --
                       what makes metrics on different scales comparable
      pct_change_clean percentage change from the first to the last offset
                       (-60 -> +60), between non-overlapping windows; the
                       headline figure, repeated on every row of the group
      pct_change_local percentage change across the inner offsets (-30 -> +30),
                       a narrower horizon kept for comparability; NaN when
                       `offsets` has fewer than 4 entries, since there is then
                       no inner pair distinct from the outer one to read

    An offset outside the sample yields NaN rather than the nearest available
    date: silently substituting a reading from months away would be worse than
    an honest gap.
    """
    if not events:
        raise ValueError("No events to study")
    if len(offsets) < 2:
        raise ValueError(f"Need at least two offsets to compute a change, got {offsets}")

    index = pd.DatetimeIndex(topology.index)
    records = []

    for event in events:
        event_date = pd.Timestamp(event.date)
        for offset in offsets:
            target = event_date + pd.Timedelta(days=offset)
            date_used, values = _read_at(topology, index, target)
            for metric in topology.columns:
                value = values[metric] if values is not None else np.nan
                records.append(
                    {
                        "event_key": event.key,
                        "event_date": event_date,
                        "label": event.label,
                        "metric": metric,
                        "offset_days": offset,
                        "date_used": date_used,
                        "value": value,
                        "percentile": _percentile(topology[metric], value),
                    }
                )

    study = pd.DataFrame.from_records(records)

    outer = _change_between(study, offsets[0], offsets[-1])
    # Below 4 offsets, offsets[1] and offsets[-2] either collide with each
    # other (3 offsets: always a 0% no-op) or with the outer pair reversed (2
    # offsets), neither of which is a genuine narrower reading. NaN says so
    # honestly instead of quietly repeating pct_change_clean under a second name.
    inner = (
        _change_between(study, offsets[1], offsets[-2])
        if len(offsets) >= 4
        else pd.Series(np.nan, index=outer.index)
    )
    study["pct_change_clean"] = _broadcast(study, outer)
    study["pct_change_local"] = _broadcast(study, inner)

    return study


def _read_at(
    topology: pd.DataFrame, index: pd.DatetimeIndex, target: pd.Timestamp
) -> tuple[pd.Timestamp, pd.Series | None]:
    """The row nearest `target`, or (NaT, None) if the target is off the sample.

    The index is daily and gapless (crypto has no market holidays), so "nearest"
    resolves to the exact date whenever the target is in range; the tolerance is
    there for the endpoints, not to paper over a distant substitution.
    """
    if target < index[0] or target > index[-1]:
        return pd.NaT, None

    position = index.get_indexer([target], method="nearest")[0]
    return index[position], topology.iloc[position]


def _percentile(series: pd.Series, value: float) -> float:
    """Share of the full historical sample strictly below `value`, in [0, 1]."""
    if pd.isna(value):
        return np.nan
    return float((series < value).mean())


def _change_between(study: pd.DataFrame, start_offset: int, end_offset: int) -> pd.Series:
    """Percentage change of each (event, metric) between two offsets."""
    pivot = study.pivot_table(
        index=["event_key", "metric"], columns="offset_days", values="value", dropna=False
    )
    return 100.0 * (pivot[end_offset] - pivot[start_offset]) / pivot[start_offset]


def _broadcast(study: pd.DataFrame, change: pd.Series) -> np.ndarray:
    """Repeat a per-(event, metric) value across every offset row of its group."""
    keys = pd.MultiIndex.from_arrays([study["event_key"], study["metric"]])
    return change.reindex(keys).to_numpy()

"""Tests for cryptognn.summary: the write-up's index of every number.

summary.md fails silently in three ways, and this module exists for those three.
A `nan` in a Markdown cell renders exactly as written and gets copied into the
thesis. A section dropped by an edit leaves a gap discovered during the write-up,
which is the one moment there is no time to regenerate it. And a document that is
not byte-stable across runs makes `git status` report a change that is not one,
which is what would let the run manifest record a dirty tree that is clean.

The numbers themselves are not re-tested: they are the artifacts, tested where
they are computed. What is tested is that every number reaches the page, formatted
the way the thesis will print it.

The fixtures describing a finished run live in tests/conftest.py, shared with
test_tables -- see that module's docstring for why they are not duplicated.
"""
from __future__ import annotations

import re

import pandas as pd
import pytest
from synthetic import BACKTEST_REFERENCE, GCN_ARMS, MODELS, N_FOLDS, N_OBS, SYMBOLS

from cryptognn.evaluation.walkforward import make_folds_from_config
from cryptognn.summary import (
    METRIC_LABELS,
    SECTION_TITLES,
    build_summary,
    check_summary,
)
from cryptognn.tables import MISSING, plain_model_label

CONFIG_SHA1 = "2ac1abadf9d7360164e6b5803e35d779314b6b77"
EVENTS = {"china_crackdown": "Stretta cinese", "terra_luna": "Terra/Luna", "ftx": "FTX"}
OFFSETS = (-60, -30, 0, 30, 60)


# --------------------------------------------------------------------------
# Fixtures this module alone needs
# --------------------------------------------------------------------------


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    return pd.date_range("2021-01-02", periods=N_OBS, freq="D", tz="UTC")


@pytest.fixture
def folds(config, dates):
    return make_folds_from_config(config, len(dates))


@pytest.fixture
def topology(dates, config) -> pd.DataFrame:
    """The metric series, on the naive index topology.parquet actually carries.

    Naive rather than UTC-aware on purpose: .npy has no timezone concept, so the
    correlation index round-trips through it without one, and fold_test_means()
    reconciles the two. A tz-aware fixture would test a reconciliation that never
    has to happen.
    """
    index = dates[config.graph.window - 1 :].tz_localize(None)
    rng = pd.Series(range(len(index)), index=index)
    frame = pd.DataFrame(index=index)
    for position, metric in enumerate(METRIC_LABELS):
        # Varying, and never constant: rank_association refuses a constant series,
        # which is the correct behaviour and not what this fixture is testing.
        frame[metric] = 0.4 + 0.5 * ((rng + 37 * position) % 101) / 101.0
    frame["eigs_outside_mp"] = (rng % 2 + 1).astype("int64")
    return frame


@pytest.fixture
def event_study(topology) -> pd.DataFrame:
    rows = []
    for position, (key, label) in enumerate(EVENTS.items()):
        event_date = topology.index[200 + 300 * position]
        for metric in METRIC_LABELS:
            for offset in OFFSETS:
                date_used = event_date + pd.Timedelta(days=offset)
                value = float(topology.loc[date_used, metric])
                rows.append(
                    {
                        "event_key": key,
                        "event_date": event_date,
                        "label": label,
                        "metric": metric,
                        "offset_days": offset,
                        "date_used": date_used,
                        "value": value,
                        "percentile": float((topology[metric] < value).mean()),
                        "pct_change_clean": 12.5,
                        "pct_change_local": -3.25,
                    }
                )
    return pd.DataFrame(rows)


@pytest.fixture
def by_fold() -> pd.DataFrame:
    rows = []
    for fold in range(N_FOLDS):
        for position, model in enumerate(MODELS):
            rows.append(
                {
                    "model": model,
                    "fold": fold,
                    "rmse": 0.0414 + 0.001 * position,
                    "mae": 0.0276 + 0.001 * position,
                    "directional_accuracy": float("nan") if model == "zero" else 0.5 + 0.001 * fold,
                    "coverage": 0.0 if model == "zero" else 0.995,
                    # Alternating sign, so "folds with positive skill" is neither
                    # all nor none: a count that is always 0/24 would pass whether
                    # or not the filter works.
                    "skill_score": 0.0 if model == "zero" else (-1) ** fold * 0.01 * (position + 1),
                    "n_predictions": 945,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def by_asset() -> pd.DataFrame:
    rows = []
    for asset_position, asset in enumerate(SYMBOLS):
        for position, model in enumerate(MODELS):
            rows.append(
                {
                    "model": model,
                    "asset": asset,
                    "rmse": 0.0414 + 0.001 * position,
                    "mae": 0.0276 + 0.001 * position,
                    "directional_accuracy": float("nan") if model == "zero" else 0.51,
                    "coverage": 0.0 if model == "zero" else 0.995,
                    "skill_score": 0.0 if model == "zero" else 0.002 * asset_position - 0.01 * position,
                    "n_predictions": 1512,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def build(
    config, calibration, descriptive, diagnostics, summary, by_asset, by_fold, dm, backtest,
    topology, event_study, folds, dates,
):
    """A callable rather than the string itself, so a test can build twice.

    Byte-stability across runs is a property of the module, and the only way to
    check it is to call it again on the same inputs.
    """
    baselines, gcn = diagnostics
    acf = pd.DataFrame(
        {symbol: [1.0, -0.04, *[0.01] * 29] for symbol in SYMBOLS},
        index=pd.Index(range(31), name="lag"),
    )
    acf_abs = pd.DataFrame(
        {symbol: [1.0, 0.24, *[0.09] * 29] for symbol in SYMBOLS},
        index=pd.Index(range(31), name="lag"),
    )
    ljung = pd.DataFrame(
        {"lb_stat": [46.6] * len(SYMBOLS), "lb_pvalue": [0.027] * (len(SYMBOLS) - 1) + [0.074]},
        index=list(SYMBOLS),
    )
    ljung_abs = pd.DataFrame(
        {"lb_stat": [563.9] * len(SYMBOLS), "lb_pvalue": [1e-99] * len(SYMBOLS)},
        index=list(SYMBOLS),
    )

    def once() -> str:
        return build_summary(
            config=config,
            config_sha1=CONFIG_SHA1,
            descriptive=descriptive,
            acf=acf,
            acf_abs=acf_abs,
            ljung_box=ljung,
            ljung_box_abs=ljung_abs,
            calibration=calibration,
            topology=topology,
            event_study=event_study,
            diagnostics_baselines=baselines,
            diagnostics_gcn=gcn,
            overall=summary,
            by_asset=by_asset,
            by_fold=by_fold,
            dm=dm,
            backtest=backtest,
            folds=folds,
            dates=dates,
        )

    return once


@pytest.fixture
def document(build) -> str:
    return build()


# --------------------------------------------------------------------------
# The fixtures have to describe the run they claim to
# --------------------------------------------------------------------------


def test_the_synthetic_panel_yields_the_studys_fold_count(folds):
    """N_FOLDS is a constant the per-fold fixtures are built from, and make_folds
    is what decides the real number. If the two drifted apart, every fold-level
    fixture would describe folds that do not exist.
    """
    assert len(folds) == N_FOLDS


# --------------------------------------------------------------------------
# Properties the document must have
# --------------------------------------------------------------------------


def test_every_section_is_present_and_in_order(document):
    positions = [document.index(f"## {title}") for title in SECTION_TITLES]

    assert positions == sorted(positions)


def test_no_non_number_reaches_the_page(document):
    """A `nan` in a Markdown cell renders as written and is copied into the thesis."""
    check_summary(document)


class TestCheckSummary:
    """A check that does not fail when the fault is there protects nothing."""

    def test_a_nan_in_a_cell_is_refused(self, document):
        corrupted = document.replace("| BTC |", "| BTC | nan |", 1)

        with pytest.raises(ValueError, match="'nan'"):
            check_summary(corrupted)

    @pytest.mark.parametrize("token", ["NaN", "inf", "None", "NaT"])
    def test_every_non_number_token_is_refused(self, token):
        with pytest.raises(ValueError, match=token):
            check_summary(f"## {SECTION_TITLES[0]}\n\n| a | {token} |\n")

    def test_a_missing_section_is_refused(self, document):
        truncated = document.replace(f"## {SECTION_TITLES[4]}", "## Altro", 1)

        with pytest.raises(ValueError, match="missing 1 section"):
            check_summary(truncated)

    def test_binance_is_not_mistaken_for_a_nan(self):
        """"nan" is a substring of "Binance", which Section 6.1 legitimately names.

        Matching on substrings rather than on word boundaries would make the check
        fail on a correct document, which is how a check gets switched off.
        """
        body = "\n".join(f"## {title}" for title in SECTION_TITLES)

        check_summary(body + "\n\nchiusure giornaliere Binance, infinitamente noiose.\n")


def test_the_document_is_reproducible_byte_for_byte(build):
    """Two builds from the same artifacts must be identical, or `git status`
    reports a modification that is only the file being rewritten -- and the run
    manifest then records a dirty tree that is clean.
    """
    assert build() == build()


def test_no_latex_math_leaks_into_the_markdown(document):
    """One model label carries LaTeX: `VAR ($p=5$)`. Rendered verbatim it would
    show the dollar signs on the page.
    """
    assert "$" not in document
    assert plain_model_label("var-p5") in document


def test_decimals_use_the_italian_comma(document):
    """A full stop between digits is the English convention, which the thesis does
    not use -- a value copied from here into the prose would be the only number in
    the document written the other way.

    The artifact index is excluded: its section numbers are `6.5`, which is a
    reference and not a quantity.
    """
    body = document[: document.index(f"## {SECTION_TITLES[-1]}")]
    rows = [line for line in body.splitlines() if line.startswith("|")]

    assert rows
    for row in rows:
        assert not re.search(r"\d\.\d", row), row


# --------------------------------------------------------------------------
# What the document has to contain
# --------------------------------------------------------------------------


class TestContents:
    def test_every_model_appears_under_its_printed_name(self, document):
        for model in (*MODELS, BACKTEST_REFERENCE):
            assert plain_model_label(model) in document, model

    @pytest.mark.parametrize("identifier", ["gcn-nograph", "var-bic", "var-p5"])
    def test_no_internal_identifier_reaches_the_page(self, document, identifier):
        """`gcn-nograph` in a document meant for the write-up is an identifier
        that would be copied into the thesis as-is.

        Only the hyphenated ones are checked: "ar" and "zero" are substrings of
        ordinary Italian words, so searching for them would fail on a correct
        document -- which is how a test gets deleted rather than fixed.
        """
        assert identifier not in document

    def test_the_undefined_directional_accuracy_is_a_dash(self, document):
        zero = next(line for line in document.splitlines() if line.startswith("| Zero |"))

        assert MISSING in zero

    def test_every_asset_of_the_universe_is_listed(self, document, descriptive):
        for symbol in descriptive.index:
            assert f"| {symbol} |" in document

    def test_the_three_thresholds_are_all_reported(self, document, calibration):
        for label in ("τ calibrata", "τ FWER", "τ fissa"):
            assert label in document
        assert "0,2145" in document

    def test_the_frozen_grid_is_reported_cell_by_cell(self, document, config):
        """The four cells are the evidence that the search was not narrow, which
        is the objection a negative result invites.
        """
        grid = config.model.gcn
        rows = [line for line in document.splitlines() if line.startswith(("| 16 |", "| 32 |"))]

        assert len(rows) == len(grid.hidden) * len(grid.dropout)

    def test_the_ablation_comparison_is_stated(self, document):
        assert "GCN contro ablazione senza grafo" in document

    def test_both_cost_levels_appear_in_the_backtest(self, document):
        assert "Sharpe 0 bps" in document
        assert "Sharpe 10 bps" in document

    def test_the_spearman_association_is_reported_for_both_thresholds(self, document):
        block = document[document.index("Associazione tra densità") :]

        for label in (METRIC_LABELS["graph_density"], METRIC_LABELS["graph_density_fwer"]):
            assert label in block

    def test_every_event_gets_its_own_table(self, document):
        for label in EVENTS.values():
            assert f"**{label}**" in document

    def test_the_event_date_itself_is_not_written_as_a_signed_offset(self, document):
        """`+0` reads as a direction the offset does not have: it is the event
        date, not zero days after it.
        """
        assert "| +0g |" not in document
        assert "| 0g |" in document

    def test_the_limitations_include_the_ones_that_emerged(self, document):
        """Four were foreseen in the plan; these four came out of the numbers, and
        they are the ones a list assembled from memory would omit.
        """
        for finding in (
            "Il grafo non è rado",
            "La baseline VAR selezionata degenera",
            "Nessun modello batte la previsione nulla",
            "Il vantaggio economico svanisce con i costi",
        ):
            assert finding in document, finding

    def test_the_provenance_points_at_the_manifest_rather_than_repeating_it(self, document):
        assert CONFIG_SHA1 in document
        assert "run_manifest.json" in document

    def test_the_document_carries_no_generation_instant(self, document):
        """A generation timestamp would make every rerun a diff, which is what the
        byte-stability property forbids.

        An ISO instant is the shape such a stamp takes here, since that is what
        build_manifest() writes into run_manifest.json -- the artifact where it
        belongs. A bare clock reading is not searched for: the limitations section
        legitimately says that Binance closes its candles at 00:00 UTC.
        """
        assert not re.search(r"\d{4}-\d{2}-\d{2}T", document)

    def test_the_artifact_index_lists_every_figure_and_table(self, document):
        from cryptognn.tables import TABLE_NAMES
        from cryptognn.viz.figures import FIGURE_NAMES

        index = document[document.index(f"## {SECTION_TITLES[-1]}") :]
        for name in (*FIGURE_NAMES, *TABLE_NAMES):
            assert name in index, name


class TestGcnArms:
    @pytest.mark.parametrize("arm", GCN_ARMS)
    def test_both_arms_are_reported_in_the_grid_and_the_selection(self, document, arm):
        assert plain_model_label(arm) in document

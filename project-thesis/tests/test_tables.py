"""Tests for cryptognn.tables: the LaTeX Chapter 6 includes.

Generated LaTeX fails in two ways, and only one of them is visible. A row with
the wrong number of cells stops the build with an error pointing at the wrong
line; a `nan` where a number should be compiles perfectly and reaches the
printed page. Both are checked here, on every table, because both become
expensive at exactly the moment there is no time left -- during the write-up.

The numbers themselves are not re-tested: they are the artifacts, and they are
tested where they are computed. What is tested is the formatting contract --
Italian decimal commas, an en dash for a value that legitimately does not exist,
a thin-space thousands separator, and a p-value that never claims to be zero.

The fixtures describing a finished run -- config, calibration, descriptive,
summary, dm, backtest, diagnostics -- are in tests/conftest.py, shared with
test_summary: the two modules format the same artifacts into two documents, and
a fixture that differed between them would let a table and the summary disagree
in the suite while agreeing in production.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from cryptognn.tables import (
    MISSING,
    TABLE_NAMES,
    THIN_SPACE,
    build_all,
    latex_integer,
    latex_number,
    latex_pvalue,
    latex_table,
    model_label,
    plain_integer,
    plain_number,
    plain_pvalue,
)


@pytest.fixture
def tables(descriptive, calibration, diagnostics, summary, dm, backtest, config) -> dict[str, str]:
    baselines, gcn = diagnostics
    return build_all(
        descriptive=descriptive,
        calibration=calibration,
        baselines=baselines,
        gcn=gcn,
        summary=summary,
        dm=dm,
        backtest=backtest,
        config=config,
    )


# --------------------------------------------------------------------------
# The formatting primitives
# --------------------------------------------------------------------------


class TestLatexNumber:
    def test_the_decimal_separator_is_a_braced_comma(self):
        """A bare comma in math mode is read as a list separator, so LaTeX puts a
        thin space after it and `$0,5$` prints as "0, 5".
        """
        assert latex_number(0.5, decimals=1) == "$0{,}5$"
        assert "," in latex_number(0.5, decimals=1)
        assert "0.5" not in latex_number(0.5, decimals=1)

    def test_negatives_keep_a_math_minus(self):
        assert latex_number(-0.0156, decimals=4) == "$-0{,}0156$"

    def test_rounds_to_the_requested_precision(self):
        assert latex_number(0.0414531, decimals=5) == "$0{,}04145$"
        assert latex_number(0.0414531, decimals=2) == "$0{,}04$"

    def test_percent_scales_and_escapes_the_sign(self):
        assert latex_number(0.5779, decimals=1, percent=True) == "$57{,}8\\%$"

    @pytest.mark.parametrize("value", [None, float("nan"), float("inf"), float("-inf")])
    def test_a_value_that_does_not_exist_prints_as_a_dash(self, value):
        """Both real cases are meaningful: the zero forecast has no directional
        accuracy and no Sharpe ratio, because it takes no position at all.
        """
        assert latex_number(value) == MISSING

    def test_accepts_a_numpy_scalar(self):
        assert latex_number(np.float64(0.25), decimals=2) == "$0{,}25$"


class TestPlainAndLatexAgree:
    """The LaTeX forms are the plain ones wrapped, and must stay so.

    summary.md quotes the plain form and the thesis typesets the LaTeX one. If
    the two ever rounded differently, the working notes and the printed table
    would disagree by a digit -- and the disagreement would be discovered by a
    reader, since nothing else compares them.
    """

    @pytest.mark.parametrize("value", [0.5, -0.0156, 0.0414531, 22680.0, 1140.0, 0.077])
    def test_the_latex_form_is_the_plain_form_in_math_mode(self, value):
        assert latex_number(value, decimals=4) == "$" + plain_number(value, decimals=4).replace(",", "{,}") + "$"

    def test_a_percentage_differs_only_by_the_escaped_sign(self):
        assert plain_number(0.5779, decimals=1, percent=True) == "57,8%"
        assert latex_number(0.5779, decimals=1, percent=True) == "$57{,}8\\%$"

    def test_the_thousands_separator_is_a_space_in_one_and_a_thin_space_in_the_other(self):
        assert plain_integer(1140) == f"1{THIN_SPACE}140"
        assert latex_integer(1140) == "$1\\,140$"

    def test_the_p_value_bound_survives_both_forms(self):
        assert plain_pvalue(7.6e-26) == "<0,001"
        assert latex_pvalue(7.6e-26) == "$<0{,}001$"

    @pytest.mark.parametrize("value", [None, float("nan"), float("inf")])
    def test_a_missing_value_is_the_same_dash_in_both(self, value):
        assert plain_number(value) == latex_number(value) == MISSING
        assert plain_integer(value) == latex_integer(value) == MISSING
        assert plain_pvalue(value) == latex_pvalue(value) == MISSING


class TestLatexInteger:
    def test_groups_thousands_with_a_thin_space(self):
        """The prose of Chapter 4 writes `12\\,550`."""
        assert latex_integer(1140) == "$1\\,140$"
        assert latex_integer(22680) == "$22\\,680$"

    def test_leaves_small_numbers_alone(self):
        assert latex_integer(105) == "$105$"

    def test_rounds_a_fold_average(self):
        assert latex_integer(17.67) == "$18$"

    def test_missing_prints_as_a_dash(self):
        assert latex_integer(float("nan")) == MISSING


class TestLatexPvalue:
    def test_a_p_below_the_resolution_is_a_bound_not_a_zero(self):
        """7.6e-26 rounded to three decimals reads 0,000, which asserts the
        probability is zero. It is not, and a table must not claim it is.
        """
        assert latex_pvalue(7.6e-26) == "$<0{,}001$"

    def test_an_ordinary_p_is_printed_as_itself(self):
        assert latex_pvalue(0.077) == "$0{,}077$"

    def test_missing_prints_as_a_dash(self):
        assert latex_pvalue(float("nan")) == MISSING


class TestLatexTable:
    def test_follows_the_house_order_caption_then_label_then_tabular(self):
        body = latex_table(
            rows=[["a", "$1$"]],
            header=["X", "Y"],
            column_spec="@{}lr@{}",
            caption="Lunga",
            short_caption="Breve",
            label="tab:x",
            small=True,
        )

        order = [body.index(token) for token in ("\\caption[", "\\label{tab:x}", "\\small", "\\begin{tabular}")]
        assert order == sorted(order)
        assert body.startswith("\\begin{table}[H]\n\\centering")
        assert "\\toprule" in body and "\\bottomrule" in body

    def test_a_row_with_the_wrong_cell_count_is_refused(self):
        """The one error in a generated table that is invisible on inspection and
        fatal at compile time. It is also what changes when a column is added.
        """
        with pytest.raises(ValueError, match="3 cells but the header declares 2"):
            latex_table(
                rows=[["a", "b", "c"]],
                header=["X", "Y"],
                column_spec="@{}lr@{}",
                caption="c",
                short_caption="s",
                label="tab:x",
            )

    def test_an_empty_table_is_refused(self):
        with pytest.raises(ValueError, match="no rows"):
            latex_table(rows=[], header=["X"], column_spec="@{}l@{}", caption="c", short_caption="s", label="tab:x")

    def test_midrules_separate_the_declared_blocks(self):
        body = latex_table(
            rows=[["a"], ["b"], ["c"]],
            header=["X"],
            column_spec="@{}l@{}",
            caption="c",
            short_caption="s",
            label="tab:x",
            midrules=(2,),
        )

        # One rule under the header, one before the third row.
        assert body.count("\\midrule") == 2


def test_model_label_falls_back_to_the_identifier():
    """An unlabelled identifier in a proof copy is a visible prompt to add one;
    raising here would stop a build over a cosmetic gap.
    """
    assert model_label("gcn") == "GCN"
    assert model_label("lstm") == "lstm"


# --------------------------------------------------------------------------
# Properties every table must have
# --------------------------------------------------------------------------


def _body_rows(table: str) -> list[str]:
    lines = table.splitlines()
    start = lines.index("\\midrule") + 1
    end = lines.index("\\bottomrule")
    return [line for line in lines[start:end] if line.rstrip().endswith("\\\\")]


def test_every_expected_table_is_built(tables):
    assert set(tables) == set(TABLE_NAMES)


@pytest.mark.parametrize("name", TABLE_NAMES)
class TestEveryTable:
    def test_environments_are_balanced(self, tables, name):
        table = tables[name]

        for opening, closing in (("\\begin{table}", "\\end{table}"), ("\\begin{tabular}", "\\end{tabular}")):
            assert table.count(opening) == 1, name
            assert table.count(closing) == 1, name

    def test_every_body_row_has_the_declared_cell_count(self, tables, name):
        table = tables[name]
        header = next(line for line in table.splitlines() if line.startswith("\\textbf{"))
        expected = header.count("&")

        for row in _body_rows(table):
            assert row.count("&") == expected, f"{name}: {row}"

    def test_no_cell_is_a_non_number(self, tables, name):
        """`nan` in a cell compiles perfectly and reaches the printed page."""
        for row in _body_rows(table := tables[name]):
            cells = {cell.strip() for cell in row.removesuffix("\\\\").split("&")}
            assert not cells & {"nan", "NaN", "$nan$", "inf", "None", ""}, f"{name}: {row}"
        assert "nan" not in table.replace("Binance", "")

    def test_carries_a_short_caption_and_a_label(self, tables, name):
        table = tables[name]

        assert "\\caption[" in table
        assert "\\label{tab:" in table

    def test_uses_no_package_the_thesis_lacks(self, tables, name):
        """The thesis preamble has booktabs, but no tabularx, threeparttable,
        multirow or siunitx configuration.
        """
        for forbidden in ("tabularx", "tablenotes", "multirow", "resizebox", "\\hline", "\\num{", "\\SI{"):
            assert forbidden not in tables[name], name

    def test_decimal_points_never_reach_a_cell(self, tables, name):
        """A full stop inside a number would be the English convention, which the
        rest of the document does not use.
        """
        for row in _body_rows(tables[name]):
            for cell in row.removesuffix("\\\\").split("&"):
                stripped = cell.strip()
                if stripped.startswith("$") and any(character.isdigit() for character in stripped):
                    assert "." not in stripped, f"{name}: {stripped}"


# --------------------------------------------------------------------------
# What each table has to contain
# --------------------------------------------------------------------------


class TestTableContents:
    def test_the_universe_lists_every_asset(self, tables, descriptive):
        rows = _body_rows(tables["tab_universe"])

        assert len(rows) == len(descriptive)
        for symbol in descriptive.index:
            assert any(row.startswith(symbol) for row in rows)

    def test_the_graph_table_carries_all_three_thresholds(self, tables, calibration):
        table = tables["tab_graph_params"]

        assert latex_number(calibration.tau, decimals=4) in table
        assert latex_number(calibration.tau_fwer, decimals=4) in table
        assert latex_number(calibration.tau_fixed, decimals=4) in table
        # Density is defined only for the thresholds, so the rest of the column
        # is dashes -- and there must be some, or the column means nothing.
        assert table.count(MISSING) > 3

    def test_the_model_table_reports_the_var_dimensionality(self, tables):
        """The empirical form of the argument in sec:var-baseline: 1140
        coefficients from a 365 x 15 panel is under five observations each.
        """
        row = next(row for row in _body_rows(tables["tab_models"]) if "$p=5$" in row)

        assert latex_integer(1140) in row
        assert "$4{,}8$" in row

    def test_the_results_table_has_one_row_per_model(self, tables, summary):
        rows = _body_rows(tables["tab_results_main"])

        assert len(rows) == len(summary)

    def test_the_results_table_dashes_the_undefined_accuracy(self, tables):
        zero = next(row for row in _body_rows(tables["tab_results_main"]) if row.startswith("Zero"))

        assert MISSING in zero

    def test_the_backtest_pairs_the_two_cost_levels(self, tables, backtest):
        table = tables["tab_backtest"]
        rows = _body_rows(table)

        assert len(rows) == backtest["model"].nunique()
        assert "\\cmidrule(lr){2-3}" in table
        # Once per group, three groups. Matched on the full header cell because
        # "10 bps" contains "0 bps" as a substring.
        assert table.count("\\textbf{0 bps}") == 3
        assert table.count("\\textbf{10 bps}") == 3

    def test_the_backtest_refuses_a_single_cost_level(self, backtest):
        from cryptognn.tables import table_backtest

        with pytest.raises(ValueError, match="two cost levels"):
            table_backtest(backtest[backtest["cost_bps"] == 0.0])


def test_captions_state_the_period_and_the_universe(tables, config):
    """A table lifted into a slide has to stay self-explanatory."""
    assert str(config.data.start.year) in tables["tab_universe"]
    assert str(config.data.end.year) in tables["tab_universe"]


def test_no_table_is_wider_than_seven_columns(tables):
    """\\textwidth is 426.79pt; past seven columns an Overfull box is likely and
    the house standard forbids one.
    """
    for name, table in tables.items():
        line = next(line for line in table.splitlines() if line.startswith("\\begin{tabular}"))
        # Only the spec between the braces: the word "tabular" itself carries an
        # l and an r, which would count as two phantom columns.
        spec = line.removeprefix("\\begin{tabular}{").removesuffix("}").replace("@{}", "")
        assert 0 < sum(spec.count(kind) for kind in "lrc") <= 7, f"{name}: {spec}"


def test_percentages_stay_within_a_plausible_range(tables):
    """A percentage formatter applied twice would print 5779%; a missing scaling
    would print 0,6%. Both compile.
    """
    for name, table in tables.items():
        for value in [float(token) for token in _percentages(table)]:
            assert -200.0 <= value <= 200.0, f"{name}: {value}%"


def _percentages(table: str) -> list[str]:
    import re

    return [match.replace("{,}", ".") for match in re.findall(r"\$(-?[\d{},]+)\\%\$", table)]


def test_math_is_balanced_in_every_cell(tables):
    """An odd number of dollar signs runs math mode into the next cell and the
    error surfaces pages away from its cause.
    """
    for name, table in tables.items():
        for row in _body_rows(table):
            assert row.count("$") % 2 == 0, f"{name}: {row}"


def test_the_universe_reports_positive_excess_kurtosis(tables, descriptive):
    """Not a formatting property: it is the stylized fact the table exists to
    show, and a sign error upstream would be invisible in the numbers alone.
    """
    assert (descriptive["excess_kurtosis"] > 0).all()
    assert all(not math.isnan(value) for value in descriptive["excess_kurtosis"])

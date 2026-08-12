"""LaTeX tables for Chapter 6, emitted from the artifacts rather than typed out.

Five tables, each one the printed form of a parquet file the pipeline already
produced. Generating them instead of transcribing them is the point: a number
copied by hand into a thesis is a number that will disagree with its source the
first time anything upstream is rerun, and nobody will notice until a reviewer
adds up a row.

The output matches the house style of ../latex-thesis, read off the three tables
already in the document: float `[H]`, `\\centering`, the caption **above** the
tabular with a short form for the list of tables, `\\label` after it, booktabs
rules with no vertical lines, `@{}` at both ends of the column spec, and bold
headers. Nothing here loads a package the thesis preamble does not already have.

Two conventions that are decisions rather than formatting:

  - **Decimals are Italian commas written literally**, as `$0{,}0415$` in math
    mode, which is how the prose of Chapters 3 and 4 writes them. siunitx is
    loaded by the thesis but never configured, and an `S` column would silently
    print a full stop until someone adds `\\sisetup{output-decimal-marker={,}}`.
    A generated file that renders correctly on its own is worth more than the
    decimal alignment an `S` column would add -- and with a fixed number of
    decimals per column, `r` alignment is exact anyway.
  - **A missing value prints as an en dash, never as `nan`.** Two are real and
    both are meaningful: ZeroForecaster has no directional accuracy because it
    takes no side, and no Sharpe ratio because it holds no position. Formatting
    them as blanks would suggest the run failed; printing `nan` in a thesis is
    worse.

Exports:
  - plain_number(), plain_integer(), plain_pvalue(): the number policy itself,
    without markup, so cryptognn.summary can write the same values into Markdown
  - latex_number(), latex_table(): the two primitives everything else is built on
  - table_universe(), table_graph_params(), table_models(),
    table_results_main(), table_backtest()
  - TABLES: the name -> builder mapping scripts/08_make_tables.py iterates

Integration: called by scripts/08_make_tables.py, which saves the strings to
  results/tables/ and copies them into ../latex-thesis/tables/.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import pandas as pd

from cryptognn.config import Config
from cryptognn.graph.threshold import TauCalibration

# An en dash for a value that does not exist, distinguishable from a minus sign.
MISSING = "--"

# The thousands separator of the plain forms, U+00A0. A non-breaking space rather
# than an ordinary one so a grouped number cannot be split across a line break,
# and named rather than inlined because the character is invisible at the call
# site. Its LaTeX counterpart is the thin space `\,` the thesis prose already uses.
# Suppressed inline rather than through allowed-confusables in pyproject: this is
# the single place the character appears, so the reason belongs beside it.
THIN_SPACE = " "  # noqa: RUF001 - U+00A0 is the point, not a typo

# Printed names, so the thesis never shows an internal identifier. var-bic and
# var-p5 are spelled out because the difference between them -- selected order
# against the pre-registered fixed one -- is the point of Section 6.4.
MODEL_LABELS = {
    "zero": "Zero",
    "mean": "Media storica",
    "ar": "AR",
    "var-bic": "VAR (BIC)",
    "var-p5": r"VAR ($p=5$)",
    "gcn": "GCN",
    "gcn-nograph": "GCN senza grafo",
    "buy-and-hold": "Buy-and-hold",
}


def model_label(name: str) -> str:
    """Printed name of a model, falling back to the identifier itself.

    Falls back rather than raising: a table is not the place to discover that a
    model was renamed, and an unlabelled identifier in a proof copy is a visible
    prompt to add one.
    """
    return MODEL_LABELS.get(name, name)


def plain_model_label(name: str) -> str:
    """The printed name with its math mode stripped, for Markdown.

    One label carries LaTeX: `VAR ($p=5$)`. Rendering it verbatim into summary.md
    would show the dollar signs, and keeping a second mapping would let the two
    drift apart the moment a model is renamed -- so the markup is removed from the
    one mapping rather than duplicated beside it.
    """
    return model_label(name).replace("$", "")


def plain_number(value: float | None, decimals: int = 4, percent: bool = False) -> str:
    """A number as unmarked Italian-decimal text, or an en dash if it is missing.

    The single place the study's number policy lives. The LaTeX forms below wrap
    what this returns, and cryptognn.summary writes it into Markdown unchanged,
    so the value quoted in the write-up notes and the value typeset in the thesis
    are the same string produced by the same call.

    Infinities are treated as missing rather than printed: an infinity in a
    results table is a computation that failed, and printing it invites a
    question that has no good answer.
    """
    if value is None:
        return MISSING
    number = float(value)
    if not math.isfinite(number):
        return MISSING

    if percent:
        number *= 100.0
    body = f"{number:.{decimals}f}".replace(".", ",")
    return f"{body}%" if percent else body


def plain_integer(value: float | None) -> str:
    """A whole number with a non-breaking-space thousands separator.

    The prose of Chapter 4 writes `12 550`, so a bare `1140` beside it would be
    the only unseparated four-digit number in the document.
    """
    if value is None or not math.isfinite(float(value)):
        return MISSING

    return f"{round(float(value)):,}".replace(",", THIN_SPACE)


def plain_pvalue(value: float | None, decimals: int = 3) -> str:
    """A p-value, floored at the resolution it is printed with.

    One of the p-values here is 7.6e-26. Rounded to three decimals it reads
    `0,000`, which states that the probability is zero -- it is not, and a table
    should not claim it is. Below the resolution the honest form is the bound.
    """
    if value is None or not math.isfinite(float(value)):
        return MISSING

    floor = 10.0**-decimals
    if float(value) < floor:
        return f"<{f'{floor:.{decimals}f}'.replace('.', ',')}"
    return plain_number(value, decimals=decimals)


def _math(text: str) -> str:
    """Wrap a plain number for math mode, escaping what math mode reads differently.

    `{,}` rather than a bare comma because in math mode LaTeX puts a thin space
    after a comma, reading it as a list separator -- so `$0,5$` prints as "0, 5".
    The leading minus needs nothing: inside math mode it is already a proper
    minus sign rather than a hyphen.
    """
    if text == MISSING:
        return MISSING

    body = text.replace(",", "{,}").replace("%", "\\%").replace(THIN_SPACE, "\\,")
    return f"${body}$"


def latex_number(value: float | None, decimals: int = 4, percent: bool = False) -> str:
    """A number as Italian-decimal math-mode LaTeX, or an en dash if it is missing."""
    return _math(plain_number(value, decimals=decimals, percent=percent))


def latex_integer(value: float | None) -> str:
    """A whole number with a thin-space thousands separator, as the thesis writes them."""
    return _math(plain_integer(value))


def latex_pvalue(value: float | None, decimals: int = 3) -> str:
    """A p-value, floored at the resolution it is printed with."""
    return _math(plain_pvalue(value, decimals=decimals))


def latex_table(
    rows: Sequence[Sequence[str]],
    header: Sequence[str],
    column_spec: str,
    caption: str,
    short_caption: str,
    label: str,
    small: bool = False,
    midrules: Sequence[int] = (),
    header_extra: str | None = None,
) -> str:
    """Assemble a booktabs table in the thesis's house style.

    `midrules` gives row indices to precede with a `\\midrule`, which is how a
    table separates blocks -- the graph parameters from the thresholds, the
    baselines from the GCN arms -- without a heading row that would need its own
    column. `header_extra` is a raw line inserted before the header, used for the
    grouped `\\cmidrule` header of the backtest table.

    Raises when a row does not have as many cells as the header. That mismatch is
    the one error in a generated table that is invisible on inspection and fatal
    at compile time, and it is exactly the kind of thing that changes when a
    column is added upstream.
    """
    if not rows:
        raise ValueError(f"Table {label!r} has no rows: there is nothing to print")
    for position, row in enumerate(rows):
        if len(row) != len(header):
            raise ValueError(
                f"Table {label!r} row {position} has {len(row)} cells but the header declares {len(header)}"
            )

    lines = [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption[{short_caption}]{{{caption}}}",
        f"\\label{{{label}}}",
    ]
    if small:
        lines.append("\\small")
    lines.append(f"\\begin{{tabular}}{{{column_spec}}}")
    lines.append("\\toprule")
    if header_extra:
        lines.append(header_extra)
    lines.append(" & ".join(f"\\textbf{{{cell}}}" for cell in header) + " \\\\")
    lines.append("\\midrule")

    for position, row in enumerate(rows):
        if position in midrules:
            lines.append("\\midrule")
        lines.append(" & ".join(row) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Section 6.1 -- the universe
# --------------------------------------------------------------------------


def table_universe(descriptive: pd.DataFrame, config: Config) -> str:
    """The 15 assets with the stylized facts that justify the modelling choices.

    Excess kurtosis is the column that earns its place: it is positive on every
    asset, which is the empirical form of the fat-tails argument that
    Section 4.4 uses to reject a Gaussian treatment of these returns, and it is
    also why the threshold is calibrated on a permutation null rather than on a
    parametric one.
    """
    rows = [
        [
            symbol,
            latex_number(row["mean"], decimals=3, percent=True),
            latex_number(row["volatility_annualized"], decimals=1, percent=True),
            latex_number(row["skewness"], decimals=2),
            latex_number(row["excess_kurtosis"], decimals=1),
        ]
        for symbol, row in descriptive.iterrows()
    ]

    return latex_table(
        rows=rows,
        header=["Asset", "Media giorn.", "Volatilità ann.", "Asimmetria", "Curtosi in ecc."],
        column_spec="@{}lrrrr@{}",
        caption=(
            f"Statistiche descrittive dei {len(descriptive)} asset dell'universo, "
            f"calcolate sui rendimenti logaritmici giornalieri dal "
            f"{config.data.start:%d/%m/%Y} al {config.data.end:%d/%m/%Y} "
            f"({len(descriptive)} serie, chiusure giornaliere Binance sulle coppie /USDT). "
            "La volatilità è annualizzata su 365 giorni, perché il mercato delle "
            "criptovalute non ha giorni di chiusura. La curtosi è in eccesso rispetto "
            "alla normale: è positiva su tutti gli asset, senza eccezioni."
        ),
        short_caption="Statistiche descrittive dell'universo",
        label="tab:universe",
    )


# --------------------------------------------------------------------------
# Section 6.2 -- the graph
# --------------------------------------------------------------------------


def table_graph_params(calibration: TauCalibration, config: Config) -> str:
    """Everything needed to rebuild the graph, plus what each threshold produces.

    The density column is what makes the table an argument rather than a list of
    settings: at the calibrated threshold the graph is 97% complete, so the
    thresholding step removes almost nothing, and the two robustness variants
    show that this is a property of the threshold and not of the market. Stating
    it here is what lets Section 6.6 discuss the saturation instead of being
    caught by it.
    """
    n_assets = len(config.data.symbols)
    q = n_assets / calibration.window
    density = calibration.density or {}

    def density_of(key: str) -> str:
        return latex_number(density.get(key, {}).get("mean"), decimals=3)

    geometry = [
        ["Ampiezza della finestra", "$T_w$", latex_integer(calibration.window), MISSING],
        ["Numero di asset", "$N$", latex_integer(n_assets), MISSING],
        ["Rapporto di aspetto", "$q = N/T_w$", latex_number(q, decimals=3), MISSING],
        ["Bordo superiore MP", r"$(1+\sqrt{q})^2$", latex_number((1 + math.sqrt(q)) ** 2, decimals=3), MISSING],
        ["Coppie di asset", r"$\binom{N}{2}$", latex_integer(calibration.n_pairs), MISSING],
    ]
    thresholds = [
        ["Soglia calibrata", r"$\tau$", latex_number(calibration.tau, decimals=4), density_of("tau")],
        [
            "Soglia FWER",
            r"$\tau_{\mathrm{FWER}}$",
            latex_number(calibration.tau_fwer, decimals=4),
            density_of("tau_fwer"),
        ],
        [
            "Soglia fissa",
            r"$\tau_{\mathrm{fissa}}$",
            latex_number(calibration.tau_fixed, decimals=4),
            density_of("tau_fixed"),
        ],
    ]
    calibration_rows = [
        ["Livello del test", r"$\alpha$", latex_number(calibration.alpha, decimals=2), MISSING],
        ["Permutazioni", "$B$", latex_integer(calibration.n_permutations), MISSING],
        ["Finestre di calibrazione", MISSING, latex_integer(calibration.n_calibration_windows), MISSING],
    ]

    rows = geometry + thresholds + calibration_rows
    return latex_table(
        rows=rows,
        header=["Parametro", "Simbolo", "Valore", "Densità media"],
        column_spec="@{}llrr@{}",
        caption=(
            "Parametri di costruzione del grafo di correlazione dinamico. La densità media "
            "è la frazione delle "
            f"{calibration.n_pairs} coppie che sopravvive alla soglia, mediata sulle finestre "
            "dell'intero periodo, ed è definita solo per le tre soglie. "
            r"$\tau$ è calibrata sul null di permutazione al livello $\alpha$ indicato, con $B$ "
            "rimescolamenti indipendenti per colonna su "
            f"{calibration.n_calibration_windows} finestre equispaziate; "
            r"$\tau_{\mathrm{FWER}}$ controlla l'errore per famiglia sulle coppie e "
            r"$\tau_{\mathrm{fissa}}$ è la soglia convenzionale, entrambe pre-registrate come "
            "varianti di robustezza. Il bordo di Marchenko-Pastur non concorre alla scelta della "
            "soglia: governa lo spettro della matrice, non la singola correlazione."
        ),
        short_caption="Parametri del grafo dinamico",
        label="tab:graph-params",
        midrules=(len(geometry), len(geometry) + len(thresholds)),
    )


# --------------------------------------------------------------------------
# Section 6.4 -- how each model is parametrized
# --------------------------------------------------------------------------


def table_models(baselines: pd.DataFrame, gcn: pd.DataFrame, config: Config) -> str:
    """Parameters estimated per fold, and how many observations support each one.

    The table Section 6.4 needs in order to make its central argument concretely:
    the VAR at the pre-registered lag 5 estimates 1140 coefficients from 365
    observations, under five data points per parameter. That is the curse of
    dimensionality of `sec:var-baseline` measured rather than asserted.

    It also records the fact that makes the fixed-lag variant necessary at all:
    BIC selects order 0 on every fold, so the "multivariate" baseline it would
    otherwise produce estimates not one cross-asset coefficient.
    """
    n_train = config.walkforward.train
    n_assets = len(config.data.symbols)
    # The panel a fold actually offers: every model here is fitted on all 15
    # series at once, and the VAR's own diagnostics count observations the same
    # way (365 x 15 / 1140 = 4.8 for the fixed-lag variant).
    n_observations = n_train * n_assets

    def mean_of(frame: pd.DataFrame, model: str, column: str) -> float:
        block = frame[frame["model"] == model]
        return float("nan") if block.empty or column not in block else float(block[column].mean())

    def row(model: str, selection: str, n_params: float) -> list[str]:
        supported = math.isfinite(n_params) and n_params > 0
        return [
            model_label(model),
            selection,
            latex_integer(n_params) if supported else MISSING,
            latex_number(n_observations / n_params, decimals=1) if supported else MISSING,
        ]

    var_bic_lag = mean_of(baselines, "var-bic", "var_lag_order")
    var_p5_params = mean_of(baselines, "var-p5", "n_params")
    ar_lag = latex_number(mean_of(baselines, "ar", "ar_lag_mean"), decimals=2)
    grid = config.model.gcn
    dropouts = " o ".join(latex_number(value, decimals=1) for value in grid.dropout)
    hidden = " o ".join(f"${value}$" for value in grid.hidden)

    rows = [
        [model_label("zero"), "nessuna stima", MISSING, MISSING],
        row("mean", "media del train", float(len(config.data.symbols))),
        row("ar", f"BIC, $p \\leq {config.model.ar.max_lag}$ (medio {ar_lag})", mean_of(baselines, "ar", "n_params")),
        row("var-bic", f"BIC, $p = {int(var_bic_lag)}$ su ogni fold", mean_of(baselines, "var-bic", "n_params")),
        row("var-p5", f"fisso, $p = {config.model.var.fixed_lag}$", var_p5_params),
        row("gcn", "griglia su validazione", mean_of(gcn, "gcn", "n_params")),
        row("gcn-nograph", "griglia su validazione", mean_of(gcn, "gcn-nograph", "n_params")),
    ]

    return latex_table(
        rows=rows,
        header=["Modello", "Selezione dell'ordine", "Parametri", "Oss./par."],
        column_spec="@{}llrr@{}",
        caption=(
            "Parametrizzazione dei modelli confrontati. I parametri sono la media sui fold, e le "
            f"osservazioni per parametro li rapportano al pannello di addestramento di un fold: "
            f"{n_train} giorni per {n_assets} asset, cioè {latex_integer(n_observations)} valori. "
            "Due letture, entrambe rilevanti per il capitolo: il VAR con ordine fissato a "
            f"{config.model.var.fixed_lag} stima {latex_integer(var_p5_params)} coefficienti, meno "
            "di cinque osservazioni ciascuno, che è la maledizione della dimensionalità in forma "
            f"misurata anziché asserita; e il criterio BIC seleziona ordine ${int(var_bic_lag)}$ su "
            "tutti i fold, per cui il VAR selezionato non stima alcun coefficiente incrociato e "
            "coincide numericamente con la media storica. La griglia della GCN "
            f"({hidden} unità nascoste, dropout {dropouts}) è stata congelata prima di osservare "
            "qualunque risultato; la configurazione è scelta sulla validazione interna al fold e la "
            f"previsione di test è la media sui {len(grid.seeds)} semi."
        ),
        short_caption="Parametrizzazione dei modelli",
        label="tab:models",
    )


# --------------------------------------------------------------------------
# Section 6.5 -- accuracy and the economic reading
# --------------------------------------------------------------------------


def table_results_main(summary: pd.DataFrame, dm: pd.DataFrame, baseline: str = "zero") -> str:
    """The comparison table: accuracy, skill, and whether the gap is real.

    The Diebold-Mariano statistic and its Holm-corrected p-value are columns
    rather than asterisks. The thesis has no `threeparttable`, so a star
    convention would have to be explained in the caption anyway, and at that
    point the number says more than the symbol. Holm rather than the raw p-value
    because 21 pairwise tests are precisely the data-snooping problem the thesis
    raises with `white2000reality`.
    """
    against = dm[dm["model_b"] == baseline].set_index("model_a")

    rows = []
    for _, entry in summary.iterrows():
        model = entry["model"]
        test = against.loc[model] if model in against.index else None
        rows.append(
            [
                model_label(model),
                latex_number(entry["rmse"], decimals=5),
                latex_number(entry["mae"], decimals=5),
                latex_number(entry["directional_accuracy"], decimals=1, percent=True),
                latex_number(entry["skill_score"], decimals=4),
                MISSING if test is None else latex_number(test["statistic"], decimals=2),
                MISSING if test is None else latex_pvalue(test["p_value_holm"]),
            ]
        )

    n_predictions = latex_integer(summary["n_predictions"].iloc[0])
    return latex_table(
        rows=rows,
        header=["Modello", "RMSE", "MAE", "Acc. dir.", "Skill", "DM", "$p$ (Holm)"],
        column_spec="@{}lrrrrrr@{}",
        caption=(
            f"Accuratezza fuori campione sui {n_predictions} valori previsti del periodo di test, "
            "ordinata per RMSE crescente. Lo skill score è la frazione di errore quadratico "
            "rimossa rispetto alla previsione nulla, quindi è positivo solo per un modello che la "
            "batte. La statistica di Diebold-Mariano confronta ciascun modello con la previsione "
            "nulla ed è \\emph{negativa quando il modello è il più accurato dei due}; è calcolata sul "
            "differenziale di perdita mediato tra i 15 asset a ogni data, non sulle previsioni "
            "singole, che non sono indipendenti. Il $p$ è corretto secondo Holm sull'insieme dei "
            "confronti a coppie. L'accuratezza direzionale della previsione nulla non è definita: "
            "non esprime alcuna direzione."
        ),
        short_caption="Confronto predittivo tra i modelli",
        label="tab:results-main",
        small=True,
    )


def table_backtest(backtest: pd.DataFrame) -> str:
    """Sharpe, drawdown and cumulative return, gross of costs and net of them.

    Grouped columns rather than one block per cost level: the quantity the
    section argues about is the distance between the two, and a reader compares
    adjacent cells far more reliably than cells eight rows apart.
    """
    costs = sorted(backtest["cost_bps"].unique())
    if len(costs) != 2:
        raise ValueError(f"Expected two cost levels for the paired columns, got {costs}")
    blocks = [backtest[backtest["cost_bps"] == cost].set_index("model") for cost in costs]

    rows = []
    for model in blocks[0].index:
        entries = [block.loc[model] for block in blocks]
        rows.append(
            [
                model_label(model),
                *[latex_number(entry["sharpe"], decimals=3) for entry in entries],
                *[latex_number(entry["max_drawdown"], decimals=1, percent=True) for entry in entries],
                *[latex_number(entry["cumulative_return"], decimals=1, percent=True) for entry in entries],
            ]
        )

    labels = [f"{cost:g}" for cost in costs]
    header_extra = (
        " & \\multicolumn{2}{c}{\\textbf{Sharpe}} & \\multicolumn{2}{c}{\\textbf{Max drawdown}} "
        "& \\multicolumn{2}{c}{\\textbf{Rend. cumulato}} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}"
    )
    n_days = latex_integer(backtest["n_days"].iloc[0])

    return latex_table(
        rows=rows,
        header=["Modello", *[f"{label} bps" for label in labels] * 3],
        column_spec="@{}lrrrrrr@{}",
        caption=(
            f"Strategia sul segno della previsione, equipesata sui 15 asset e ribilanciata "
            f"ogni giorno, sui {n_days} giorni del periodo di test. Ogni misura è riportata a "
            f"{labels[0]} e a {labels[1]} punti base di costo, applicati a ogni cambio di "
            "posizione; lo Sharpe è annualizzato su 365 giorni e non sottrae un tasso privo di "
            "rischio. Il buy-and-hold è il paniere equipesato acquistato il primo giorno e "
            "mantenuto, e non usa alcuna previsione: serve da termine di paragone. La previsione "
            "nulla non prende posizione, quindi il suo Sharpe non è definito."
        ),
        short_caption="Metriche economiche della strategia sul segno",
        label="tab:backtest",
        small=True,
        header_extra=header_extra,
    )


# The tables the pipeline emits, in the order Chapter 6 introduces them.
TABLE_NAMES = (
    "tab_universe",
    "tab_graph_params",
    "tab_models",
    "tab_results_main",
    "tab_backtest",
)


def build_all(
    descriptive: pd.DataFrame,
    calibration: TauCalibration,
    baselines: pd.DataFrame,
    gcn: pd.DataFrame,
    summary: pd.DataFrame,
    dm: pd.DataFrame,
    backtest: pd.DataFrame,
    config: Config,
) -> dict[str, str]:
    """Every table, keyed by filename stem, in the order the chapter uses them."""
    return {
        "tab_universe": table_universe(descriptive, config),
        "tab_graph_params": table_graph_params(calibration, config),
        "tab_models": table_models(baselines, gcn, config),
        "tab_results_main": table_results_main(summary, dm),
        "tab_backtest": table_backtest(backtest),
    }



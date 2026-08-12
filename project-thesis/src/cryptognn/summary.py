"""results/summary.md: every number Chapter 6 will quote, in one generated document.

The last artifact of the pipeline and the only one addressed to the author rather
than to the thesis. Its purpose is that writing Chapter 6 requires reading one
file instead of reopening eight parquets, and that no number reaches the prose by
being read off a figure or retyped from a terminal.

Generated for the same reason the LaTeX tables are: a number transcribed by hand
disagrees with its source the first time anything upstream is rerun, and nobody
notices until a reader adds up a column. The difference is one of audience, and
it shows in two places. The document carries values the tables deliberately leave
out -- the per-cell validation MSE of the frozen grid, the per-asset skill, the
full event study -- because a working note may be exhaustive where a printed
table must be selective. And it states the findings that are awkward, in the
"Limiti" section, because that section exists to be transplanted into
`sec:limitations` and the awkward ones are the ones a thesis is graded on.

Two properties the file has to keep:

  - **No timestamp anywhere.** Regenerating with nothing upstream changed must
    produce a byte-identical file, so that `git status` stays clean and the run
    manifest can honestly record an undirty tree. Provenance -- commit, packages,
    digests -- belongs in run_manifest.json, which is the artifact for it.
  - **Numbers formatted exactly as the thesis will print them**, via the plain_*
    primitives of cryptognn.tables. A value copied from here into the LaTeX is
    then a copy, not a re-rounding.

Exports:
  - SECTION_TITLES: every heading of the document, in order
  - build_summary(): the whole document as one string
  - check_summary(): reject a document containing a non-number

Integration: called by scripts/08_make_tables.py, which loads the artifacts and
  saves the result through artifacts.save_summary_markdown().
Why this module imports from viz: the Spearman association of Section 6.5 is
  computed by fold_test_means() + rank_association(), and calling them here is
  what guarantees the rho quoted in the prose is the rho annotated on
  fig_density_vs_error.pdf. Recomputing it a second way is how a thesis ends up
  disagreeing with its own figure.
"""
from __future__ import annotations

import math
import re
from collections.abc import Sequence

import pandas as pd

from cryptognn.config import Config
from cryptognn.evaluation.metrics import rank_association
from cryptognn.graph.threshold import TauCalibration
from cryptognn.tables import (
    MISSING,
    TABLE_NAMES,
    plain_integer,
    plain_model_label,
    plain_number,
    plain_pvalue,
)
from cryptognn.viz.figures import FIGURE_NAMES, fold_test_means

# The conventional level at which the Ljung-Box and Spearman results are read.
# Not a study parameter -- it does not belong in config/default.yaml, which holds
# the settings that define the experiment -- but naming it keeps the two places
# that count rejections from drifting apart.
SIGNIFICANCE = 0.05

# The task is r_{t+1}: a prediction made at position t is true of position t+1,
# which is how run_walkforward() dates its rows. Needed here only to name the
# first and last day the models actually forecast.
HORIZON = 1

# Every heading of the document, in order. Checked against what build_summary()
# emits, so a section quietly dropped by an edit is a failure rather than a gap
# discovered during the write-up.
SECTION_TITLES = (
    "6.1 Dati e fatti stilizzati",
    "6.2 Costruzione del grafo dinamico",
    "6.3 Architettura e griglia della GCN",
    "6.4 Protocollo di valutazione",
    "6.5 Confronto predittivo",
    "6.6 Struttura topologica e crisi",
    "Limiti da dichiarare",
    "Indice degli artefatti",
)

# Printed names of the topological metrics, so the document never shows a column
# identifier. The order is the order of topology.parquet.
METRIC_LABELS = {
    "mean_correlation": "Correlazione media",
    "graph_density": "Densità (τ calibrata)",
    "graph_density_fwer": "Densità (τ FWER)",
    "graph_density_fixed": "Densità (τ fissa)",
    "algebraic_connectivity": "Connettività algebrica (normalizzata)",
    "algebraic_connectivity_combinatorial": "Connettività algebrica (combinatoria)",
    "mst_length": "Lunghezza MST normalizzata",
    "spectral_entropy": "Entropia spettrale",
    "market_mode_share": "Quota del modo di mercato",
    "eigs_outside_mp": "Autovalori fuori dal bulk MP",
}


# --------------------------------------------------------------------------
# Markdown primitives
# --------------------------------------------------------------------------


def _heading(title: str, level: int = 2) -> list[str]:
    return ["", "#" * level + f" {title}", ""]


def _table(header: Sequence[str], rows: Sequence[Sequence[str]], align: str) -> list[str]:
    """A GitHub-flavoured Markdown table.

    `align` is one character per column, "l" or "r", and must cover the header:
    a table whose separator row has a different width than its header renders as
    literal pipes rather than as a table, which is invisible until someone opens
    the file.
    """
    if len(align) != len(header):
        raise ValueError(f"Alignment {align!r} covers {len(align)} columns, header declares {len(header)}")
    for position, row in enumerate(rows):
        if len(row) != len(header):
            raise ValueError(f"Row {position} has {len(row)} cells but the header declares {len(header)}")

    rule = ["---" if kind == "l" else "---:" for kind in align]
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(rule) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
        "",
    ]


def _bullets(items: Sequence[tuple[str, str]]) -> list[str]:
    return [f"- **{name}**: {value}" for name, value in items] + [""]


def _mean_of(frame: pd.DataFrame, model: str, column: str) -> float:
    """Mean of one column over one model's rows, NaN when either is absent.

    NaN rather than a raise: several diagnostics columns are defined only for the
    models that have them -- a VAR lag order means nothing for the zero forecast --
    and the formatters already turn a NaN into an en dash.
    """
    block = frame[frame["model"] == model]
    if block.empty or column not in block:
        return float("nan")
    return float(block[column].mean())


# --------------------------------------------------------------------------
# Section 6.1 -- the data
# --------------------------------------------------------------------------


def _section_data(
    config: Config,
    descriptive: pd.DataFrame,
    acf: pd.DataFrame,
    acf_abs: pd.DataFrame,
    ljung_box: pd.DataFrame,
    ljung_box_abs: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> list[str]:
    rows = [
        [
            str(symbol),
            plain_number(row["mean"], decimals=3, percent=True),
            plain_number(row["volatility_annualized"], decimals=1, percent=True),
            plain_number(row["skewness"], decimals=2),
            plain_number(row["excess_kurtosis"], decimals=1),
        ]
        for symbol, row in descriptive.iterrows()
    ]

    rejected = int((ljung_box["lb_pvalue"] < SIGNIFICANCE).sum())
    rejected_abs = int((ljung_box_abs["lb_pvalue"] < SIGNIFICANCE).sum())
    n_assets = len(descriptive)

    lines = _heading(SECTION_TITLES[0])
    lines += _bullets(
        [
            ("Universo", f"{plain_integer(len(config.data.symbols))} asset, "
             f"coppie /{config.data.quote} su {config.data.source.capitalize()}, intervallo {config.data.interval}"),
            ("Periodo", f"{config.data.start:%d/%m/%Y} – {config.data.end:%d/%m/%Y}"),
            ("Rendimenti logaritmici", f"{plain_integer(len(dates))} osservazioni per asset, "
             f"dal {dates[0]:%d/%m/%Y} al {dates[-1]:%d/%m/%Y}"),
        ]
    )
    lines += _table(
        ["Asset", "Media giorn.", "Volatilità ann.", "Asimmetria", "Curtosi in ecc."],
        rows,
        align="lrrrr",
    )
    lines += _bullets(
        [
            ("Curtosi in eccesso", f"positiva su {plain_integer(int((descriptive['excess_kurtosis'] > 0).sum()))} "
             f"asset su {plain_integer(n_assets)}, da {plain_number(descriptive['excess_kurtosis'].min(), decimals=1)} "
             f"a {plain_number(descriptive['excess_kurtosis'].max(), decimals=1)}"),
            ("ACF(1) dei rendimenti", f"media {plain_number(acf.loc[1].mean(), decimals=4)} "
             f"(intervallo {plain_number(acf.loc[1].min(), decimals=4)} – "
             f"{plain_number(acf.loc[1].max(), decimals=4)})"),
            ("ACF(1) dei rendimenti assoluti", f"media {plain_number(acf_abs.loc[1].mean(), decimals=4)}"),
            ("ACF(30) dei rendimenti assoluti", f"media {plain_number(acf_abs.loc[30].mean(), decimals=4)}, "
             "ancora positiva: decadimento lento"),
            ("Ljung–Box a 30 ritardi", f"rifiuta al {plain_number(SIGNIFICANCE, decimals=0, percent=True)} "
             f"su {plain_integer(rejected)} asset su {plain_integer(n_assets)} per i rendimenti, "
             f"su {plain_integer(rejected_abs)} su {plain_integer(n_assets)} per i rendimenti assoluti"),
        ]
    )
    lines += [
        "Lettura per la sez. 6.1: code pesanti su ogni serie, dipendenza lineare nei rendimenti "
        "debole, dipendenza nella volatilità forte e persistente. È il quadro che giustifica il "
        "null di permutazione della sez. 6.2 — che preserva le marginali — al posto di un null "
        "gaussiano, e che rende plausibile a priori un esito negativo sul livello dei rendimenti.",
        "",
        "Tabella corrispondente: `tab_universe.tex`.",
    ]
    return lines


# --------------------------------------------------------------------------
# Section 6.2 -- the graph
# --------------------------------------------------------------------------


def _section_graph(config: Config, calibration: TauCalibration) -> list[str]:
    n_assets = len(config.data.symbols)
    q = n_assets / calibration.window
    density = calibration.density or {}

    def density_row(key: str, label: str, value: float) -> list[str]:
        stats = density.get(key, {})
        return [
            label,
            plain_number(value, decimals=4),
            plain_number(stats.get("mean"), decimals=3),
            plain_number(stats.get("min"), decimals=3),
            plain_number(stats.get("max"), decimals=3),
            plain_number(stats.get("sd"), decimals=3),
        ]

    lines = _heading(SECTION_TITLES[1])
    lines += _bullets(
        [
            ("Finestra", f"T_w = {plain_integer(calibration.window)} giorni, passo 1"),
            ("Rapporto di aspetto", f"q = N/T_w = {plain_number(q, decimals=3)}"),
            ("Bordo superiore Marchenko–Pastur", f"(1+√q)² = {plain_number((1 + math.sqrt(q)) ** 2, decimals=3)}"),
            ("Coppie di asset", plain_integer(calibration.n_pairs)),
            ("Pesi", "Mantegna: d = √(2(1−ρ)), w = 1 − d/2, con self-loop e renormalization trick"),
            ("Null di calibrazione", f"permutazione indipendente per colonna, "
             f"α = {plain_number(calibration.alpha, decimals=2)}, "
             f"B = {plain_integer(calibration.n_permutations)}, "
             f"{plain_integer(calibration.n_calibration_windows)} finestre equispaziate, "
             f"statistica «{calibration.statistic}», seed {plain_integer(calibration.seed)}"),
        ]
    )
    lines += _table(
        ["Soglia", "Valore", "Densità media", "Min", "Max", "Dev. std"],
        [
            density_row("tau", "τ calibrata", calibration.tau),
            density_row("tau_fwer", "τ FWER", calibration.tau_fwer),
            density_row("tau_fixed", "τ fissa", calibration.tau_fixed),
        ],
        align="lrrrrr",
    )
    lines += [
        "Lettura per la sez. 6.2: la soglia calibrata lascia in piedi quasi tutti gli archi. "
        "Il grafo su cui la GCN opera non è rado, è quasi completo, e la sogliatura non è il "
        "passo selettivo che il nome suggerisce. Va detto nel capitolo prima che sia il lettore "
        "a dedurlo dalla densità della sez. 6.6.",
        "",
        "Il bordo MP non concorre alla scelta di τ: governa lo spettro della matrice, non la "
        "singola correlazione. Serve invece in 6.6 per contare gli autovalori fuori dal bulk.",
        "",
        "Tabella corrispondente: `tab_graph_params.tex`.",
    ]
    return lines


# --------------------------------------------------------------------------
# Section 6.3 -- the model
# --------------------------------------------------------------------------


def _section_model(config: Config, diagnostics_gcn: pd.DataFrame) -> list[str]:
    grid = config.model.gcn
    arms = list(dict.fromkeys(diagnostics_gcn["model"]))

    grid_rows = []
    for hidden in grid.hidden:
        for dropout in grid.dropout:
            column = f"val_mse_h{hidden}_d{dropout}"
            grid_rows.append(
                [
                    plain_integer(hidden),
                    plain_number(dropout, decimals=1),
                    *[plain_number(_mean_of(diagnostics_gcn, arm, column), decimals=6) for arm in arms],
                ]
            )

    selection_rows = []
    for arm in arms:
        block = diagnostics_gcn[diagnostics_gcn["model"] == arm]
        chosen = block.groupby(["selected_hidden", "selected_dropout"]).size().sort_values(ascending=False)
        (hidden, dropout), count = chosen.index[0], int(chosen.iloc[0])
        selection_rows.append(
            [
                plain_model_label(arm),
                f"h={plain_integer(hidden)}, p={plain_number(dropout, decimals=1)} "
                f"({plain_integer(count)}/{plain_integer(len(block))} fold)",
                plain_number(_mean_of(diagnostics_gcn, arm, "val_mse"), decimals=6),
                plain_number(_mean_of(diagnostics_gcn, arm, "epochs_mean"), decimals=1),
                plain_number(_mean_of(diagnostics_gcn, arm, "early_stopped_share"), decimals=1, percent=True),
                plain_integer(_mean_of(diagnostics_gcn, arm, "n_params")),
                plain_number(_mean_of(diagnostics_gcn, arm, "fit_seconds"), decimals=2),
            ]
        )

    n_configs = len(grid.hidden) * len(grid.dropout)
    lines = _heading(SECTION_TITLES[2])
    lines += _bullets(
        [
            ("Architettura", "Dropout → GCNLayer(F, h) → ReLU → Dropout → GCNLayer(h, 1); "
             "il layer è Â·H·W + b, cioè `eq:gcn` applicata due volte"),
            ("Feature per nodo", f"F = {plain_integer(config.features.lags + len(config.features.vol_windows) + 1)} "
             f"({plain_integer(config.features.lags)} rendimenti ritardati, volatilità realizzata a "
             f"{' e '.join(plain_integer(w) for w in config.features.vol_windows)} giorni, "
             "z-score del log-volume a 20 giorni)"),
            ("Ablazione", "`use_graph=False` sostituisce Â con l'identità: stessa capacità, stesse "
             "feature, nessun grafo. È l'isolamento del contributo del grafo"),
            ("Griglia congelata", f"{plain_integer(n_configs)} configurazioni "
             f"({'/'.join(plain_integer(h) for h in grid.hidden)} unità nascoste × "
             f"dropout {'/'.join(plain_number(d, decimals=1) for d in grid.dropout)}), "
             f"{plain_integer(len(grid.seeds))} semi, "
             f"{plain_integer(n_configs * len(grid.seeds))} fit per fold e per arm"),
            ("Ottimizzazione", f"Adam, lr = {plain_number(grid.lr, decimals=4)}, "
             f"weight decay = {plain_number(grid.weight_decay, decimals=5)}, "
             f"max {plain_integer(grid.epochs)} epoche, early stopping con pazienza "
             f"{plain_integer(grid.patience)} sull'MSE di validazione e ripristino dei pesi migliori"),
            ("Selezione", "configurazione scelta sulla validazione interna al fold, mai sul test; "
             f"previsione di test = media sui {plain_integer(len(grid.seeds))} semi"),
        ]
    )
    lines += ["MSE di validazione medio per cella della griglia:", ""]
    lines += _table(
        ["Unità nascoste", "Dropout", *[plain_model_label(arm) for arm in arms]],
        grid_rows,
        align="rr" + "r" * len(arms),
    )
    lines += ["Esito della selezione e costo dell'addestramento:", ""]
    lines += _table(
        ["Arm", "Configurazione più scelta", "MSE val.", "Epoche", "Early stop", "Parametri", "Secondi"],
        selection_rows,
        align="llrrrrr",
    )
    lines += [
        "Lettura per la sez. 6.3: le quattro celle della griglia sono separate alla sesta cifra "
        "decimale. La scelta dell'iperparametro non discrimina, il che è coerente con un segnale "
        "assente più che con un modello mal calibrato — e va detto, perché protegge il capitolo "
        "dall'obiezione «avete cercato poco».",
        "",
        "Tabella corrispondente: `tab_models.tex`.",
    ]
    return lines


# --------------------------------------------------------------------------
# Section 6.4 -- the protocol
# --------------------------------------------------------------------------


def _section_protocol(
    config: Config,
    diagnostics_baselines: pd.DataFrame,
    overall: pd.DataFrame,
    folds: Sequence,
    dates: pd.DatetimeIndex,
) -> list[str]:
    walk = config.walkforward
    first_target = dates[folds[0].test[0] + HORIZON]
    last_target = dates[folds[-1].test[-1] + HORIZON]

    var_bic_orders = sorted(
        {int(value) for value in diagnostics_baselines.loc[
            diagnostics_baselines["model"] == "var-bic", "var_lag_order"
        ].dropna()}
    )
    n_train_values = int(walk.train) * len(config.data.symbols)

    lines = _heading(SECTION_TITLES[3])
    lines += _bullets(
        [
            ("Schema", f"walk-forward {walk.mode}, {plain_integer(len(folds))} fold"),
            ("Blocchi", f"train {plain_integer(walk.train)} / validazione {plain_integer(walk.val)} / "
             f"test {plain_integer(walk.test)} giorni, passo {plain_integer(walk.step)}"),
            ("Offset iniziale", f"{plain_integer(config.graph.window - 1)} "
             "(= finestra − 1: prima della sessantesima osservazione non esiste alcun grafo)"),
            ("Periodo previsto", f"dal {first_target:%d/%m/%Y} al {last_target:%d/%m/%Y}, "
             f"orizzonte {plain_integer(HORIZON)} giorno"),
            ("Previsioni per modello", f"{plain_integer(overall['n_predictions'].iloc[0])} "
             f"({plain_integer(len(folds) * walk.test)} giorni × "
             f"{plain_integer(len(config.data.symbols))} asset)"),
            ("Standardizzazione", "`FoldStandardizer` fittato sul solo train, per asset, e applicato "
             "invariato a validazione e test"),
            ("Difesa anti-look-ahead", "quattro test in `tests/evaluation/test_walkforward.py`, "
             "ciascuno provato contro una mutazione iniettata"),
        ]
    )
    lines += _bullets(
        [
            ("Ordine VAR per BIC", f"{', '.join(plain_integer(order) for order in var_bic_orders)} "
             f"su tutti i {plain_integer(len(folds))} fold — la baseline multivariata selezionata non "
             "stima alcun coefficiente incrociato e coincide numericamente con la media storica"),
            ("VAR a ordine fissato", f"{plain_integer(_mean_of(diagnostics_baselines, 'var-p5', 'n_params'))} "
             f"coefficienti da {plain_integer(n_train_values)} valori di addestramento, cioè "
             f"{plain_number(_mean_of(diagnostics_baselines, 'var-p5', 'obs_per_param'), decimals=2)} "
             "osservazioni per parametro"),
            ("Ordine AR medio", plain_number(_mean_of(diagnostics_baselines, "ar", "ar_lag_mean"), decimals=3)),
            ("Quota di AR con ordine 0", plain_number(
                _mean_of(diagnostics_baselines, "ar", "ar_zero_order_share"), decimals=1, percent=True
            )),
        ]
    )
    lines += ["Figura corrispondente: `fig_walkforward_scheme.pdf`."]
    return lines


# --------------------------------------------------------------------------
# Section 6.5 -- the comparison
# --------------------------------------------------------------------------


def _section_results(
    overall: pd.DataFrame,
    by_asset: pd.DataFrame,
    by_fold: pd.DataFrame,
    dm: pd.DataFrame,
    backtest: pd.DataFrame,
    topology: pd.DataFrame,
    folds: Sequence,
    dates: pd.DatetimeIndex,
    baseline: str = "zero",
) -> list[str]:
    against = dm[dm["model_b"] == baseline].set_index("model_a")
    models = list(overall["model"])

    accuracy_rows = []
    for _, entry in overall.iterrows():
        model = entry["model"]
        test = against.loc[model] if model in against.index else None
        positive = by_fold[(by_fold["model"] == model) & (by_fold["skill_score"] > 0)]
        accuracy_rows.append(
            [
                plain_model_label(model),
                plain_number(entry["rmse"], decimals=5),
                plain_number(entry["mae"], decimals=5),
                plain_number(entry["directional_accuracy"], decimals=1, percent=True),
                plain_number(entry["skill_score"], decimals=4),
                MISSING if test is None else plain_number(test["statistic"], decimals=2),
                MISSING if test is None else plain_pvalue(test["p_value_holm"]),
                f"{plain_integer(len(positive))}/{plain_integer(len(folds))}",
            ]
        )

    lines = _heading(SECTION_TITLES[4])
    lines += ["Accuratezza fuori campione, ordinata per RMSE crescente. La statistica di "
              "Diebold–Mariano confronta ciascun modello con la previsione nulla ed è **negativa "
              "quando il modello è il più accurato dei due**; il p è corretto secondo Holm "
              "sull'insieme dei confronti a coppie.", ""]
    lines += _table(
        ["Modello", "RMSE", "MAE", "Acc. dir.", "Skill", "DM vs zero", "p (Holm)", "Fold con skill > 0"],
        accuracy_rows,
        align="lrrrrrrr",
    )

    lines += _section_ablation(dm)
    lines += _section_by_asset(by_asset)
    lines += _section_backtest(backtest, models)
    lines += _section_association(by_fold, topology, folds, dates)

    lines += [
        "Tabelle corrispondenti: `tab_results_main.tex`, `tab_backtest.tex`. "
        "Figure: `fig_results_by_fold.pdf`, `fig_equity_curves.pdf`, `fig_density_vs_error.pdf`.",
    ]
    return lines


def _section_ablation(dm: pd.DataFrame) -> list[str]:
    """The one comparison the chapter's first research question turns on."""
    pair = dm[(dm["model_a"] == "gcn") & (dm["model_b"] == "gcn-nograph")]
    if pair.empty:
        return []

    entry = pair.iloc[0]
    return _bullets(
        [
            ("GCN contro ablazione senza grafo",
             f"DM = {plain_number(entry['statistic'], decimals=3)}, "
             f"p = {plain_pvalue(entry['p_value'])}, "
             f"p (Holm) = {plain_pvalue(entry['p_value_holm'])}, "
             f"differenziale medio di perdita = {plain_number(entry['mean_loss_differential'], decimals=8)}"),
            ("Lettura", "il segno positivo della statistica dice che la GCN **con** grafo è la meno "
             "accurata delle due, e il p dice che la differenza non è distinguibile dal rumore. "
             "Il grafo non peggiora in modo dimostrabile, ma non c'è alcuna evidenza che aiuti: è "
             "la risposta diretta alla prima questione di ricerca, e va formulata così"),
        ]
    )


def _section_by_asset(by_asset: pd.DataFrame) -> list[str]:
    """Per-asset skill of the two GCN arms: 15 rows the printed table has no room for."""
    arms = [arm for arm in ("gcn", "gcn-nograph") if arm in set(by_asset["model"])]
    if not arms:
        return []

    pivot = by_asset[by_asset["model"].isin(arms)].pivot(index="asset", columns="model", values="skill_score")
    rows = [
        [str(asset), *[plain_number(pivot.loc[asset, arm], decimals=4) for arm in arms]]
        for asset in pivot.index
    ]
    positive = {arm: int((pivot[arm] > 0).sum()) for arm in arms}

    lines = ["Skill score per asset, i due arm della GCN contro la previsione nulla:", ""]
    lines += _table(["Asset", *[plain_model_label(arm) for arm in arms]], rows, align="l" + "r" * len(arms))
    lines += _bullets(
        [
            ("Asset con skill positivo",
             ", ".join(f"{plain_model_label(arm)} {plain_integer(count)}/{plain_integer(len(pivot))}"
                       for arm, count in positive.items())),
        ]
    )
    return lines


def _section_backtest(backtest: pd.DataFrame, models: Sequence[str]) -> list[str]:
    costs = sorted(backtest["cost_bps"].unique())
    if len(costs) != 2:
        raise ValueError(f"Expected two cost levels, got {costs}")
    blocks = [backtest[backtest["cost_bps"] == cost].set_index("model") for cost in costs]

    ordered = [model for model in models if model in blocks[0].index]
    ordered += [model for model in blocks[0].index if model not in ordered]

    rows = []
    for model in ordered:
        entries = [block.loc[model] for block in blocks]
        rows.append(
            [
                plain_model_label(model),
                *[plain_number(entry["sharpe"], decimals=3) for entry in entries],
                *[plain_number(entry["max_drawdown"], decimals=1, percent=True) for entry in entries],
                *[plain_number(entry["cumulative_return"], decimals=1, percent=True) for entry in entries],
                plain_number(blocks[0].loc[model, "mean_turnover"], decimals=3),
            ]
        )

    labels = [f"{cost:g} bps" for cost in costs]
    lines = [
        f"Strategia sul segno della previsione, equipesata e ribilanciata ogni giorno, su "
        f"{plain_integer(backtest['n_days'].iloc[0])} giorni. Sharpe annualizzato su 365 giorni, "
        "senza tasso privo di rischio. Il buy-and-hold non usa alcuna previsione.",
        "",
    ]
    lines += _table(
        ["Modello", *[f"Sharpe {label}" for label in labels],
         *[f"Max DD {label}" for label in labels],
         *[f"Cumulato {label}" for label in labels], "Turnover"],
        rows,
        align="lrrrrrrr",
    )
    return lines


def _section_association(
    by_fold: pd.DataFrame, topology: pd.DataFrame, folds: Sequence, dates: pd.DatetimeIndex
) -> list[str]:
    """Spearman rho between a fold's mean graph density and the GCN's skill on it.

    The empirical form of the tension the thesis is built around: structure is
    most pronounced exactly when it is hardest to exploit. Reported whatever it
    comes out as -- an absence of relation is itself an answer, and one the
    chapter has to state rather than omit.
    """
    skill = by_fold[by_fold["model"] == "gcn"].sort_values("fold")["skill_score"].to_numpy()

    rows = []
    for column in ("graph_density", "graph_density_fwer"):
        if column not in topology:
            continue
        density = fold_test_means(topology, folds, dates, column)
        association = rank_association(density, skill)
        rows.append(
            [
                METRIC_LABELS.get(column, column),
                plain_number(association.rho, decimals=3),
                plain_pvalue(association.p_value),
                plain_integer(association.n),
                f"{plain_number(density.min(), decimals=3)} – {plain_number(density.max(), decimals=3)}",
                plain_integer(int((density >= 1.0).sum())),
            ]
        )

    lines = [
        "Associazione tra densità media del grafo nel fold e skill score della GCN nello stesso "
        "fold, per le due soglie:",
        "",
    ]
    lines += _table(
        ["Misura di densità", "ρ di Spearman", "p", "n", "Intervallo", "Fold a densità 1"],
        rows,
        align="lrrrrr",
    )
    lines += [
        "Spearman e non Pearson perché l'ipotesi è di monotonicità, e perché alla soglia calibrata "
        "un blocco di fold è saturo a densità esattamente 1: una misura che satura, che i ranghi "
        "reggono e un momento prodotto no.",
        "",
    ]
    return lines


# --------------------------------------------------------------------------
# Section 6.6 -- topology
# --------------------------------------------------------------------------


def _section_topology(topology: pd.DataFrame, event_study: pd.DataFrame) -> list[str]:
    metrics = [column for column in METRIC_LABELS if column in topology]
    rows = [
        [
            METRIC_LABELS[metric],
            plain_number(topology[metric].mean(), decimals=3),
            plain_number(topology[metric].std(), decimals=3),
            plain_number(topology[metric].min(), decimals=3),
            plain_number(topology[metric].median(), decimals=3),
            plain_number(topology[metric].max(), decimals=3),
        ]
        for metric in metrics
    ]

    lines = _heading(SECTION_TITLES[5])
    lines += _bullets(
        [
            ("Finestre", f"{plain_integer(len(topology))}, dal {topology.index[0]:%d/%m/%Y} "
             f"al {topology.index[-1]:%d/%m/%Y}"),
            ("Grafo usato", "metriche sul grafo **completo** pesato Mantegna, dove ogni w > 0; la "
             "densità è l'unica calcolata sul grafo soglia, perché è l'unica che per definizione "
             "dipende da τ"),
        ]
    )
    lines += _table(
        ["Metrica", "Media", "Dev. std", "Min", "Mediana", "Max"], rows, align="lrrrrr"
    )
    lines += _event_tables(event_study)
    lines += [
        "Figure corrispondenti: `fig_topology_timeseries.pdf`, `fig_correlation_heatmaps.pdf`, "
        "`fig_graph_snapshots.pdf`, `fig_mp_spectrum.pdf`.",
    ]
    return lines


def _event_tables(event_study: pd.DataFrame) -> list[str]:
    """One table per crisis, reading every metric at every offset.

    The full grid rather than a selection: a metric at date t is computed on
    [t-59, t], so the value on the event date is almost entirely pre-event data,
    and the honest comparison is between the two non-overlapping windows at -60
    and +60. A table that stopped at +/-30 would understate every effect for a
    reason that is an artifact of the rolling window.
    """
    offsets = sorted(event_study["offset_days"].unique())
    lines = ["### Studio degli eventi", "",
             f"Metriche lette a {', '.join(_offset(offset) for offset in offsets)} giorni da ciascun "
             "evento. La variazione riportata è tra i due offset estremi, cioè tra due finestre che "
             "non condividono alcuna osservazione; il percentile colloca il valore nella "
             "distribuzione storica completa.", ""]

    for key in dict.fromkeys(event_study["event_key"]):
        block = event_study[event_study["event_key"] == key]
        label = str(block["label"].iloc[0])
        date = block["event_date"].iloc[0]

        values = block.pivot(index="metric", columns="offset_days", values="value")
        percentiles = block.pivot(index="metric", columns="offset_days", values="percentile")
        changes = block.groupby("metric")[["pct_change_clean", "pct_change_local"]].first()

        rows = []
        for metric in [column for column in METRIC_LABELS if column in values.index]:
            rows.append(
                [
                    METRIC_LABELS[metric],
                    *[plain_number(values.loc[metric, offset], decimals=3) for offset in offsets],
                    plain_number(changes.loc[metric, "pct_change_clean"], decimals=1) + "%",
                    plain_number(changes.loc[metric, "pct_change_local"], decimals=1) + "%",
                    plain_number(percentiles.loc[metric, offsets[-1]], decimals=1, percent=True),
                ]
            )

        lines += [f"**{label}** — {date:%d/%m/%Y}", ""]
        lines += _table(
            ["Metrica", *[f"{_offset(offset)}g" for offset in offsets],
             f"Δ {_offset(offsets[0])}→{_offset(offsets[-1])}",
             f"Δ {_offset(offsets[1])}→{_offset(offsets[-2])}",
             f"Perc. a {_offset(offsets[-1])}g"],
            rows,
            align="l" + "r" * (len(offsets) + 3),
        )
    return lines


def _offset(days: int) -> str:
    """A signed day offset, with the event date itself written as plain "0".

    `f"{0:+d}"` gives "+0", which reads as a direction the offset does not have:
    it is the event date, not thirty days after it.
    """
    return "0" if days == 0 else f"{days:+d}"


# --------------------------------------------------------------------------
# Limitations
# --------------------------------------------------------------------------


def _section_limitations(
    config: Config,
    calibration: TauCalibration,
    overall: pd.DataFrame,
    diagnostics_baselines: pd.DataFrame,
    backtest: pd.DataFrame,
) -> list[str]:
    """The list `sec:limitations` is written from, ready to be transplanted.

    Four of these were foreseen in the plan and four emerged from the numbers.
    They are kept in one list, undistinguished, because the distinction matters
    to the project's history and not to the reader of the thesis -- and because
    separating them would invite presenting the foreseen ones and quietly
    dropping the others.
    """
    density = (calibration.density or {}).get("tau", {})
    positive_skill = int((overall["skill_score"] > 0).sum())
    var_bic_order = _mean_of(diagnostics_baselines, "var-bic", "var_lag_order")
    costly = backtest[backtest["cost_bps"] == backtest["cost_bps"].max()].set_index("model")

    def sharpe_of(model: str) -> str:
        return plain_number(costly.loc[model, "sharpe"], decimals=3) if model in costly.index else MISSING

    items = [
        ("Survivorship bias dell'universo",
         f"i {plain_integer(len(config.data.symbols))} asset sono selezionati perché quotati con "
         "continuità su Binance dall'inizio del periodo. Gli asset delistati o collassati nel "
         "frattempo — Terra/Luna fra tutti — non compaiono, e il campione è per costruzione quello "
         "dei sopravvissuti. L'effetto è nella direzione che favorisce i risultati riportati."),
        ("Periodo, fonte e valuta unici",
         f"un solo intervallo ({config.data.start:%Y}–{config.data.end:%Y}), un solo exchange "
         f"({config.data.source.capitalize()}), una sola valuta di quotazione ({config.data.quote}). Nessuna "
         "verifica di robustezza rispetto a un'altra fonte, e le chiusure a 00:00 UTC sono una "
         "convenzione dell'exchange, non un fatto del mercato."),
        ("Snapshot indipendenti",
         "il modello vede un grafo per data e nessuna dinamica: nessuna memoria fra finestre "
         "consecutive, nessun modulo ricorrente. È il limite già riconosciuto in "
         "`sec:dynamic-graphs-temporal-question`, e resta il primo candidato per un lavoro futuro."),
        ("Il grafo non è rado",
         f"alla soglia calibrata la densità media è {plain_number(density.get('mean'), decimals=3)} "
         f"(min {plain_number(density.get('min'), decimals=3)}, "
         f"max {plain_number(density.get('max'), decimals=3)}): la sogliatura elimina pochi archi e "
         "il substrato della GCN è quasi il grafo completo. Le proprietà che la letteratura "
         "attribuisce alla sparsità non sono verificate qui, e ogni affermazione sul contributo "
         "della *struttura* va letta a questa luce."),
        ("La baseline VAR selezionata degenera",
         f"il BIC sceglie ordine {plain_integer(var_bic_order)} su ogni fold, quindi il VAR "
         "selezionato non stima alcun coefficiente incrociato e coincide numericamente con la media "
         "storica. Il confronto multivariato effettivo è quello con il VAR a ordine fissato, "
         "pre-registrato proprio per questa eventualità."),
        ("Nessun modello batte la previsione nulla",
         f"lo skill score è positivo per {plain_integer(positive_skill)} modelli su "
         f"{plain_integer(len(overall))}, e per la GCN il test di Diebold–Mariano contro la "
         "previsione nulla non è significativo dopo correzione di Holm. Il risultato del capitolo è "
         "negativo, ed è riportato come tale: non è un fallimento dell'esperimento, è il suo esito."),
        ("Il vantaggio economico svanisce con i costi",
         f"a {plain_number(backtest['cost_bps'].max(), decimals=0)} punti base lo Sharpe della GCN "
         f"scende a {sharpe_of('gcn')} e quello dell'ablazione senza grafo a "
         f"{sharpe_of('gcn-nograph')}, contro {sharpe_of('buy-and-hold')} del buy-and-hold. Le "
         "strategie sul segno hanno turnover elevato e il costo se lo mangia: un backtest a costo "
         "zero non sarebbe stato confrontabile con nulla."),
        ("Soglia unica per l'intero periodo",
         f"τ = {plain_number(calibration.tau, decimals=4)} è calibrata una volta e tenuta fissa, "
         "per rendere la densità confrontabile fra periodi. È una scelta dichiarata, non un difetto "
         "scoperto, ma implica che la soglia non si adatta ai regimi — e i regimi sono l'oggetto "
         "della sez. 6.6."),
    ]

    lines = _heading(SECTION_TITLES[6])
    lines += ["Elenco pronto per `sec:limitations`. Ogni voce è verificabile su un artefatto "
              "di `results/metrics/`.", ""]
    lines += [f"{position}. **{name}** — {text}" for position, (name, text) in enumerate(items, start=1)]
    return lines


# --------------------------------------------------------------------------
# Artifact index
# --------------------------------------------------------------------------


def _section_index() -> list[str]:
    figure_sections = {
        "fig_topology_timeseries": "6.6",
        "fig_correlation_heatmaps": "6.6",
        "fig_graph_snapshots": "6.6",
        "fig_mp_spectrum": "6.6",
        "fig_walkforward_scheme": "6.4",
        "fig_results_by_fold": "6.5",
        "fig_equity_curves": "6.5",
        "fig_density_vs_error": "6.5",
    }
    table_sections = {
        "tab_universe": "6.1",
        "tab_graph_params": "6.2",
        "tab_models": "6.3",
        "tab_results_main": "6.5",
        "tab_backtest": "6.5",
    }

    lines = _heading(SECTION_TITLES[7])
    lines += ["Tutti pubblicati in `../latex-thesis/` da `scripts/08_make_tables.py`.", ""]
    lines += _table(
        ["Artefatto", "Tipo", "Sezione"],
        [[f"`{name}.pdf`", "figura", figure_sections.get(name, MISSING)] for name in FIGURE_NAMES]
        + [[f"`{name}.tex`", "tabella", table_sections.get(name, MISSING)] for name in TABLE_NAMES],
        align="llr",
    )
    return lines


# --------------------------------------------------------------------------
# Assembly and validation
# --------------------------------------------------------------------------


def build_summary(
    *,
    config: Config,
    config_sha1: str,
    descriptive: pd.DataFrame,
    acf: pd.DataFrame,
    acf_abs: pd.DataFrame,
    ljung_box: pd.DataFrame,
    ljung_box_abs: pd.DataFrame,
    calibration: TauCalibration,
    topology: pd.DataFrame,
    event_study: pd.DataFrame,
    diagnostics_baselines: pd.DataFrame,
    diagnostics_gcn: pd.DataFrame,
    overall: pd.DataFrame,
    by_asset: pd.DataFrame,
    by_fold: pd.DataFrame,
    dm: pd.DataFrame,
    backtest: pd.DataFrame,
    folds: Sequence,
    dates: pd.DatetimeIndex,
) -> str:
    """The whole document, from artifacts the caller has already loaded.

    Keyword-only and unabbreviated: eighteen positional arguments of which nine
    are DataFrames is a call nobody can read, and two frames swapped at the call
    site would produce a document that is wrong rather than one that fails.
    """
    lines = [
        "# Sintesi dei risultati — Capitolo 6",
        "",
        "> **Documento generato.** Rigenerare con `python scripts/08_make_tables.py`; una modifica "
        "manuale viene sovrascritta al primo rerun.",
        f"> Configurazione: `config/default.yaml`, SHA-1 `{config_sha1}`.",
        "> Provenienza del run — commit, ambiente, digest di ogni artefatto — in "
        "`results/run_manifest.json`.",
        "",
        "Ogni numero è letto dagli artefatti di `results/metrics/` e formattato come lo stamperà la "
        "tesi: virgola decimale, trattino per un valore che non esiste. Un valore copiato da qui "
        "nel LaTeX è una copia, non un arrotondamento nuovo.",
    ]

    lines += _section_data(config, descriptive, acf, acf_abs, ljung_box, ljung_box_abs, dates)
    lines += _section_graph(config, calibration)
    lines += _section_model(config, diagnostics_gcn)
    lines += _section_protocol(config, diagnostics_baselines, overall, folds, dates)
    lines += _section_results(overall, by_asset, by_fold, dm, backtest, topology, folds, dates)
    lines += _section_topology(topology, event_study)
    lines += _section_limitations(config, calibration, overall, diagnostics_baselines, backtest)
    lines += _section_index()

    return "\n".join(lines).rstrip() + "\n"


# A token that reached the document is a formatting failure that renders as
# written. Matched on word boundaries rather than searched for as a substring:
# "nan" occurs inside "Binance", which Section 6.1 legitimately names.
NON_NUMBERS = ("nan", "NaN", "inf", "None", "NaT")
_NON_NUMBER = re.compile(r"(?<![\w-])(" + "|".join(NON_NUMBERS) + r")(?![\w-])")


def check_summary(body: str) -> None:
    """Reject a summary that prints a non-number, or that lost a section.

    Both failures are silent. A `nan` in a Markdown cell renders perfectly and is
    copied into the thesis; a section dropped by an edit leaves a gap discovered
    during the write-up, which is the one moment there is no time to regenerate
    it.
    """
    match = _NON_NUMBER.search(body)
    if match:
        line = body[: match.start()].count("\n") + 1
        context = body.splitlines()[line - 1].strip()
        raise ValueError(f"summary.md line {line}: {match.group(0)!r} would be read as written, in {context!r}")

    missing = [title for title in SECTION_TITLES if f"## {title}" not in body]
    if missing:
        raise ValueError(f"summary.md is missing {len(missing)} section(s): {missing}")

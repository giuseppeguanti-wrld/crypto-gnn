# PLANNING — Studio di caso empirico (Cap. 6)

**Progetto**: `crypto-gnn` / `project-thesis`
**Redatto**: 2026-08-03 · **Inizio lavori**: mer 5 agosto 2026 · **Scadenza reale**: ven 14 agosto 2026

---

## 0. Inquadramento

La parte teorica della tesi (`../latex-thesis/`) è completa: Introduzione, Cap. 1–4 e Conclusioni. Il **Cap. 6 (studio di caso) non è attivato** — `\input` commentato in `main.tex` — e il 2026-08-03 ne è stata fissata l'impostazione: **confronto predittivo GCN vs. VAR vs. naive** (sez. 6.1–6.6).

Il capitolo va svolto come **verifica di un'ipotesi**, non come dimostrazione di un vantaggio. La probabilità che una GCN batta un random walk sui rendimenti giornalieri cripto è bassa: un risultato negativo, ben documentato, è un esito valido; presentato come fallimento, è una tesi debole.

**Perimetro di questo piano**: codice, risultati e figure. La stesura del Cap. 6 e la revisione dei Cap. 1–5/7 (checklist in `../latex-thesis/chapters/06-case-study.tex`) sono un blocco successivo separato.

**Esito atteso**: `results/summary.md` con tutti i numeri, figure PDF vettoriali in `../latex-thesis/figures/`, tabelle LaTeX pronte da includere — così che la stesura del capitolo non richieda di rieseguire nulla.

**Budget**: 5 sprint da 4–5h (mer 5 → mar 11 agosto), più 3 giorni di slittamento (mer 12 → ven 14). Il 15 agosto è Ferragosto: non è un giorno di lavoro disponibile.

---

## 1. Decisioni congelate

Chiuse in sede di pianificazione. Riaprirne una in corsa costa un giorno di slittamento.

| Ambito | Decisione | Motivazione |
|---|---|---|
| Universo | 15 asset: BTC, ETH, BNB, XRP, ADA, SOL, DOGE, DOT, AVAX, LINK, LTC, BCH, XLM, TRX, ETC | Tutti quotati con continuità su Binance da dicembre 2020; niente stablecoin (correlazione ~0, nodo isolato nel grafo) |
| Periodo | 2021-01-01 → 2026-06-30 (2007 giorni) | Copre Terra/Luna (2022-05) e FTX (2022-11) con ampio pre-periodo |
| Fonte | Binance REST `/api/v3/klines`, coppie `/USDT`, intervallo `1d` | Endpoint pubblico senza API key, chiusura uniforme 00:00 UTC, provenienza citabile in tesi |
| Finestra correlazione | $T_w = 60$ giorni, passo 1 | $q = N/T_w = 0{,}25$ → bordo Marchenko–Pastur superiore $(1+\sqrt q)^2 = 2{,}25$ |
| Pesi | Mantegna: $d_{ij}=\sqrt{2(1-\rho_{ij})}$, $w_{ij}=1-d_{ij}/2 \in [0,1]$ | Garantisce $w \ge 0$, quindi la semidefinita positività del Laplaciano (`prop:psd`) |
| Soglia $\tau$ | Calibrata su null di permutazione, **fissa** per tutto lo studio | Una soglia fissa rende la densità confrontabile tra periodi diversi |
| Task | Regressione di $r_{t+1}$; accuratezza direzionale derivata dal segno | Un solo modello, due metriche (la sez. 5.4 della tesi chiede metriche multiple) |
| Architettura | GCN a 2 layer in **PyTorch puro** (niente PyTorch Geometric) | Su $N=15$ PyG non offre nulla; il layer è `Â @ H @ W`, cioè `eq:gcn` senza mediazioni, e l'installazione su Windows è il principale rischio di ambiente evitabile |
| Temporalità | Snapshot indipendenti | Coerente con Cap. 6.3; limite già riconosciuto in `sec:dynamic-graphs-temporal-question` |
| Lingua | Codice, identificatori e docstring in inglese; documenti (PLANNING, README, CLAUDE) in italiano | |

### 1.1 Precisazione metodologica sulla soglia

Il testo del Cap. 3 dice di calibrare $\tau$ "sul bordo di Marchenko–Pastur". La formulazione va resa precisa in fase di stesura: **MP governa lo spettro** della matrice (struttura collettiva), non la singola $\rho_{ij}$. Il piano usa quindi due strumenti distinti, entrambi non arbitrari:

- **Marchenko–Pastur** → in sez. 6.6, per contare gli autovalori fuori dal bulk e la quota di varianza assorbita dal modo di mercato. È l'analisi di `laloux1999noise`, già citata in tesi.
- **Null di permutazione** → per $\tau$. Si mescola ogni serie indipendentemente all'interno della finestra (distrugge la dipendenza incrociata, **preserva le code pesanti** delle marginali), si ricalcolano le 105 correlazioni, si ripete $B$ volte. È robusto ai momenti quarti — applica cioè l'argomento stesso di `sec:correlation-dependence-pitfalls` invece di aggirarlo.

> **Voce da emendare nella checklist di revisione**: `06-case-study.tex` elenca già `sec:correlation-matrix-to-graph` (Cap. 4.3), ma l'istruzione attuale — collegare la giustificazione di $\tau$ alla "soglia calibrata sul bordo di Marchenko-Pastur, eq. `eq:marchenko-pastur`" — va corretta. L'ancoraggio al livello di rumore resta il criterio giusto; è l'attribuzione a quella equazione come sua fonte a essere imprecisa, per la ragione appena esposta.

### 1.2 Separazione grafo-per-metriche / grafo-per-modello

Coerente con `sec:network-structure-crisis-regimes`, che apprezza $\lambda_2$ proprio perché non richiede una soglia:

- **Metriche topologiche** (6.6) → sul grafo **completo** pesato Mantegna, dove tutti i $w>0$: $\lambda_2$, $\bar\rho$, entropia spettrale, quota del modo di mercato, lunghezza MST. Nessuna soglia, e nessun rischio che $\lambda_2 \equiv 0$ per grafo disconnesso nei periodi calmi.
- **Densità** (6.6) → sul grafo **soglia**: è l'unica metrica che per definizione ne dipende.
- **Substrato della GCN** (6.3) → grafo soglia, più self-loop e renormalization trick.

---

## 2. Struttura del repository

```
project-thesis/
├── PLANNING.md                 # questo documento
├── README.md                   # come eseguire la pipeline
├── CLAUDE.md                   # convenzioni stabili (scritto nello sprint 5)
├── requirements.txt
├── pyproject.toml              # solo [tool.pytest] e [tool.ruff]
├── config/
│   ├── default.yaml            # ogni parametro dello studio: unica fonte di verità
│   └── events.yaml             # date degli eventi di crisi
├── data/
│   ├── raw/                    # klines grezze per simbolo (gitignored)
│   └── processed/              # prices/returns/corr (gitignored)
├── src/cryptognn/
│   ├── __init__.py
│   ├── config.py               # dataclass annidate + load_config() + config_hash()
│   ├── paths.py                # costanti di percorso, ensure_dirs() idempotente
│   ├── data/
│   │   ├── download.py         # fetch_klines(), download_universe()
│   │   └── returns.py          # build_price_panel(), validate_panel(), log_returns()
│   ├── graph/
│   │   ├── correlation.py      # rolling_correlation() -> (T, N, N)
│   │   ├── threshold.py        # permutation_null(), calibrate_tau()
│   │   ├── build.py            # mantegna_distance(), mantegna_weights(),
│   │   │                       # apply_threshold(), normalized_adjacency()
│   │   └── metrics.py          # algebraic_connectivity(), graph_density(),
│   │                           # mst_length(), spectral_entropy(),
│   │                           # market_mode_share(), eigs_outside_mp()
│   ├── features.py             # build_node_features(), FoldStandardizer
│   ├── models/
│   │   ├── base.py             # Protocol Forecaster: fit(X, y) / predict(X)
│   │   ├── naive.py            # ZeroForecaster, HistoricalMeanForecaster
│   │   ├── ar.py               # PerAssetARForecaster (statsmodels AutoReg)
│   │   ├── var.py              # VARForecaster (statsmodels VAR)
│   │   └── gcn.py              # GCNLayer, GCN2, GCNForecaster (+ ablazione no-graph)
│   ├── evaluation/
│   │   ├── walkforward.py      # make_folds(), run_walkforward()
│   │   ├── metrics.py          # rmse, mae, directional_accuracy,
│   │   │                       # skill_score, diebold_mariano
│   │   └── backtest.py         # sign_strategy(), sharpe(), max_drawdown()
│   └── viz/
│       ├── style.py            # rcParams matplotlib, palette, flag --usetex
│       ├── topology.py         # figure della sez. 6.6
│       ├── graphs.py           # node-link a layout fisso
│       └── results.py          # figure della sez. 6.5
├── scripts/                    # entry point, da eseguire in ordine
│   ├── 01_download_data.py
│   ├── 02_build_graphs.py
│   ├── 03_topology_analysis.py
│   ├── 04_run_baselines.py
│   ├── 05_run_gcn.py
│   ├── 06_run_backtest.py
│   ├── 07_make_figures.py
│   └── 08_make_tables.py
├── exploration/
│   └── explore.py              # script con celle `# %%` — niente .ipynb nel repo
├── app/
│   └── streamlit_app.py        # sprint 6, senza giorno assegnato
├── results/
│   ├── metrics/                # *.parquet, *.json (committati: sono piccoli)
│   ├── figures/                # PDF, poi copiati in ../latex-thesis/figures/
│   ├── tables/                 # *.tex generati
│   ├── summary.md              # tutti i numeri per la stesura
│   └── run_manifest.json       # commit git, hash config, versioni pacchetti
└── tests/
    ├── test_graph.py
    ├── test_walkforward.py
    └── test_gcn.py
```

**Nota su `.gitignore`**: quello in root copre già i pattern Python, ma non `project-thesis/data/`. Le voci vanno aggiunte **al file in root**, non creandone uno di sottocartella (convenzione stabilita nel `CLAUDE.md` di root). Attenzione: `var/` è già ignorato come *directory* — il modulo `models/var.py` è un file e non viene toccato dalla regola.

---

## 3. Stack tecnologico

Ambiente verificato in locale: **Python 3.13.14**, pip 26.1.2.

```
numpy>=2.1        pandas>=2.2      pyarrow>=17     requests>=2.32
scipy>=1.14       statsmodels>=0.14 networkx>=3.4  torch>=2.6
matplotlib>=3.9   pyyaml>=6.0      tqdm>=4.66      pytest>=8.3
# sprint 6 (aggiunto a requirements.txt in S6.5)
streamlit>=1.39
```

`torch` va installato **CPU-only** (`--index-url https://download.pytorch.org/whl/cpu`): niente CUDA, download di ~200 MB invece di ~2,5 GB.

**Costo computazionale** — $\hat A$ per tutte le finestre occupa $2000\times15\times15$ float32 ≈ 1,8 MB. Un addestramento GCN completo è un `bmm` su tensori $[365,15,15]\times[365,15,8]$: sotto il secondo. L'intera griglia (4 configurazioni × 24 fold × 5 semi = 480 run) resta nell'ordine dei minuti. **Il calcolo non è mai il collo di bottiglia**: non serve GPU e non serve ottimizzare nulla.

---

## 4. Milestone

| # | Milestone | Sprint | Definition of Done |
|---|---|---|---|
| **M1** | Dati riproducibili | 1 | `returns.parquet` (2006×15, zero NaN) e `corr_60.npy` rigenerabili da zero con un comando |
| **M2** | **Sez. 6.6 chiusa** | 2 | Metriche topologiche complete e 4 figure prodotte. Da qui il progetto ha un risultato qualunque cosa accada dopo |
| **M3** | Metro di giudizio fissato | 3 | Walk-forward con test anti-look-ahead verdi, 4 baseline eseguite, metriche salvate |
| **M4** | **Sez. 6.5 chiusa** | 4 | GCN e ablazione senza grafo sulla stessa griglia; tabella comparativa con test di Diebold–Mariano |
| **M5** | Pacchetto per la stesura | 5 | Figure vettoriali in `../latex-thesis/figures/`, tabelle `.tex`, `summary.md`, manifest di run |
| **M6** | Visualizzazione interattiva | 6 (senza giorno fisso) | App Streamlit che legge gli artefatti già prodotti e mostra il grafo evolvere nel tempo, riusando le stesse funzioni di disegno delle figure della tesi |

L'ordine è deliberato: **M2 precede M3 e M4**. Dallo sprint 2 esiste già un risultato pubblicabile, così l'esito del confronto predittivo non è mai un rischio esistenziale per il capitolo.

M6 non ha un giorno assegnato: cadrà con ogni probabilità dopo i 5 sprint pianificati. Non è però facoltativo — è l'unico artefatto che mostra il grafo *evolvere*, cosa che nessuna figura statica può fare, ed è quanto serve in sede di discussione.

---

## Sprint 0 — Pre-volo (~15 min, nessun codice)

- [ ] Mail ai relatori, in coda alle domande già pendenti (motore bibliografico, pagine bianche `openright`): una riga sul Cap. 6 — confronto GCN vs. VAR vs. naive, impostato come *verifica di un'ipotesi*, con un risultato negativo ben documentato trattato come esito valido.
  - **Perché ora**: dal 10 agosto sono presumibilmente irraggiungibili. È l'unico rischio del progetto che non si può risolvere scrivendo codice.

---

## Sprint 1 — mer 5 agosto · Fondamenta e dati (4–5h) → M1

**Obiettivo**: da zero a $\{C_t\}$ su disco, rigenerabile con un comando.

### S1.1 Ambiente (30')

- [ ] `python -m venv .venv` dentro `project-thesis/`; attivare con `.venv\Scripts\Activate.ps1`
- [ ] Scrivere `requirements.txt` con le versioni della sez. 3
- [ ] `pip install -r requirements.txt`, poi torch dall'index CPU dedicato
- [ ] Verifica: `python -c "import torch, statsmodels, networkx; print(torch.__version__)"`
- [ ] Aggiungere al `.gitignore` **in root**: `project-thesis/data/`, `project-thesis/.venv/`, `project-thesis/results/figures/*.png`

### S1.2 Scheletro e configurazione (40')

- [ ] Creare l'albero di cartelle della sez. 2, con gli `__init__.py`
- [ ] `config/default.yaml` con **tutti** i parametri dello studio:

```yaml
data:
  source: binance
  quote: USDT
  interval: 1d
  start: 2021-01-01
  end: 2026-06-30
  symbols: [BTC, ETH, BNB, XRP, ADA, SOL, DOGE, DOT, AVAX,
            LINK, LTC, BCH, XLM, TRX, ETC]
graph:
  window: 60
  weight: mantegna
  self_loops: true
  threshold:
    method: permutation
    alpha: 0.05
    n_permutations: 500
    n_calibration_windows: 24
    statistic: pooled
features:
  lags: 5
  vol_windows: [5, 20]
  use_volume: true
walkforward:
  train: 365
  val: 63
  test: 63
  step: 63
  mode: rolling
model:
  gcn:
    hidden: [16, 32]
    dropout: [0.2, 0.5]
    lr: 0.005
    weight_decay: 5.0e-4
    epochs: 300
    patience: 30
    seeds: [0, 1, 2, 3, 4]
  var: {max_lag: 5, ic: bic}
  ar:  {max_lag: 5, ic: bic}
backtest:
  cost_bps: 10
seed: 42
```

- [ ] `src/cryptognn/config.py`: dataclass annidate, `load_config(path) -> Config`, `config_hash()` (SHA-1 dello YAML normalizzato) per il manifest
- [ ] `src/cryptognn/paths.py`: costanti `ROOT`, `DATA_RAW`, `DATA_PROCESSED`, `RESULTS`, `FIGURES`; funzione `ensure_dirs()`

> **La griglia si congela qui**: `hidden × dropout` = **4 configurazioni**. Il numero va riportato in tesi (cfr. sez. 5.4, `white2000reality`). Non si tocca dopo lo sprint 4.

### S1.3 Download (60')

- [ ] `data/download.py::fetch_klines(symbol, start, end, interval)`:
  - `GET https://api.binance.com/api/v3/klines`, `limit=1000`
  - paginazione su `startTime` finché l'ultima candela precede `end`; `time.sleep(0.15)` tra le chiamate
  - ritorna un DataFrame indicizzato su `open_time` (UTC, normalizzato a mezzanotte) con colonne `close`, `volume`, `quote_volume`, `trades`
  - cache: se `data/raw/{SYMBOL}USDT_1d.parquet` esiste e copre il range richiesto, non riscarica
- [ ] `download_universe(config)`: itera i 15 simboli con barra `tqdm`
- [x] `scripts/01_download_data.py` con CLI `--config`, `--force`
- [ ] **Contingenza** (time-box 20', poi si passa al piano B): se Binance risponde 451/403, sostituire con `yfinance` sui ticker `BTC-USD` ecc., mantenendo la stessa firma di ritorno — il resto della pipeline non cambia
  - **NOT IMPLEMENTED (Sprint 1)**: Binance è stato correttamente raggiungibile in tutti i test. La contingenza rimane scrivibile in ~20' se il bisogno si presenta (geo-blocking, regione bloccata, ecc.); è catalogato come "ready to implement" piuttosto che "urgent". Da attivare solo se necessario.

### S1.4 Panel e validazione (45')

- [ ] `data/returns.py::build_price_panel(raw_dir, symbols)` → DataFrame `(2007, 15)` di prezzi di chiusura, indice di date UTC
- [ ] `validate_panel(df)`, che **solleva** invece di limitarsi a segnalare:
  - indice giornaliero continuo senza buchi (le cripto non hanno festivi: il confronto con `pd.date_range` deve combaciare esattamente)
  - zero NaN dopo l'allineamento; se un simbolo ne ha, riportare quali date e **rivedere l'universo**, mai interpolare
  - nessun prezzo $\le 0$; nessuna barra a volume nullo
  - primo e ultimo giorno coincidenti con `config.data.start/end`
- [ ] `log_returns(prices)` → $r_t = \log P_t - \log P_{t-1}$, prima riga eliminata → `(2006, 15)`
- **Salvataggio di `data/processed/prices.parquet` e `returns.parquet` DEFERRED a S1.6**: il salvataggio è accoppiato all'esecuzione del pipeline orchestrato (`scripts/02_build_graphs.py --corr-only` in S1.6) — verrà fatto insieme con il salvataggio di `corr_60.npy` e `corr_index.npy`. Le funzioni `build_price_panel()`, `validate_panel()`, `log_returns()` sono complete e pronte; lo script di orchestrazione che le chiama e salva i risultati appartiene a S1.6.

### S1.5 Fatti stilizzati (45')

Serve sia come controllo di sanità dei dati sia come materiale per la sez. 6.1.

- [ ] Per asset: media, deviazione standard annualizzata, **curtosi**, asimmetria → `results/metrics/descriptive.parquet`
- [ ] ACF di $r_t$ e di $|r_t|$ fino a 30 ritardi
- [ ] Asserzioni attese (se falliscono, i dati sono sbagliati): curtosi > 3 su tutti gli asset; ACF(1) di $r$ prossima a zero; ACF di $|r|$ nettamente positiva e a decadimento lento
- [ ] Test di Ljung–Box su $r$ e $|r|$ (`statsmodels.stats.diagnostic.acorr_ljungbox`)

### S1.6 Correlazioni rolling (30')

- [ ] `graph/correlation.py::rolling_correlation(returns, window)` → `np.ndarray (T-window+1, N, N)` in float32
  - implementazione vettorizzata su `sliding_window_view`, **non** un ciclo `df.rolling().corr()` (circa 100× più lento)
  - `corr_index`: array di date in cui l'elemento $k$ è la data **finale** della finestra — è il perno di tutta la difesa anti-look-ahead
- [ ] Asserzioni: diagonale $\equiv 1$ entro 1e-6, simmetria, valori in $[-1,1]$
- [ ] Salvare `data/processed/corr_60.npy` e `corr_index.npy`

### S1.7 Chiusura (20')

- [ ] `pytest tests/` (ancora vuoto: serve a verificare che il runner parta)
- [ ] Commit: `feat: data pipeline and rolling correlation`

**DoD M1** — cancellando `data/` ed eseguendo `01_download_data.py` e `02_build_graphs.py --corr-only` si ricostruisce tutto senza alcun intervento manuale.

---

## Sprint 2 — gio 6 agosto · Grafo dinamico e topologia (4–5h) → M2

### S2.1 Calibrazione di $\tau$ (50')

- [ ] `graph/threshold.py::permutation_null(returns_window, n_perm, rng)`:
  - a ogni permutazione, `rng.permutation` applicata **indipendentemente per colonna** (preserva le marginali, distrugge la dipendenza incrociata)
  - ritorna sia le 105 correlazioni pooled sia il massimo per permutazione
- [ ] `calibrate_tau(returns, config)`: 24 finestre equispaziate sul periodo → quantile $1-\alpha$ della distribuzione pooled → mediana tra le finestre → **$\tau$ unico per tutto lo studio**
- [ ] Calcolare e salvare anche due soglie di robustezza: `tau_fwer` (quantile del massimo, controllo family-wise sulle 105 coppie) e `tau_fixed = 0.30`
- [ ] `results/metrics/tau_calibration.json`: valore, $\alpha$, $B$, finestre usate, le tre soglie
- [ ] **Valore atteso**: $\tau \approx 0{,}20$–$0{,}28$ con $T_w=60$. Se esce ben oltre $0{,}4$, c'è un errore nella permutazione

### S2.2 Costruzione del grafo (45')

- [ ] `graph/build.py`:
  - `mantegna_distance(C)` → $\sqrt{2(1-C)}$, con `np.clip(C, -1, 1)` a monte
  - `mantegna_weights(C)` → $1 - d/2$, con asserzione $w \in [0,1]$
  - `apply_threshold(C, W, tau)` → azzera $w_{ij}$ dove $\rho_{ij} \le \tau$. Soglia su $\rho$ **con segno**, non su $|\rho|$: un asset fortemente anticorrelato ha già $w \approx 0$ per costruzione, quindi tenerlo come arco di peso nullo non aggiungerebbe nulla. Scelta da dichiarare esplicitamente in sez. 6.2
  - `normalized_adjacency(W)` → $\tilde A = W + I$, $\hat A = \tilde D^{-1/2}\tilde A\tilde D^{-1/2}$ (`eq:renormalization`)
- [ ] `scripts/02_build_graphs.py` → `data/processed/W_full.npy`, `W_thresh.npy`, `A_hat.npy`

### S2.3 Metriche topologiche (60')

In `graph/metrics.py`, tutte con firma `(W | C) -> float`, applicate finestra per finestra:

- [ ] `mean_correlation(C)` — media del triangolo superiore
- [ ] `graph_density(W_thresh)` — archi non nulli su $\binom{15}{2}=105$ ← **unica** calcolata sul grafo soglia
- [ ] `algebraic_connectivity(W_full)` — $\lambda_2$ di $L_{\mathrm{sym}}$ via `scipy.linalg.eigvalsh`, sul grafo **completo**
- [ ] `mst_length(D_mantegna)` — `networkx.minimum_spanning_tree`, somma dei pesi diviso $(N-1)$: è la metrica di `onnela2003dynamics`
- [ ] `spectral_entropy(C)` — $-\sum_k \tilde\lambda_k\log\tilde\lambda_k$ con $\tilde\lambda = \lambda/\sum\lambda$
- [ ] `market_mode_share(C)` — $\lambda_{\max}/\sum\lambda$
- [ ] `eigs_outside_mp(C, q)` — conteggio degli autovalori oltre $(1+\sqrt q)^2$
- [ ] `scripts/03_topology_analysis.py` → `results/metrics/topology.parquet` (1947 righe × 7 colonne)

### S2.4 Analisi degli eventi (30')

- [ ] `config/events.yaml` con `2021-05-19` (crollo legato alle restrizioni cinesi), `2022-05-09` (Terra/Luna), `2022-11-08` (FTX). **Solo eventi documentabili con citazione**: nessun evento aggiunto per completare visivamente il quadro
- [ ] Per ciascun evento, su finestra ±60 giorni: valore delle metriche a −30/0/+30 giorni, variazione percentuale, e percentile del valore rispetto all'intera distribuzione storica → `results/metrics/event_study.parquet`

### S2.5 Figure (60')

Prima di tutto `viz/style.py`: `rcParams` serif, dimensioni in pollici calibrate su `\textwidth` (≈ 4,8"), `savefig(bbox_inches='tight')`, flag `--usetex` con **default `False`** in sviluppo (MiKTeX è lento e fragile se invocato in ciclo) e `True` solo nel run finale.

- [ ] `fig_topology_timeseries.pdf` — 4 pannelli con asse x condiviso ($\bar\rho$, densità, $\lambda_2$, lunghezza MST), `axvline` tratteggiate sugli eventi con etichetta. **È la figura centrale della sez. 6.6**
- [ ] `fig_correlation_heatmaps.pdf` — 3 heatmap affiancate (periodo calmo / Terra-Luna / FTX), scala colore condivisa, asset ordinati secondo un clustering gerarchico calcolato **una volta sola** e riusato in tutte e tre
- [ ] `fig_graph_snapshots.pdf` — 2 diagrammi node-link (calmo vs. crisi). **Layout calcolato una volta sola** (`nx.spring_layout(seed=42)` sulla matrice $W$ media dell'intero periodo) e riusato identico: senza questo accorgimento il compattamento è invisibile perché i nodi si spostano comunque. Spessore degli archi ∝ $w$, dimensione dei nodi ∝ grado pesato
- [ ] `fig_mp_spectrum.pdf` — istogramma degli autovalori contro la densità teorica di Marchenko–Pastur, calmo vs. crisi, con il bordo $(1+\sqrt q)^2$ marcato

### S2.6 Chiusura (15')

- [ ] Commit: `feat: dynamic graph construction and topological metrics`

**DoD M2** — le 4 figure esistono, `topology.parquet` è completo, e la sez. 6.6 è scrivibile senza aggiungere altro codice.

---

## Sprint 3 — ven 7 agosto · Protocollo e baseline (4–5h) → M3

> Le baseline si scrivono **prima** del modello: si fissa il metro di giudizio quando non c'è ancora un modello da difendere.

### S3.1 Walk-forward (60')

- [ ] `evaluation/walkforward.py::make_folds(n_obs, train, val, test, step, offset)` → lista di `Fold(train_idx, val_idx, test_idx)`
  - `offset = window - 1 = 59`: prima della sessantesima osservazione non esiste alcun grafo
  - con 2006 rendimenti: **24 fold**, primo test a partire dal 2022-05-03 (indice ≈ 487)
  - la scelta `train=365` invece di 504 è deliberata: sposta il primo fold di test **prima** di Terra/Luna, così entrambe le crisi cadono nel periodo di test e diventa possibile l'incrocio tra sez. 6.5 e 6.6
- [ ] `run_walkforward(forecaster_factory, data, folds)` → DataFrame in formato lungo `(fold, date, asset, y_true, y_pred, model)`

### S3.2 Test anti-look-ahead (40')

**È il rischio più costoso del progetto**: un errore qui invalida tutti i risultati a valle e si scopre tardi. In `tests/test_walkforward.py`:

- [ ] `test_fold_ordering` — per ogni fold vale `max(train) < min(val) < min(test)`
- [ ] `test_graph_precedes_target` — il grafo usato per prevedere $r_{t+1}$ è costruito su $[t-59,\,t]$, mai oltre $t$
- [ ] `test_standardizer_train_only` — un `FoldStandardizer` fittato sul train produce le stesse statistiche anche alterando a caso le righe di test
- [ ] `test_no_target_leak_in_features` — le feature all'istante $t$ non contengono $r_{t+1}$ (verifica per corruzione: alterare `y` non deve modificare `X`)

### S3.3 Feature dei nodi (45')

- [ ] `features.py::build_node_features(returns, volumes, config)` → tensore `(T, N, F)`:
  - 5 rendimenti ritardati, $r_t \dots r_{t-4}$
  - volatilità realizzata a 5 e a 20 giorni
  - z-score del log-volume su 20 giorni
  - → **F = 8**
- [ ] `FoldStandardizer`: `fit(X_train)` salva μ e σ **per asset** usando solo il train; `transform()` le applica a train, validazione e test. È esattamente la variante di look-ahead che `sec:evaluation-metrics-methodological-pitfalls` indica come "la più insidiosa"

### S3.4 Baseline (60')

- [ ] `models/naive.py::ZeroForecaster` — $\hat r_{t+1}=0$. È il vero avversario: con rendimenti a media quasi nulla ottiene già un RMSE prossimo all'ottimo
- [ ] `models/naive.py::HistoricalMeanForecaster` — media del train, per asset
- [ ] `models/ar.py::PerAssetARForecaster` — `statsmodels.tsa.ar_model.AutoReg`, ordine $p$ scelto per BIC su `max_lag=5`, **una stima per asset**. È il *modello univariato* letterale della prima questione di ricerca
- [ ] `models/var.py::VARForecaster` — `statsmodels.tsa.api.VAR`, $p$ per BIC. Registrare il $p$ scelto e il numero di parametri per ogni fold: con $N=15$ e $p=5$ sono 1140 coefficienti stimati da 365 osservazioni, cioè la dimostrazione empirica dell'argomento di `sec:var-baseline` — va in tabella
- [ ] Tutti conformi al `Protocol Forecaster` definito in `models/base.py`

### S3.5 Metriche (45')

In `evaluation/metrics.py`:

- [ ] `rmse`, `mae`, `directional_accuracy` (sul segno, escludendo gli $y=0$ esatti)
- [ ] `skill_score(y, pred, baseline_pred)` $= 1 - \mathrm{MSE}_{\text{modello}}/\mathrm{MSE}_{\text{base}}$ rispetto a `ZeroForecaster`: rende diretta la domanda "batte il naive?"
- [ ] `diebold_mariano(e1, e2, h=1)` — varianza HAC di Newey–West con correzione campionaria di Harvey–Leybourne–Newbold; ritorna statistica e p-value. È lo strumento che trasforma un "0,3% meglio" in un'affermazione difendibile
- [ ] `scripts/04_run_baselines.py` → `results/metrics/predictions_baselines.parquet`

> **Checkpoint go/no-go (fine sprint 3)**: se l'harness e le 4 baseline non girano, **è qui che si attinge ai giorni di slittamento**, non allo sprint 5.

---

## Sprint 4 — sab 8 / lun 10 agosto · GCN (4–5h) → M4

### S4.1 Implementazione (60')

- [ ] `models/gcn.py::GCNLayer(nn.Module)` — `forward(A_hat, H)` calcola `A_hat @ (H @ self.W) + self.b`, con `A_hat` batched `(B,N,N)` e `H` `(B,N,F)` via `torch.bmm`
- [ ] `GCN2` — `Dropout → GCNLayer(F,h) → ReLU → Dropout → GCNLayer(h,1)`: è letteralmente `eq:gcn` applicata due volte, e come tale citabile in sez. 6.3
- [ ] Flag `use_graph`: se `False`, sostituisce `A_hat` con l'identità → **ablazione senza grafo, a parità di capacità e di feature**. È la versione più pulita della prima questione di ricerca, perché isola il contributo del *grafo* e non quello delle feature
- [ ] `tests/test_gcn.py`:
  - `test_permutation_equivariance` — permutando i nodi, l'output si permuta in modo identico: verifica sperimentale della proprietà dimostrata nel Cap. 2
  - `test_renormalized_spectrum` — autovalori di $\hat A$ contenuti in $[-1,1]$
  - `test_output_shape`

### S4.2 Addestramento (60')

- [ ] `GCNForecaster.fit(X_train, y_train, X_val, y_val)`: Adam, perdita MSE, early stopping sull'MSE di validazione con `patience=30` e ripristino dei pesi migliori
- [ ] Determinismo: `torch.manual_seed`, `np.random.default_rng`, `torch.use_deterministic_algorithms(True)`
- [ ] Log per fold: epoche effettive, MSE finale su train e validazione → `results/metrics/gcn_training_log.parquet`

### S4.3 Esecuzione della griglia congelata (45')

- [ ] Per ogni fold: 4 configurazioni × 5 semi; selezione della configurazione **sulla validazione interna al fold**, mai sul test
- [ ] Previsione di test = **media sui 5 semi** della configurazione selezionata (riduce la varianza da inizializzazione; da dichiarare in tesi)
- [ ] Stessa identica procedura per l'ablazione `use_graph=False`
- [ ] `scripts/05_run_gcn.py` → `results/metrics/predictions_gcn.parquet`

### S4.4 Tabelle comparative (45')

- [ ] Aggregazione di RMSE, MAE, accuratezza direzionale e skill score per modello: complessivi, per asset e per fold
- [ ] Matrice dei test di Diebold–Mariano tra tutte le coppie di modelli
- [ ] `results/metrics/results_main.parquet`, più una stampa a schermo per lettura immediata

> **Regola dello sprint**: qualunque sia il risultato, si passa allo sprint 5. Nessun ritocco a $\tau$, alla finestra, alle feature o alla griglia dopo aver visto il test. Questa riga è la difesa contro il rischio numero uno del progetto.

---

## Sprint 5 — dom 9 / mar 11 agosto · Consolidamento (4–5h) → M5

### S5.1 Metriche economiche (60')

- [x] `evaluation/backtest.py::sign_strategy(predictions, cost_bps)` — posizione pari al segno della previsione, equipesata sui 15 asset, con costo applicato **a ogni cambio di posizione**
  - **Firma senza `returns`**: la tabella lunga porta già `y_true` accanto a `y_pred` sulla stessa riga. Un secondo argomento potrebbe solo introdurre un disallineamento, e un disallineamento qui non produce un errore ma una curva plausibile.
- [x] `sharpe(r, periods=365)`, `max_drawdown(equity)`, rendimento cumulato
- [x] Eseguire per ogni modello a **0 e a 10 bps**: se il vantaggio sparisce a 10 bps, è un risultato da riportare, non da nascondere (`sec:evaluation-metrics-methodological-pitfalls`: "un backtest senza costi di transazione dichiarati non è confrontabile con nulla")
- [x] **Aggiunto rispetto al piano**: riga di riferimento `buy-and-hold` equipesata, e conversione dei log-rendimenti in rendimenti semplici (`expm1`) prima di pesarli — una posizione corta rende $-(e^r-1)$, non $-r$
- [x] **Aggiunto rispetto al piano**: `scripts/06_run_backtest.py`, con rinumerazione di figure (07) e tabelle (08). Il backtest è un passo di calcolo e i suoi artefatti sono letti sia dalle figure di S5.2 sia dalle tabelle di S5.3, quindi deve precedere entrambi

### S5.2 Figure dei risultati (60')

- [x] `fig_walkforward_scheme.pdf` — diagramma del protocollo, con barre train/validazione/test per ciascun fold
- [x] `fig_results_by_fold.pdf` — skill score per fold e per modello, con la linea dello zero evidenziata
- [x] `fig_equity_curves.pdf` — curve cumulate, con i costi dichiarati in didascalia
- [x] `fig_density_vs_error.pdf` — **scatter tra densità media del grafo nel fold ed errore della GCN nel fold**, con retta di regressione e $\rho$ di Spearman. È la verifica empirica diretta della tensione centrale della tesi (`sec:summary-research-questions`, "il punto di incontro delle due questioni"): la struttura è più informativa proprio quando è più difficile da sfruttare. **Da produrre anche se il risultato è nullo** — un'assenza di relazione è essa stessa una risposta
  - **Esito**: nullo, e riportato. $\rho = -0{,}100$ ($p = 0{,}643$) sulla soglia calibrata, $\rho = 0{,}005$ ($p = 0{,}981$) su quella FWER, con 8 fold su 24 a densità esattamente 1. Lo stesso `rank_association()` è chiamato da `viz/figures.py` e da `summary.py`, così il numero della prosa non può divergere da quello annotato sulla figura

### S5.3 Output per la tesi (60')

- [x] `scripts/08_make_tables.py` → file `.tex` con `booktabs` (`\toprule`/`\midrule`/`\bottomrule`), coerenti con `tab:architectures-comparison` e `tab:graph-construction` già presenti in tesi:
  - `tab_universe.tex` — asset, periodo, statistiche descrittive
  - `tab_graph_params.tex` — $T_w$, $q$, bordo MP, $\tau$ e le due soglie di robustezza
  - `tab_results_main.tex` — RMSE / MAE / accuratezza direzionale / skill per modello, con marcatura della significatività DM
  - `tab_backtest.tex` — Sharpe, massimo drawdown, rendimento cumulato a 0 e 10 bps
- [x] **Aggiunto rispetto al piano**: `tab_models.tex`, quinta tabella. La sez. 6.4 argomenta sulla dimensionalità del VAR — 1140 coefficienti da un pannello 365×15, sotto cinque osservazioni per parametro — e sul fatto che il BIC seleziona ordine 0 su ogni fold. Entrambi sono numeri, e un numero che regge un argomento va in tabella invece che solo in prosa
- [x] Rigenerare **tutte** le figure con `--usetex`, per far combaciare i font con quelli del documento
- [x] Copiare i PDF in `../latex-thesis/figures/`
- [x] `results/run_manifest.json`: commit git, `config_hash`, `pip freeze`, timestamp, durate
  - **Divergenza deliberata, argomentata in `08_make_tables.py::build_manifest`**: al posto di un `pip freeze` completo il manifest registra le versioni degli 8 pacchetti che possono cambiare un numero (una lista di 39 pin transitivi li seppellirebbe; `requirements.txt` resta il lockfile esatto), e **non** registra le durate — non sono recuperabili a posteriori da uno script che non ha eseguito la pipeline, e strumentare gli altri sette perché si cronometrino è un'altra modifica. Al loro posto ci sono i digest SHA-256 di ogni artefatto, che è ciò di cui il criterio di riproducibilità ha effettivamente bisogno; le date di modifica danno comunque l'ordine in cui i passi sono girati

### S5.4 Pacchetto per la stesura (45')

- [x] `results/summary.md` — ogni numero che servirà nel Cap. 6, organizzato sezione per sezione (6.1 … 6.6), così che la stesura non richieda di rieseguire nulla
  - **Divergenza deliberata**: il documento è **generato** da `src/cryptognn/summary.py`, invocato da `08_make_tables.py`, non redatto a mano. È la stessa regola già applicata alle tabelle — un numero trascritto smette di essere d'accordo con la sua fonte al primo rerun — e qui la superficie di trascrizione sarebbe la più grande del progetto. Conseguenza vincolante: il file non contiene alcun timestamp, così due rigenerazioni a monte invariato sono byte-identiche e `run_manifest.json` può registrare onestamente un albero pulito
  - Nessun nono script: la pipeline resta a 8 passi e il manifest resta l'ultima cosa scritta, dato che descrive tutte le altre
- [x] Elenco dei **limiti emersi in corso d'opera** da riportare in `sec:limitations`: survivorship bias dell'universo, singolo periodo, singola fonte dati, snapshot indipendenti
  - Ai quattro previsti se ne aggiungono quattro emersi dai numeri, tutti nella sezione «Limiti da dichiarare» di `summary.md`: la densità media di 0,974 alla soglia calibrata (il grafo **non è rado**, e la sogliatura non è il passo selettivo che il nome suggerisce); il BIC che seleziona ordine 0 su ogni fold, per cui il VAR selezionato coincide numericamente con la media storica; skill score negativo per tutti i modelli, con DM non significativo per la GCN dopo Holm; e il vantaggio del backtest che svanisce a 10 bps, con l'ablazione senza grafo davanti alla GCN
- [x] `README.md` — prerequisiti, installazione, i 7 script nell'ordine di esecuzione, tempi attesi
  - Sono **8**: il conteggio del piano è anteriore a `06_run_backtest.py` (cfr. S5.1)
- [x] `CLAUDE.md` di `project-thesis/` — convenzioni stabili: la configurazione è l'unica fonte di verità, mai numeri magici nel codice, ogni figura prodotta da script, griglia congelata
- [x] Commit finale e tag `v1.0-results`

---

## Sprint 6 — senza giorno assegnato · Visualizzazione interattiva (3–4h) → M6

Cadrà quasi certamente dopo i 5 sprint pianificati, ma **non è opzionale**: è l'unico artefatto che mostra il grafo *evolvere nel tempo*. Una figura statica mostra due istanti; l'app mostra la transizione, ed è quanto serve in sede di discussione. Serve inoltre da strumento di ispezione quando un numero della sez. 6.6 sembra sospetto.

> **Vincolo architetturale, da rispettare già nello sprint 2.** Le funzioni di disegno in `viz/` devono accettare un `ax: matplotlib.axes.Axes` come parametro e **non chiamare `savefig()` al loro interno**: sono gli script `07_make_figures.py` a comporre la figura e a salvarla. Senza questa disciplina, lo sprint 6 è costretto a duplicare il codice di disegno e l'app finisce per mostrare qualcosa di diverso dalle figure stampate in tesi — che è il modo più rapido di rendere entrambe inaffidabili.

### S6.1 Impalcatura e caching (45')

- [ ] `app/streamlit_app.py`, con `st.set_page_config(layout="wide", page_title="Grafo di correlazione cripto")`
- [ ] Loader con cache, che leggono **solo artefatti già prodotti** (`data/processed/`, `results/metrics/`):
  - `@st.cache_data load_returns()`, `load_corr(window)`, `load_topology()`, `load_tau()`
  - `@st.cache_resource load_layout()` → posizioni dei nodi calcolate una volta sola sulla $W$ media, identiche a quelle di `fig_graph_snapshots.pdf`
- [ ] Guardia iniziale: se un artefatto manca, `st.error()` con il comando esatto da eseguire (`python scripts/02_build_graphs.py`) invece di un traceback
- [ ] **Principio**: l'app non ricalcola mai la pipeline. L'unica eccezione è la soglia — cambiare $\tau$ è un confronto NumPy su una matrice $15\times15$, quindi istantaneo. È anche il controllo didatticamente più utile, perché mostra dal vivo quanto la densità dipenda dalla soglia, che è l'obiezione mossa a questa metrica in `sec:network-structure-crisis-regimes`

### S6.2 Controlli in barra laterale (30')

- [ ] `st.select_slider` sulla data, con etichette formattate `YYYY-MM-DD` (non l'indice numerico)
- [ ] Scelta della soglia via `st.radio`: `calibrata` (default, dal JSON dello sprint 2) · `FWER` · `manuale`; se manuale, `st.slider` in $[0{,}0,\,0{,}9]$ passo $0{,}01$, con il valore calibrato indicato in didascalia come riferimento
- [ ] `st.selectbox` su $T_w$ tra le finestre precalcolate (60 di default; 30 e 90 se generate come analisi di sensibilità)
- [ ] `st.checkbox("Layout fisso", value=True)` — disattivandolo si vede *perché* il layout fisso è necessario
- [ ] `st.checkbox("Confronta due date")` — divide la colonna del grafo in due snapshot affiancati con due slider indipendenti. È la modalità effettivamente utile in discussione (calmo vs. crisi)
- [ ] `st.download_button` per esportare lo snapshot corrente in PDF

### S6.3 Pannello principale (60')

- [ ] `col_graph, col_heat = st.columns([3, 2])`
- [ ] `col_graph` → node-link alla data selezionata, disegnato da `viz.graphs.draw_snapshot(ax, W_thresh_t, pos, ...)`, **la stessa funzione** usata da `07_make_figures.py`
- [ ] `col_heat` → heatmap di $C_t$ da `viz.topology.draw_heatmap(ax, C_t, order)`, con lo stesso ordinamento gerarchico delle figure e scala colore fissa in $[-1,1]$ (non riadattata alla data, altrimenti i colori non sono confrontabili tra istanti)
- [ ] Riga di `st.metric` sopra le due colonne: densità, $\bar\rho$, $\lambda_2$, lunghezza MST, autovalori fuori dal bulk — ciascuno con `delta` rispetto a 30 giorni prima, così il compattamento si legge come numero e non solo come immagine

### S6.4 Fascia temporale (45')

- [ ] Sotto le colonne, a tutta larghezza: le serie topologiche dello sprint 2 con `axvline` sulla data selezionata, disegnate da `viz.topology.draw_metric_series(ax, df, metric, events)`
- [ ] Marcatori degli eventi letti da `config/events.yaml`, gli stessi di `fig_topology_timeseries.pdf`
- [ ] Lo slider resta l'unica sorgente di verità sulla data selezionata (Streamlit non offre selezione da click su una figura matplotlib): nessun secondo meccanismo di navigazione che possa desincronizzarsi

### S6.5 Rifiniture e chiusura (30')

- [ ] `st.session_state` per conservare data e soglia tra i rerun
- [ ] Nota in calce all'app: $\tau$ calibrata, $T_w$, periodo, numero di asset — così uno screenshot resta autoesplicativo fuori contesto
- [ ] Sezione nel `README.md`: `streamlit run app/streamlit_app.py`, con l'avvertenza che richiede gli artefatti degli sprint 1–2
- [ ] Aggiungere `streamlit>=1.39` a `requirements.txt` (fin qui era commentato come opzionale)
- [ ] Commit: `feat: interactive graph explorer`

**DoD M6** — l'app si avvia da artefatti freschi, mostra il grafo a qualunque data del periodo, e i numeri visualizzati coincidono con `topology.parquet`.

---

## Blocco successivo (fuori dal perimetro di questo piano)

- **Stesura del Cap. 6** (~1,5–2 giorni) e revisione dei Cap. 1–5/7 secondo la checklist in `../latex-thesis/chapters/06-case-study.tex` (~0,5 giorno), integrata con il punto su `sec:correlation-matrix-to-graph` segnalato nella sez. 1.1.

---

## Registro dei rischi

| # | Rischio | Probabilità | Impatto | Mitigazione |
|---|---|---|---|---|
| R1 | Ciclo di ritocchi allo sprint 4 dopo un esito negativo | **Alta** | Consuma tutto lo slittamento | Griglia congelata allo sprint 1 e dichiarata in tesi; regola esplicita di fine sprint 4 |
| R2 | Errore nel walk-forward scoperto tardi | Media | Invalida tutti i risultati | I 4 test anti-look-ahead si scrivono in S3.2, **prima** di generare qualunque risultato |
| R3 | Attrito su ambiente o API allo sprint 1 | Media | Mezza giornata | Piano B `yfinance` a firma identica, con time-box di 20' |
| R4 | Grafo troppo rado o troppo denso dopo la calibrazione | Media | Rilavorazione dello sprint 2 | Tre soglie calcolate insieme (permutazione, FWER, fissa a 0,30); $\lambda_2$ è calcolato sul grafo completo, quindi immune |
| R5 | Un asset con buchi nello storico | Bassa | 1h | `validate_panel` solleva già allo sprint 1; sostituzione dell'asset documentata |
| R6 | `usetex` fragile su MiKTeX | Bassa | 1h | Default `False` in sviluppo, `True` solo in S5.3; ripiego su font serif senza usetex |
| R7 | L'app dello sprint 6 mostra numeri diversi dalle figure della tesi | Media | Entrambe inaffidabili | Vincolo architetturale dichiarato in testa allo sprint 6: funzioni di disegno che accettano un `ax` e non salvano, riusate da app e script; l'app legge artefatti, non ricalcola |

---

## Verifica end-to-end

Da `project-thesis/`, con l'ambiente attivo e partendo da `data/` e `results/` vuote:

```powershell
pytest tests/ -v                             # tutti verdi, in particolare i 4 anti-look-ahead
python scripts/01_download_data.py           # ~2'   -> data/raw/*.parquet
python scripts/02_build_graphs.py            # ~1'   -> W_full, W_thresh, A_hat, tau_calibration.json
python scripts/03_topology_analysis.py       # ~1'   -> topology.parquet, event_study.parquet
python scripts/04_run_baselines.py           # ~3'   -> predictions_baselines.parquet
python scripts/05_run_gcn.py                 # ~10'  -> predictions_gcn.parquet
python scripts/06_run_backtest.py            # ~2"   -> backtest_all.parquet, backtest_curves_all.parquet
python scripts/07_make_figures.py --usetex   # ~3'   -> 8 PDF
python scripts/08_make_tables.py             # ~10"  -> 5 file .tex, summary.md, manifest
streamlit run app/streamlit_app.py           # sprint 6, richiede gli artefatti sopra
```

Criteri di accettazione:

- ogni script è idempotente e rieseguibile senza pulizia manuale
- `results/summary.md` contiene ogni numero citabile nel Cap. 6
- le 8 figure sono in `../latex-thesis/figures/` e `latexmk -pdf -bibtex main.tex` compila senza `Undefined`, `Underfull` o `Overfull`
- `run_manifest.json` permette di riprodurre il run a distanza di mesi

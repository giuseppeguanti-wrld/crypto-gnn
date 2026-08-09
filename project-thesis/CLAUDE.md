# CLAUDE.md — project-thesis

Questo file guida il lavoro del codice sperimentale per il Capitolo 6 (studio di caso) della tesi.

## Cosa è questo progetto

Studio di caso empirico che implementa e confronta tre modelli di previsione multi-asset su 15 criptovalute:

- **GCN** (Graph Convolutional Network) — modello a due layer su grafo di correlazione dinamico
- **VAR** (Vector AutoRegression) — baseline multivariata
- **Naive forecaster** — baseline univariata e zero

Il progetto è strutturato come **verifica di un'ipotesi**, non dimostrazione di un vantaggio: un risultato negativo ben documentato è un esito valido.

## Convenzioni stabili

### Importi e modulo

Tutto il codice applicativo sta in `src/cryptognn/`. Gli import sono sempre:

```python
from cryptognn.config import load_config
from cryptognn.data import download_universe
from cryptognn.graph.metrics import algebraic_connectivity
# etc.
```

**Mai** import da `src.cryptognn` o relativi da root.

Questo funziona perché il pacchetto è **installato in modalità editable**, non per manipolazione di `sys.path`. È un prerequisito una tantum su un clone nuovo o dopo aver ricreato il venv:

```powershell
uv pip install -e . --no-deps      # il venv di questo progetto è gestito da uv
```

`--no-deps` è deliberato: `torch` è installato dall'index CPU dedicato e una nuova risoluzione da PyPI tirerebbe la build CUDA (~2,5 GB invece di ~200 MB). Le dipendenze si installano prima, da `requirements.txt`.

**Mai** aggiungere `sys.path.insert(...)` in cima a uno script: se un import fallisce, manca l'installazione editable, non il path.

### Configurazione come unica fonte di verità

**Ogni parametro dello studio** — universo, periodo, finestra, soglia, iperparametri — vive in `config/default.yaml`. Nel codice **nessun numero magico**:

- ❌ Evita: `window = 60`, `tau = 0.25`, `hidden_size = [16, 32]`
- ✅ Usa: `config.graph.window`, `config.graph.threshold.alpha`, `config.model.gcn.hidden`

Carica la config con `config_hash = load_config('config/default.yaml')` — il suo SHA-1 finisce in `run_manifest.json` per riproducibilità.

### Griglia congelata

Gli iperparametri della griglia (hidden × dropout = 4 configurazioni per GCN) si congelano allo **Sprint 1**. Toccarli dopo Sprint 1 invalida i risultati; riaprirli costa un giorno di slittamento del progetto. Mai ritocchi dopo aver visto il test (regola di fine Sprint 4).

### Ogni figura da script

Tutte le figure sono generate da `scripts/06_make_figures.py` e salvate in `results/figures/` come PDF vettoriali. **Mai figure generate manualmente o salvate ad hoc.** Le funzioni di disegno in `viz/` accettano un `ax: matplotlib.axes.Axes` e **non chiamano `savefig()` al loro interno** — è lo script a comporre e salvare. Questo vincolo architetturale protegge anche Sprint 6 (app Streamlit) da incoerenze.

### Linguaggio

- **Codice, identificatori, docstring**: inglese
- **Commit message, README, CLAUDE**: italiano
- **Documenti rivolti a lettori della tesi** (sez. 6.1–6.6): italiano, ma sono nella tesi (`latex-thesis/`), non qui

### Anti-look-ahead rigoroso

Lo sprint 3 scrive 4 test critici in `tests/test_walkforward.py`:

- `test_fold_ordering` — train < val < test
- `test_graph_precedes_target` — il grafo usato per predire $r_{t+1}$ è costruito su $[t-59, t]$, mai oltre
- `test_standardizer_train_only` — il fit è sul train soltanto
- `test_no_target_leak_in_features` — nessuna feature contiene $r_{t+1}$

Questi test passano **prima** di qualunque risultato. Scoprire un errore qui tardi invalida tutto.

### Commit e branching

- Un commit per milestone completata, non micro-commit per file singoli
- Messaggio: breve imperativo italiano, es. `feat: data pipeline and rolling correlation` (ma in inglese, per coerenza con codice)
- Branch di lavoro: `sprnt1`, `sprnt2`, ecc. — uno per sprint
- Non force-push su main

### Testing

- Ogni modulo ha test in `tests/test_*.py`
- `pytest tests/` deve passare pulitamente prima di qualunque run
- Test anti-look-ahead in `test_walkforward.py` sono **critici** — se falliscono, niente prosegue

## Struttura key

```
project-thesis/
├── config/default.yaml       ← unica fonte di verità
├── src/cryptognn/
│   ├── config.py             ← load_config(), config_hash()
│   ├── paths.py              ← costanti di percorso, ensure_dirs()
│   ├── data/                 ← download, preprocessing
│   ├── graph/                ← correlazioni, grafo, metriche topologiche
│   ├── models/               ← GCN, VAR, naive, AR
│   ├── evaluation/           ← walk-forward, metriche, backtest
│   └── viz/                  ← stile, figure
├── scripts/01-07_*.py        ← entry point, da eseguire in ordine
├── tests/                    ← pytest
└── results/                  ← output (metrics, figures, tables)
```

## Esecuzione della pipeline

```bash
uv pip install -e . --no-deps                # una tantum, vedi "Importi e modulo"
pytest tests/ -v                             # verifica anti-look-ahead
python scripts/01_download_data.py           # ~2'
python scripts/02_build_graphs.py            # ~1'
python scripts/03_topology_analysis.py       # ~1'
python scripts/04_run_baselines.py           # ~3'
python scripts/05_run_gcn.py                 # ~10'
python scripts/06_make_figures.py --usetex   # ~3'
python scripts/07_make_tables.py             # ~10"
streamlit run app/streamlit_app.py           # Sprint 6
```

## Regole per lavorare su questo progetto

1. **Config first**: leggi `config/default.yaml` prima di aggiungere parametri al codice
2. **Test first**: scrivi test anti-look-ahead prima di generare risultati
3. **Figure from script**: ogni figura generata da `scripts/06_make_figures.py`, niente manuale
4. **Determinismo**: `torch.manual_seed()`, `np.random.default_rng()`, `torch.use_deterministic_algorithms(True)` in ogni modello
5. **Riproducibilità**: `run_manifest.json` deve contenere commit git, config_hash, pip freeze, timestamp, durate
6. **Limiti espliciti**: ogni risultato dichiara i limiti (survivorship bias, singolo periodo, snapshot indipendenti, ecc.) pronti per sez. 7.3

Questo file è l'unico documento di convenzioni del progetto: le motivazioni delle scelte non ovvie stanno nei docstring, accanto al codice che le applica, non in un piano separato.

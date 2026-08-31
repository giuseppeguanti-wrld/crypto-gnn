# crypto-gnn — studio di caso empirico (Capitolo 6)

Codice sperimentale della tesi *Graph Neural Network per la previsione multi-asset
di criptovalute su grafi di correlazione dinamici*. Costruisce un grafo di
correlazione a finestra mobile su 15 criptovalute e confronta una **GCN a due
layer** con un **VAR**, un **AR per asset** e due **previsioni naive**, su un
protocollo walk-forward con test anti-look-ahead.

Il capitolo è impostato come **verifica di un'ipotesi**, non come dimostrazione di
un vantaggio. L'esito misurato è negativo — nessun modello batte la previsione
nulla, e l'ablazione senza grafo non è distinguibile dalla GCN — ed è riportato
come tale: `results/summary.md` lo dichiara insieme a tutti i numeri che lo
sostengono.

## Prerequisiti

- **Python 3.13+** (l'ambiente di riferimento è 3.14.4)
- **[uv](https://docs.astral.sh/uv/)** per gestire il virtualenv
- **MiKTeX** (o altra distribuzione LaTeX) solo per `--usetex` in `07_make_figures.py`;
  senza, le figure si generano ugualmente con font serif di matplotlib
- Connessione a internet per il solo primo script: l'endpoint pubblico
  `api.binance.com/api/v3/klines` non richiede API key

Nessuna GPU. L'intera griglia — 4 configurazioni × 5 semi × 24 fold × 2 arm — gira
in minuti su CPU: il calcolo non è mai il collo di bottiglia.

## Installazione

```powershell
cd project-thesis
uv venv
.venv\Scripts\Activate.ps1

# torch dall'index CPU: ~200 MB invece dei ~2,5 GB della build CUDA
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -r requirements.txt

# il pacchetto in editable, senza risolvere di nuovo le dipendenze
uv pip install -e . --no-deps
```

`--no-deps` sull'ultima riga è deliberato: una nuova risoluzione da PyPI
reinstallerebbe `torch` nella variante CUDA, annullando la riga precedente.

Verifica dell'ambiente:

```powershell
python -c "import torch, statsmodels, networkx; print(torch.__version__)"
```

Se un import di `cryptognn` fallisce, manca l'installazione editable — non il
`PYTHONPATH`. Nessuno script manipola `sys.path`.

## Esecuzione della pipeline

Gli script vanno eseguiti in ordine: ognuno consuma gli artefatti del precedente e
si rifiuta di partire se ne manca uno, indicando il comando che lo produce. Sono
tutti idempotenti e rieseguibili senza pulizia manuale.

```powershell
pytest tests/ -q                             # 566 test, inclusi i 4 anti-look-ahead
ruff check src/ tests/ scripts/

python scripts/01_download_data.py           # ~2'    klines Binance -> data/raw/
python scripts/02_build_graphs.py            # ~1'    correlazioni, soglia, grafi
python scripts/03_topology_analysis.py       # ~1'    metriche topologiche, eventi
python scripts/04_run_baselines.py           # ~3'    zero, media, AR, VAR
python scripts/05_run_gcn.py                 # ~10'   GCN e ablazione senza grafo
python scripts/06_run_backtest.py            # ~2"    strategia sul segno, 0 e 10 bps
python scripts/07_make_figures.py --usetex   # ~3'    8 PDF vettoriali
python scripts/08_make_tables.py             # ~10"   5 .tex, summary.md, manifest
```

Ogni script accetta `--config` per puntare a un YAML diverso da
`config/default.yaml`. Flag specifici: `--force` (01, riscarica ignorando la
cache), `--corr-only` (02), `--usetex` (07), `--no-publish` (08, scrive in
`results/` senza copiare nella tesi).

Il primo script scarica ~2 000 candele giornaliere per ciascuno dei 15 simboli e
mette in cache `data/raw/{SYMBOL}USDT_1d.parquet`: una seconda esecuzione non
riscarica nulla.

## Che cosa produce

| Percorso | Contenuto |
|---|---|
| `data/raw/`, `data/processed/` | klines grezze, pannello prezzi/rendimenti/volumi, correlazioni e grafi (gitignored) |
| `results/metrics/` | 20 artefatti `.parquet`/`.json`: descrittive, topologia, studio degli eventi, previsioni, diagnostica per fold, sintesi di accuratezza, matrice Diebold–Mariano, backtest |
| `results/figures/` | 8 PDF vettoriali |
| `results/tables/` | 5 tabelle LaTeX `booktabs` |
| `results/summary.md` | **ogni numero del Cap. 6**, organizzato per sezione, più l'elenco dei limiti per `sec:limitations` |
| `results/run_manifest.json` | commit, stato pulito/sporco dell'albero, hash della configurazione, versioni dei pacchetti, digest SHA-256 di ogni artefatto |
| `../latex-thesis/figures/`, `../latex-thesis/tables/` | copie pubblicate da `08_make_tables.py` |

`results/summary.md` è **generato**: rigenerarlo con `08_make_tables.py`, non
modificarlo a mano. Vale la stessa regola delle tabelle — un numero trascritto
smette di essere d'accordo con la sua fonte al primo rerun.

## Configurazione

`config/default.yaml` è l'**unica fonte di verità** dello studio: universo,
periodo, finestra di correlazione, parametri della soglia, feature, schema
walk-forward, griglia della GCN, costi del backtest, seed. Nel codice non
compaiono numeri magici. `config/events.yaml` contiene le tre date di crisi, ognuna
con la citazione che la documenta.

Lo SHA-1 del file finisce in `run_manifest.json`: due run con lo stesso hash hanno
usato gli stessi parametri.

> La griglia degli iperparametri è **congelata**: 2 valori di `hidden` × 2 di
> `dropout`, fissati prima di osservare qualunque risultato. Modificarli dopo
> aver visto il test invalida il confronto, ed è la ragione per cui il numero è
> dichiarato in tesi.

## Test

```powershell
pytest tests/ -q                                       # tutto
pytest tests/evaluation/test_walkforward.py -q         # i 4 anti-look-ahead
```

L'albero di `tests/` rispecchia quello di `src/cryptognn/`. I quattro test
anti-look-ahead sono critici: verificano che i blocchi di un fold siano ordinati,
che il grafo usato per prevedere $r_{t+1}$ sia costruito su $[t-59, t]$, che lo
standardizzatore sia fittato sul solo train e che nessuna feature contenga il
target. Ognuno è stato provato contro una mutazione iniettata — un test che non
fallisce quando la fuga c'è non protegge nulla.

## Struttura

```
project-thesis/
├── config/default.yaml       unica fonte di verità
├── src/cryptognn/
│   ├── data/                 download e pannello dei rendimenti
│   ├── graph/                correlazioni, soglia, costruzione, metriche
│   ├── models/               naive, AR, VAR, GCN
│   ├── evaluation/           walk-forward, metriche, backtest
│   ├── viz/                  stile e figure
│   ├── tables.py             tabelle LaTeX
│   └── summary.py            results/summary.md
├── scripts/01-08_*.py        entry point, in ordine
├── tests/                    stessa alberatura di src/cryptognn/
└── results/                  metriche, figure, tabelle, summary, manifest
```

`CLAUDE.md` documenta le convenzioni di lavoro sul codice.

## Esploratore interattivo

```powershell
streamlit run app/streamlit_app.py
```

Richiede gli artefatti degli script `01_download_data.py`–`03_topology_analysis.py`
(pannello dei rendimenti, correlazioni, soglia calibrata, metriche topologiche): se
ne manca uno, l'app mostra il comando esatto da eseguire invece di un traceback.
Non ricalcola mai la pipeline — l'unica eccezione è la soglia, un confronto NumPy
istantaneo su una matrice $15\times15$.

Mostra il grafo di correlazione e la heatmap a una data scelta da slider, la
possibilità di confrontare due date affiancate, e la stessa fascia di serie
topologiche di `fig_topology_timeseries.pdf` con gli eventi di crisi marcati. Il
grafo e la heatmap sono disegnati dalle stesse funzioni (`viz.graphs.draw_snapshot`,
`viz.topology.draw_heatmap`) usate da `07_make_figures.py`, cosicché l'app mostra
sempre esattamente ciò che le figure della tesi mostrano.

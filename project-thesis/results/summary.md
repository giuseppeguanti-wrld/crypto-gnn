# Sintesi dei risultati — Capitolo 6

> **Documento generato.** Rigenerare con `python scripts/08_make_tables.py`; una modifica manuale viene sovrascritta al primo rerun.
> Configurazione: `config/default.yaml`, SHA-1 `2ac1abadf9d7360164e6b5803e35d779314b6b77`.
> Provenienza del run — commit, ambiente, digest di ogni artefatto — in `results/run_manifest.json`.

Ogni numero è letto dagli artefatti di `results/metrics/` e formattato come lo stamperà la tesi: virgola decimale, trattino per un valore che non esiste. Un valore copiato da qui nel LaTeX è una copia, non un arrotondamento nuovo.

## 6.1 Dati e fatti stilizzati

- **Universo**: 15 asset, coppie /USDT su Binance, intervallo 1d
- **Periodo**: 01/01/2021 – 30/06/2026
- **Rendimenti logaritmici**: 2 006 osservazioni per asset, dal 02/01/2021 al 30/06/2026

| Asset | Media giorn. | Volatilità ann. | Asimmetria | Curtosi in ecc. |
| --- | ---: | ---: | ---: | ---: |
| BTC | 0,035% | 57,8% | -0,12 | 4,0 |
| ETH | 0,038% | 78,0% | -0,20 | 5,4 |
| BNB | 0,133% | 79,4% | 0,83 | 24,3 |
| XRP | 0,074% | 96,6% | 1,27 | 17,7 |
| ADA | -0,010% | 95,6% | 0,81 | 10,8 |
| SOL | 0,184% | 110,5% | -0,36 | 8,9 |
| DOGE | 0,127% | 136,1% | 6,42 | 134,8 |
| DOT | -0,115% | 97,4% | -0,14 | 8,8 |
| AVAX | 0,029% | 113,5% | 0,29 | 8,3 |
| LINK | -0,025% | 99,5% | -0,37 | 6,0 |
| LTC | -0,055% | 85,4% | -0,75 | 9,4 |
| BCH | -0,027% | 93,1% | 0,64 | 14,2 |
| XLM | 0,018% | 95,5% | 1,73 | 20,9 |
| TRX | 0,123% | 75,7% | 2,10 | 53,7 |
| ETC | 0,010% | 99,3% | 0,66 | 9,6 |

- **Curtosi in eccesso**: positiva su 15 asset su 15, da 4,0 a 134,8
- **ACF(1) dei rendimenti**: media -0,0412 (intervallo -0,1444 – 0,0142)
- **ACF(1) dei rendimenti assoluti**: media 0,2385
- **ACF(30) dei rendimenti assoluti**: media 0,0937, ancora positiva: decadimento lento
- **Ljung–Box a 30 ritardi**: rifiuta al 5% su 14 asset su 15 per i rendimenti, su 15 su 15 per i rendimenti assoluti

Lettura per la sez. 6.1: code pesanti su ogni serie, dipendenza lineare nei rendimenti debole, dipendenza nella volatilità forte e persistente. È il quadro che giustifica il null di permutazione della sez. 6.2 — che preserva le marginali — al posto di un null gaussiano, e che rende plausibile a priori un esito negativo sul livello dei rendimenti.

Tabella corrispondente: `tab_universe.tex`.

## 6.2 Costruzione del grafo dinamico

- **Finestra**: T_w = 60 giorni, passo 1
- **Rapporto di aspetto**: q = N/T_w = 0,250
- **Bordo superiore Marchenko–Pastur**: (1+√q)² = 2,250
- **Coppie di asset**: 105
- **Pesi**: Mantegna: d = √(2(1−ρ)), w = 1 − d/2, con self-loop e renormalization trick
- **Null di calibrazione**: permutazione indipendente per colonna, α = 0,05, B = 500, 24 finestre equispaziate, statistica «pooled», seed 42

| Soglia | Valore | Densità media | Min | Max | Dev. std |
| --- | ---: | ---: | ---: | ---: | ---: |
| τ calibrata | 0,2145 | 0,974 | 0,733 | 1,000 | 0,053 |
| τ FWER | 0,4311 | 0,887 | 0,371 | 1,000 | 0,129 |
| τ fissa | 0,3000 | 0,950 | 0,562 | 1,000 | 0,079 |

Lettura per la sez. 6.2: la soglia calibrata lascia in piedi quasi tutti gli archi. Il grafo su cui la GCN opera non è rado, è quasi completo, e la sogliatura non è il passo selettivo che il nome suggerisce. Va detto nel capitolo prima che sia il lettore a dedurlo dalla densità della sez. 6.6.

Il bordo MP non concorre alla scelta di τ: governa lo spettro della matrice, non la singola correlazione. Serve invece in 6.6 per contare gli autovalori fuori dal bulk.

Tabella corrispondente: `tab_graph_params.tex`.

## 6.3 Architettura e griglia della GCN

- **Architettura**: Dropout → GCNLayer(F, h) → ReLU → Dropout → GCNLayer(h, 1); il layer è Â·H·W + b, cioè `eq:gcn` applicata due volte
- **Feature per nodo**: F = 8 (5 rendimenti ritardati, volatilità realizzata a 5 e 20 giorni, z-score del log-volume a 20 giorni)
- **Ablazione**: `use_graph=False` sostituisce Â con l'identità: stessa capacità, stesse feature, nessun grafo. È l'isolamento del contributo del grafo
- **Griglia congelata**: 4 configurazioni (16/32 unità nascoste × dropout 0,2/0,5), 5 semi, 20 fit per fold e per arm
- **Ottimizzazione**: Adam, lr = 0,0050, weight decay = 0,00050, max 300 epoche, early stopping con pazienza 30 sull'MSE di validazione e ripristino dei pesi migliori
- **Selezione**: configurazione scelta sulla validazione interna al fold, mai sul test; previsione di test = media sui 5 semi

MSE di validazione medio per cella della griglia:

| Unità nascoste | Dropout | GCN | GCN senza grafo |
| ---: | ---: | ---: | ---: |
| 16 | 0,2 | 0,001726 | 0,001745 |
| 16 | 0,5 | 0,001725 | 0,001736 |
| 32 | 0,2 | 0,001725 | 0,001738 |
| 32 | 0,5 | 0,001725 | 0,001733 |

Esito della selezione e costo dell'addestramento:

| Arm | Configurazione più scelta | MSE val. | Epoche | Early stop | Parametri | Secondi |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GCN | h=16, p=0,5 (9/24 fold) | 0,001706 | 66,7 | 100,0% | 248 | 2,57 |
| GCN senza grafo | h=32, p=0,2 (9/24 fold) | 0,001720 | 91,8 | 98,3% | 268 | 3,85 |

Lettura per la sez. 6.3: le quattro celle della griglia sono separate alla sesta cifra decimale. La scelta dell'iperparametro non discrimina, il che è coerente con un segnale assente più che con un modello mal calibrato — e va detto, perché protegge il capitolo dall'obiezione «avete cercato poco».

Tabella corrispondente: `tab_models.tex`.

## 6.4 Protocollo di valutazione

- **Schema**: walk-forward rolling, 24 fold
- **Blocchi**: train 365 / validazione 63 / test 63 giorni, passo 63
- **Offset iniziale**: 59 (= finestra − 1: prima della sessantesima osservazione non esiste alcun grafo)
- **Periodo previsto**: dal 05/05/2022 al 24/06/2026, orizzonte 1 giorno
- **Previsioni per modello**: 22 680 (1 512 giorni × 15 asset)
- **Standardizzazione**: `FoldStandardizer` fittato sul solo train, per asset, e applicato invariato a validazione e test
- **Difesa anti-look-ahead**: quattro test in `tests/evaluation/test_walkforward.py`, ciascuno provato contro una mutazione iniettata

- **Ordine VAR per BIC**: 0 su tutti i 24 fold — la baseline multivariata selezionata non stima alcun coefficiente incrociato e coincide numericamente con la media storica
- **VAR a ordine fissato**: 1 140 coefficienti da 5 475 valori di addestramento, cioè 4,80 osservazioni per parametro
- **Ordine AR medio**: 0,178
- **Quota di AR con ordine 0**: 85,0%

Figura corrispondente: `fig_walkforward_scheme.pdf`.

## 6.5 Confronto predittivo

Accuratezza fuori campione, ordinata per RMSE crescente. La statistica di Diebold–Mariano confronta ciascun modello con la previsione nulla ed è **negativa quando il modello è il più accurato dei due**; il p è corretto secondo Holm sull'insieme dei confronti a coppie.

| Modello | RMSE | MAE | Acc. dir. | Skill | DM vs zero | p (Holm) | Fold con skill > 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero | 0,04145 | 0,02764 | -- | 0,0000 | -- | -- | 0/24 |
| Media storica | 0,04156 | 0,02773 | 50,0% | -0,0052 | 2,78 | 0,077 | 6/24 |
| VAR (BIC) | 0,04156 | 0,02773 | 50,0% | -0,0052 | 2,78 | 0,077 | 6/24 |
| GCN senza grafo | 0,04162 | 0,02776 | 51,2% | -0,0080 | 1,59 | 1,000 | 7/24 |
| AR | 0,04162 | 0,02774 | 50,4% | -0,0082 | 3,36 | 0,012 | 2/24 |
| GCN | 0,04177 | 0,02801 | 50,9% | -0,0156 | 1,82 | 0,683 | 9/24 |
| VAR (p=5) | 0,04978 | 0,03471 | 50,1% | -0,4423 | 11,00 | <0,001 | 0/24 |

- **GCN contro ablazione senza grafo**: DM = 1,338, p = 0,181, p (Holm) = 1,000, differenziale medio di perdita = 0,00001296
- **Lettura**: il segno positivo della statistica dice che la GCN **con** grafo è la meno accurata delle due, e il p dice che la differenza non è distinguibile dal rumore. Il grafo non peggiora in modo dimostrabile, ma non c'è alcuna evidenza che aiuti: è la risposta diretta alla prima questione di ricerca, e va formulata così

Skill score per asset, i due arm della GCN contro la previsione nulla:

| Asset | GCN | GCN senza grafo |
| --- | ---: | ---: |
| ADA | -0,0134 | -0,0057 |
| AVAX | -0,0109 | -0,0072 |
| BCH | -0,0209 | -0,0055 |
| BNB | -0,0347 | -0,0084 |
| BTC | -0,0557 | -0,0329 |
| DOGE | -0,0038 | 0,0053 |
| DOT | -0,0087 | -0,0069 |
| ETC | -0,0138 | -0,0057 |
| ETH | -0,0307 | -0,0177 |
| LINK | -0,0089 | -0,0112 |
| LTC | -0,0202 | -0,0183 |
| SOL | -0,0051 | -0,0070 |
| TRX | -0,0468 | -0,0150 |
| XLM | -0,0099 | -0,0069 |
| XRP | -0,0148 | -0,0043 |

- **Asset con skill positivo**: GCN 0/15, GCN senza grafo 1/15

Strategia sul segno della previsione, equipesata e ribilanciata ogni giorno, su 1 512 giorni. Sharpe annualizzato su 365 giorni, senza tasso privo di rischio. Il buy-and-hold non usa alcuna previsione.

| Modello | Sharpe 0 bps | Sharpe 10 bps | Max DD 0 bps | Max DD 10 bps | Cumulato 0 bps | Cumulato 10 bps | Turnover |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Zero | -- | -- | 0,0% | 0,0% | 0,0% | 0,0% | 0,000 |
| Media storica | -0,665 | -0,671 | -84,4% | -84,5% | -80,5% | -80,7% | 0,007 |
| VAR (BIC) | -0,665 | -0,671 | -84,4% | -84,5% | -80,5% | -80,7% | 0,007 |
| GCN senza grafo | 0,610 | 0,051 | -44,2% | -59,3% | 108,2% | -31,5% | 0,735 |
| AR | -0,596 | -0,718 | -79,1% | -82,9% | -75,6% | -80,3% | 0,141 |
| GCN | 0,453 | 0,014 | -66,0% | -83,8% | 43,1% | -55,1% | 0,766 |
| VAR (p=5) | 0,147 | -0,612 | -60,6% | -87,0% | -16,2% | -81,2% | 0,987 |
| Buy-and-hold | 0,262 | 0,261 | -66,9% | -66,9% | -14,7% | -14,8% | 0,001 |

Associazione tra densità media del grafo nel fold e skill score della GCN nello stesso fold, per le due soglie:

| Misura di densità | ρ di Spearman | p | n | Intervallo | Fold a densità 1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Densità (τ calibrata) | -0,100 | 0,643 | 24 | 0,887 – 1,000 | 8 |
| Densità (τ FWER) | 0,005 | 0,981 | 24 | 0,705 – 0,999 | 0 |

Spearman e non Pearson perché l'ipotesi è di monotonicità, e perché alla soglia calibrata un blocco di fold è saturo a densità esattamente 1: una misura che satura, che i ranghi reggono e un momento prodotto no.

Tabelle corrispondenti: `tab_results_main.tex`, `tab_backtest.tex`. Figure: `fig_results_by_fold.pdf`, `fig_equity_curves.pdf`, `fig_density_vs_error.pdf`.

## 6.6 Struttura topologica e crisi

- **Finestre**: 1 947, dal 02/03/2021 al 30/06/2026
- **Grafo usato**: metriche sul grafo **completo** pesato Mantegna, dove ogni w > 0; la densità è l'unica calcolata sul grafo soglia, perché è l'unica che per definizione dipende da τ

| Metrica | Media | Dev. std | Min | Mediana | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Correlazione media | 0,675 | 0,104 | 0,372 | 0,689 | 0,900 |
| Densità (τ calibrata) | 0,974 | 0,053 | 0,733 | 1,000 | 1,000 |
| Densità (τ FWER) | 0,887 | 0,129 | 0,371 | 0,924 | 1,000 |
| Densità (τ fissa) | 0,950 | 0,079 | 0,562 | 0,990 | 1,000 |
| Connettività algebrica (normalizzata) | 1,024 | 0,022 | 0,923 | 1,028 | 1,058 |
| Connettività algebrica (combinatoria) | 7,046 | 1,252 | 3,750 | 7,060 | 10,443 |
| Lunghezza MST normalizzata | 0,590 | 0,099 | 0,331 | 0,584 | 0,883 |
| Entropia spettrale | 1,242 | 0,282 | 0,518 | 1,221 | 2,003 |
| Quota del modo di mercato | 0,711 | 0,091 | 0,449 | 0,724 | 0,907 |
| Autovalori fuori dal bulk MP | 1,004 | 0,064 | 1,000 | 1,000 | 2,000 |

### Studio degli eventi

Metriche lette a -60, -30, 0, +30, +60 giorni da ciascun evento. La variazione riportata è tra i due offset estremi, cioè tra due finestre che non condividono alcuna osservazione; il percentile colloca il valore nella distribuzione storica completa.

**Stretta cinese** — 19/05/2021

| Metrica | -60g | -30g | 0g | +30g | +60g | Δ -60→+60 | Δ -30→+30 | Perc. a +60g |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Correlazione media | 0,430 | 0,423 | 0,611 | 0,777 | 0,868 | 101,8% | 83,8% | 99,4% |
| Densità (τ calibrata) | 0,848 | 0,790 | 0,971 | 1,000 | 1,000 | 18,0% | 26,5% | 36,0% |
| Densità (τ FWER) | 0,438 | 0,562 | 0,790 | 1,000 | 1,000 | 128,3% | 78,0% | 72,9% |
| Densità (τ fissa) | 0,705 | 0,724 | 0,905 | 1,000 | 1,000 | 41,9% | 38,2% | 52,1% |
| Connettività algebrica (normalizzata) | 0,993 | 0,982 | 1,015 | 1,042 | 1,052 | 5,9% | 6,1% | 97,7% |
| Connettività algebrica (combinatoria) | 5,284 | 4,827 | 6,254 | 8,200 | 9,848 | 86,4% | 69,9% | 99,3% |
| Lunghezza MST normalizzata | 0,826 | 0,758 | 0,654 | 0,489 | 0,386 | -53,3% | -35,5% | 0,5% |
| Entropia spettrale | 1,873 | 1,812 | 1,418 | 0,961 | 0,652 | -65,2% | -47,0% | 0,5% |
| Quota del modo di mercato | 0,499 | 0,500 | 0,658 | 0,798 | 0,878 | 76,0% | 59,7% | 99,4% |
| Autovalori fuori dal bulk MP | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 0,0% | 0,0% | 0,0% |

**Terra/Luna** — 09/05/2022

| Metrica | -60g | -30g | 0g | +30g | +60g | Δ -60→+60 | Δ -30→+30 | Perc. a +60g |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Correlazione media | 0,799 | 0,764 | 0,791 | 0,834 | 0,790 | -1,1% | 9,2% | 86,7% |
| Densità (τ calibrata) | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 0,0% | 0,0% | 36,0% |
| Densità (τ FWER) | 1,000 | 1,000 | 1,000 | 1,000 | 0,981 | -1,9% | 0,0% | 62,5% |
| Densità (τ fissa) | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 0,0% | 0,0% | 52,1% |
| Connettività algebrica (normalizzata) | 1,051 | 1,040 | 1,048 | 1,048 | 1,032 | -1,7% | 0,8% | 63,7% |
| Connettività algebrica (combinatoria) | 9,460 | 8,678 | 8,504 | 8,154 | 7,551 | -20,2% | -6,0% | 64,7% |
| Lunghezza MST normalizzata | 0,496 | 0,511 | 0,471 | 0,429 | 0,487 | -1,8% | -16,0% | 14,6% |
| Entropia spettrale | 0,930 | 1,019 | 0,916 | 0,757 | 0,903 | -2,9% | -25,7% | 12,2% |
| Quota del modo di mercato | 0,814 | 0,784 | 0,811 | 0,851 | 0,811 | -0,3% | 8,5% | 87,1% |
| Autovalori fuori dal bulk MP | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 0,0% | 0,0% | 0,0% |

**FTX** — 08/11/2022

| Metrica | -60g | -30g | 0g | +30g | +60g | Δ -60→+60 | Δ -30→+30 | Perc. a +60g |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Correlazione media | 0,753 | 0,709 | 0,694 | 0,793 | 0,807 | 7,0% | 11,8% | 91,6% |
| Densità (τ calibrata) | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 0,0% | 0,0% | 36,0% |
| Densità (τ FWER) | 1,000 | 1,000 | 0,981 | 1,000 | 1,000 | 0,0% | 0,0% | 72,9% |
| Densità (τ fissa) | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 0,0% | 0,0% | 52,1% |
| Connettività algebrica (normalizzata) | 1,041 | 1,030 | 1,030 | 1,052 | 1,054 | 1,2% | 2,1% | 98,8% |
| Connettività algebrica (combinatoria) | 8,595 | 8,168 | 7,572 | 8,227 | 9,120 | 6,1% | 0,7% | 95,5% |
| Lunghezza MST normalizzata | 0,563 | 0,580 | 0,598 | 0,478 | 0,473 | -15,9% | -17,6% | 13,0% |
| Entropia spettrale | 1,079 | 1,206 | 1,252 | 0,918 | 0,888 | -17,8% | -23,9% | 11,5% |
| Quota del modo di mercato | 0,772 | 0,733 | 0,720 | 0,812 | 0,822 | 6,5% | 10,8% | 90,2% |
| Autovalori fuori dal bulk MP | 1,000 | 1,000 | 1,000 | 1,000 | 1,000 | 0,0% | 0,0% | 0,0% |

Figure corrispondenti: `fig_topology_timeseries.pdf`, `fig_correlation_heatmaps.pdf`, `fig_graph_snapshots.pdf`, `fig_mp_spectrum.pdf`.

## Limiti da dichiarare

Elenco pronto per `sec:limitations`. Ogni voce è verificabile su un artefatto di `results/metrics/`.

1. **Survivorship bias dell'universo** — i 15 asset sono selezionati perché quotati con continuità su Binance dall'inizio del periodo. Gli asset delistati o collassati nel frattempo — Terra/Luna fra tutti — non compaiono, e il campione è per costruzione quello dei sopravvissuti. L'effetto è nella direzione che favorisce i risultati riportati.
2. **Periodo, fonte e valuta unici** — un solo intervallo (2021–2026), un solo exchange (Binance), una sola valuta di quotazione (USDT). Nessuna verifica di robustezza rispetto a un'altra fonte, e le chiusure a 00:00 UTC sono una convenzione dell'exchange, non un fatto del mercato.
3. **Snapshot indipendenti** — il modello vede un grafo per data e nessuna dinamica: nessuna memoria fra finestre consecutive, nessun modulo ricorrente. È il limite già riconosciuto in `sec:dynamic-graphs-temporal-question`, e resta il primo candidato per un lavoro futuro.
4. **Il grafo non è rado** — alla soglia calibrata la densità media è 0,974 (min 0,733, max 1,000): la sogliatura elimina pochi archi e il substrato della GCN è quasi il grafo completo. Le proprietà che la letteratura attribuisce alla sparsità non sono verificate qui, e ogni affermazione sul contributo della *struttura* va letta a questa luce.
5. **La baseline VAR selezionata degenera** — il BIC sceglie ordine 0 su ogni fold, quindi il VAR selezionato non stima alcun coefficiente incrociato e coincide numericamente con la media storica. Il confronto multivariato effettivo è quello con il VAR a ordine fissato, pre-registrato proprio per questa eventualità.
6. **Nessun modello batte la previsione nulla** — lo skill score è positivo per 0 modelli su 7, e per la GCN il test di Diebold–Mariano contro la previsione nulla non è significativo dopo correzione di Holm. Il risultato del capitolo è negativo, ed è riportato come tale: non è un fallimento dell'esperimento, è il suo esito.
7. **Il vantaggio economico svanisce con i costi** — a 10 punti base lo Sharpe della GCN scende a 0,014 e quello dell'ablazione senza grafo a 0,051, contro 0,261 del buy-and-hold. Le strategie sul segno hanno turnover elevato e il costo se lo mangia: un backtest a costo zero non sarebbe stato confrontabile con nulla.
8. **Soglia unica per l'intero periodo** — τ = 0,2145 è calibrata una volta e tenuta fissa, per rendere la densità confrontabile fra periodi. È una scelta dichiarata, non un difetto scoperto, ma implica che la soglia non si adatta ai regimi — e i regimi sono l'oggetto della sez. 6.6.

## Indice degli artefatti

Tutti pubblicati in `../latex-thesis/` da `scripts/08_make_tables.py`.

| Artefatto | Tipo | Sezione |
| --- | --- | ---: |
| `fig_topology_timeseries.pdf` | figura | 6.6 |
| `fig_correlation_heatmaps.pdf` | figura | 6.6 |
| `fig_graph_snapshots.pdf` | figura | 6.6 |
| `fig_mp_spectrum.pdf` | figura | 6.6 |
| `fig_walkforward_scheme.pdf` | figura | 6.4 |
| `fig_results_by_fold.pdf` | figura | 6.5 |
| `fig_equity_curves.pdf` | figura | 6.5 |
| `fig_density_vs_error.pdf` | figura | 6.5 |
| `tab_universe.tex` | tabella | 6.1 |
| `tab_graph_params.tex` | tabella | 6.2 |
| `tab_models.tex` | tabella | 6.3 |
| `tab_results_main.tex` | tabella | 6.5 |
| `tab_backtest.tex` | tabella | 6.5 |

# CLAUDE.md — root

Questo file guida il lavoro sull'intero repository. Le sottocartelle hanno un proprio `CLAUDE.md` che si applica solo al lavoro svolto al loro interno: quello in root si applica sempre, in aggiunta.

## Cos'è questo repository

`crypto-gnn-forecasting` è il progetto per una tesi di laurea triennale su Graph Neural Network applicati alla previsione multi-asset di criptovalute tramite grafi di correlazione dinamici. Contiene due sotto-progetti distinti e indipendenti:

- **`latex-thesis/`** — i sorgenti LaTeX della tesi scritta. Vedi `latex-thesis/CLAUDE.md` per obiettivo, struttura dei capitoli e convenzioni di scrittura.
- **`project-thesis/`** — l'implementazione sperimentale del progetto (dati, grafo dinamico, GCN, baseline VAR/AR/naive, backtest, app Streamlit interattiva), corrispondente a quanto descritto come "Capitolo 6 — studio di caso" in `latex-thesis/chapters/06-case-study.tex`. Pipeline completa e taggata `v1.0-results`; vedi `project-thesis/CLAUDE.md` per struttura e convenzioni. Resta da scrivere il capitolo 6 stesso in `latex-thesis/` (ancora `\input` commentato in `main.tex`).

Le due cartelle sono **indipendenti**: la prima è un documento LaTeX (niente codice da eseguire), la seconda sarà codice (niente prosa da scrivere). Non mescolare le convenzioni delle due — ad esempio, il linguaggio formale/tecnico richiesto per la tesi (vedi `latex-thesis/CLAUDE.md`) non si applica a commenti o messaggi di commit del codice in `project-thesis/`.

## Come vengono letti i CLAUDE.md in questo repo

- Questo file (root) viene caricato sempre, qualunque cartella del repo sia coinvolta nella richiesta.
- `latex-thesis/CLAUDE.md` viene caricato automaticamente quando si legge/modifica un file dentro `latex-thesis/`.
- `project-thesis/CLAUDE.md` viene caricato allo stesso modo per i file dentro `project-thesis/`.
- Se una richiesta tocca file di entrambe le cartelle, entrambi i `CLAUDE.md` di sottocartella possono essere in contesto insieme a questo: in tal caso, applicare le regole di ciascun file solo ai file della propria sottocartella.

## Convenzioni valide per l'intero repository

- Il repository è privato/personale per la stesura della tesi: non pubblicare contenuti (es. tramite Artifact) senza che l'utente lo richieda esplicitamente.
- `.gitignore` in root copre già entrambe le sottocartelle (pattern Python per `project-thesis/`, pattern LaTeX per `latex-thesis/`) più `.claude/` (skill personali, non da pubblicare): quando si aggiungono nuovi strumenti/librerie, aggiornare `.gitignore` in root, non crearne uno per sottocartella.
- Non creare file di pianificazione o riepilogo (es. markdown intermedi) nella root se non esplicitamente richiesti: usare i `CLAUDE.md` di sottocartella per le decisioni stabili, e la memoria di conversazione per il resto.

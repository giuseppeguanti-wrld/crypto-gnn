# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# CLAUDE.md — latex-thesis/

Si applica al lavoro dentro `latex-thesis/`, in aggiunta al `CLAUDE.md` in root.

## Build

MiKTeX (`pdflatex`, `latexmk`, `biber` nel PATH utente). Da `latex-thesis/`:

```
latexmk -pdf -bibtex -shell-escape main.tex
```

- `-shell-escape` è **obbligatorio**: gli snippet Python usano `minted` (serve `pygmentize` nel PATH). Il commento in testa a `main.tex` e il `README.md` riportano ancora il comando senza il flag: sono obsoleti.
- Bibliografia `biblatex` + `biber` (stile `numeric`, `sorting=nyt`); `-bibtex` fa lanciare a latexmk il backend giusto.
- Dopo ogni modifica, ricompilare e controllare `main.log` per `Undefined`, `Overfull`, `Underfull`. Non esiste un test suite: il log pulito è la verifica.
- Un processo avviato prima dell'installazione di MiKTeX non vede il PATH aggiornato finché non viene riavviato del tutto. Il warning `not checked for MiKTeX updates` è innocuo.
- I file `*-SAVE-ERROR` sono residui del salvataggio fallito di un editor, non sorgenti.

## Architettura del documento

`main.tex` orchestra tutto via `\input`; `preamble.tex` contiene pacchetti e ogni personalizzazione (leggerlo prima di aggiungere pacchetti o ambienti: molte scelte hanno un commento che ne spiega il motivo).

- **Numeri di file ≠ numeri di capitolo.** `01-introduction.tex` e `01-project-overview.tex` sono `\chapter*` non numerati (Introduzione, Panoramica del progetto), con `\thesection` ridefinito localmente a `1, 2, …` e ripristinato a fine file. Il primo capitolo numerato è quindi `02-graph-laplacian-theory.tex` = **Cap. 1**, e così via fino a `08-conclusions.tex` = **Cap. 7**. I commenti d'intestazione dei capitoli usano la numerazione del PDF.
- I capitoli non numerati non si possono referenziare con `\cref`: vanno citati in prosa ("L'Introduzione ha mostrato…").
- I capitoli riscritti durante la ristrutturazione (es. `05-`, `07-`, `08-`) si aprono con un blocco di commenti `% ===` che registra le decisioni di stesura di quella sessione (fonti assorbite, prefissi dei label, scelte su tabelle/snippet). Leggerlo prima di modificare il capitolo; alcune note possono riferirsi a file già rimossi (`YYY_literature-review.tex`, `ZZZ_old-case-study.tex`).
- **Snippet di codice**: `\begin{listing}[H]` + `minted{python}` (float stile ruled, numerato per capitolo, `\cref` stampa "Listing"), oppure l'ambiente `longlisting{caption}{label}` definito in `preamble.tex` per blocchi che devono spezzarsi tra pagine. Il codice è verbatim da `../project-thesis/`; le omissioni si segnano con `# ...`.
- Alberi di directory: `fancyvrb` + `pmboxdraw`, non minted (Pygments non gestisce i box-drawing).
- Ambienti `definition`/`proposition`/`theorem`/`example`/`remark` condividono un contatore per capitolo tramite `aliascnt`, così `cleveref` stampa il nome giusto. I nomi italiani per `\cref` di ogni tipo sono in `preamble.tex`: un nuovo tipo di float/ambiente richiede il suo `\crefname`/`\Crefname`.
- `\cleardoublepage` è ridefinito per lasciare vuote le pagine bianche inserite; `\raggedbottom` è voluto (evita lo stiramento degli spazi di `flushbottom` in `twoside`). `openright` è attualmente commentato in `main.tex`.

### Dati e figure dal progetto

Numeri, tabelle e figure provengono da `../project-thesis/`, mai calcolati a mano:

- ogni cifra citata nel testo → `../project-thesis/results/summary.md` (generato da `scripts/08_make_tables.py`);
- figure → prodotte in `../project-thesis/results/figures/` da `scripts/07_make_figures.py` e copiate in `figures/` con lo stesso nome;
- tabelle → generate in `tables/` e incluse con `\input{tables/...}`, oppure ricopiate inline quando il capitolo le estende (vedi commento d'intestazione del capitolo).

## Convenzioni di scrittura

- Italiano formale e tecnico, forma impersonale; nomi dei file in inglese. Terminologia coerente (non alternare "grafo"/"network").
- Notazione fissata nel Cap. 1 (`02-graph-laplacian-theory.tex`) e vincolante ovunque: $G, V, E, W, D, L, U, \Lambda, N$.
- Riferimenti sempre con `\cref`/`\Cref` (`\Cref` a inizio periodo), mai "Capitolo 2" scritto a mano. Citazioni con `\cite`/`\textcite`.
- `\enquote{}` per termini gergali alla prima occorrenza, non per terminologia tecnica stabile.
- Titoli di sezione corti, senza sottotitoli dopo i due punti.
- Nuovo acronimo → `frontmatter/acronyms.tex`, non sciolto in nota. Nuova voce bibliografica → verificare autori/anno/sede con una ricerca, mai a memoria.
- Label di capitolo nella forma `ch:<nome>`; label introdotti durante la ristrutturazione hanno un prefisso per capitolo (es. `topo-`, `bt-`) per evitare collisioni.

## Stato

Tesi in ristrutturazione dal 2026-09-09: da compilativa a tesi che integra teoria e codice. Il vecchio capitolo unico "Caso di studio" e la rassegna della letteratura sono stati smontati e distribuiti nei capitoli tecnici; la struttura attuale è quella di `main.tex`. Il piano di ristrutturazione è tenuto dall'utente fuori dalla repo.

Punto aperto con i relatori: motore bibliografico (`biblatex`/`numeric` attuale vs. BibTeX `plain` del template Uniba, che non supporta `\textcite` né stampa i DOI). Non cambiare motore senza indicazione.

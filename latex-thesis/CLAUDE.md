# CLAUDE.md — latex-thesis/

Guida al lavoro su questa cartella. Consultare prima di modificare struttura, capitoli o bibliografia.

## Obiettivo della tesi

Tesi di laurea **triennale**, compilativa (~60 pagine di corpo), dal titolo *Graph Neural Network per la previsione multi-asset di criptovalute*. Contributo duplice: (1) vantaggio predittivo di una GNN vs. modello univariato; (2) evoluzione della struttura del grafo di correlazione tra regime stabile e crisi.

**Base matematica.** Grafo pesato $G=(V,E,W)$ (nodi = criptovalute, pesi = correlazione di Pearson su finestra mobile). Laplaciano $L=D-W$, proprietà spettrali = "frequenze" del grafo. Architettura centrale: GCN di Kipf & Welling, $H^{(l+1)} = \sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}H^{(l)}W^{(l)})$. Baseline: VAR.

Filo conduttore: *perché serve un grafo → matematica del grafo → si impara sul grafo → cosa succede nei mercati cripto*.

## Struttura e decisioni tecniche

`main.tex` orchestratrice via `\input`; un `.tex` per capitolo in `chapters/` (nomi file inglesi, contenuto italiano); `frontmatter/` (titlepage, dedication, disclaimer, abstract, acronyms, acknowledgements); `bibliography/references.bib`; `figures/`.

- Classe `report`, `twoside,openright,12pt`. `\raggedbottom` impostato in `preamble.tex` (default `flushbottom` di `twoside` stirava gli spazi tra paragrafi/voci di lista fino a badness 10000 su pagine con poco contenuto).
- **Introduzione non numerata**: `\chapter*{Introduzione}` con sezioni interne rinumerate localmente come `1, 2, 3, 4` (non `1.1...`) via `\thesection` sovrascritto e ripristinato a fine capitolo — imita la convenzione del template Uniba. Di conseguenza i capitoli numerati partono da **Fondamenti matematici = Cap. 1** fino a **Conclusioni = Cap. 5** (Cap. 6 studio di caso resta non attivato). Ogni riferimento all'Introduzione altrove nel testo è in prosa ("L'Introduzione ha mostrato...", mai `\cref`, che non funziona su un capitolo non numerato).
- **Bibliografia — punto aperto**: attualmente `biblatex`+`biber`, stile `numeric`. Il template Uniba (confrontato e poi eliminato dalla repo, 2026-08-02) usa BibTeX classico (`plain`), che non supporta `\textcite`/`\Textcite` (15 occorrenze da riscrivere a mano) e non stampa i DOI (44 voci). **In attesa di risposta dei relatori** su quale stile sia vincolante prima di cambiare motore.
- Notazione fissata al Cap. 2, vincolante ovunque: $G, V, E, W, D, L, U, \Lambda, N$.
- Ambienti `definition`/`proposition`/`theorem`/`example`/`remark` in `preamble.tex`, con `aliascnt` per far stampare a `cleveref` il nome corretto invece di "Definition" (contatore condiviso).
- Titlepage compilato: relatori Mazzia/Iavernaro, candidato Guanti, "Tesi di Laurea in Calcolo Numerico", A.A. 2025–2026.

## Ambiente di build

MiKTeX via `winget` (`pdflatex`, `latexmk`, `biber` in PATH utente). Comando di riferimento da `latex-thesis/`:
```
latexmk -pdf -bibtex main.tex
```
Un processo avviato **prima** dell'installazione di MiKTeX non vede il PATH aggiornato finché non viene riavviato per intero (non basta "Reload Window" di VS Code). Il warning `pdflatex: ... not checked for MiKTeX updates` è innocuo. Dopo ogni modifica, ricompilare e controllare l'assenza di `Undefined`/`Underfull`/`Overfull` nel log.

## Stato di avanzamento

Corpo scritto per intero: Introduzione, Cap. 1–4 (Fondamenti matematici, Dalle CNN alle GNN, Modellazione serie finanziarie e grafo, Analisi comparativa), Cap. 5 (Conclusioni). **Cap. 6 (studio di caso) non attivato** — `\input` commentato in `main.tex`; verrà scritto solo dopo il lavoro sperimentale in `../project-thesis/`, con tre opzioni di progetto già valutate in testa al file (A consigliata: analisi topologica descrittiva). Se attivato, la sez. "Domande di ricerca" dell'Introduzione e diversi punti dei Cap. 3–5/Conclusioni andranno rivisti per reintrodurre il contributo empirico (elenco preciso nei commenti guida di `chapters/06-case-study.tex`).

**Frontmatter ancora da scrivere, deliberatamente rimandato**: `abstract.tex`, `dedication.tex`, `acknowledgements.tex` — tutti e tre con TODO espliciti, da completare solo dopo il Cap. 6.

**Revisione complessiva del corpo scritto** (2026-08-02): filo conduttore coerente, dimostrazioni complete (non solo asserite), buona tensione finale tra le due domande di ricerca. Nessun problema di sostanza rilevato oltre ai tre file di frontmatter mancanti. Inviato ai relatori per una prima revisione.

## Linguaggio e convenzioni di scrittura

- Italiano formale e tecnico, terza persona/impersonale, terminologia coerente (mai alternare "grafo"/"network" senza motivo).
- Citazioni sempre `\cite{}`/`\textcite{}`; capitoli/sezioni sempre `\cref{}`/`\Cref{}` (mai "Capitolo 2" scritto a mano) — eccetto l'Introduzione, non numerata (vedi sopra).
- `\Cref{}` a inizio periodo, `\cref{}` a metà periodo. `\enquote{}` per termini gergali alla prima occorrenza, non per terminologia tecnica stabile.
- Titoli di sezione: corti, senza sottotitoli descrittivi con i due punti (convenzione applicata retroattivamente a gran parte dei Cap. 3–5, 7 il 2026-08-02 — vedi git log per l'elenco puntuale delle riformulazioni).
- Nuovo acronimo mai comparso come sigla → aggiungerlo a `frontmatter/acronyms.tex`, non scioglierlo in nota.
- Nuova citazione non in `references.bib` → verificare autori/anno/rivista via ricerca web prima di aggiungerla, mai a memoria.

## Prossimi passi

Attendere risposta dei relatori su: (1) motore bibliografico (biblatex/numeric vs. BibTeX/plain), (2) pagine bianche fronte-retro prima di alcuni capitoli (convenzione `openright`, eliminabile se non richiesta). Poi: lavoro sperimentale in `project-thesis/`, quindi Cap. 6 e i tre file di frontmatter mancanti.

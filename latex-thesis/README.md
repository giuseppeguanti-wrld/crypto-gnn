# Tesi — sorgenti LaTeX

Struttura del progetto:

```
latex-thesis/
├── main.tex                 # documento principale, \input di tutte le parti
├── preamble.tex              # pacchetti e configurazione
├── frontmatter/
│   ├── titlepage.tex
│   └── abstract.tex
├── chapters/
│   ├── 01-introduction.tex
│   ├── 02-mathematical-foundations.tex
│   ├── 03-from-cnn-to-gnn.tex
│   ├── 04-financial-series-and-graph-construction.tex
│   ├── 05-literature-review.tex
│   ├── 06-case-study.tex     # modulare/opzionale, non incluso di default
│   └── 07-conclusions.tex
├── bibliography/
│   └── references.bib
└── figures/
```

## Compilazione

Con `latexmk` (richiede una distribuzione TeX con `biber`, es. TeX Live o MiKTeX):

```bash
cd latex-thesis
latexmk -pdf -bibtex main.tex
```

In alternativa, manualmente:

```bash
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex
```

Il Capitolo 6 (studio di caso) è opzionale: per includerlo nel PDF, decommentare la riga `\input{chapters/06-case-study}` in `main.tex` (vedi il commento in testa a quel file per le alternative di progetto disponibili).

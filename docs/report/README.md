# Report source

LaTeX source of the project report, kept under version control so that the
document and the code that produced its figures stay in sync. It is the same
project that was written on Overleaf, with the changes described in the
"what changed" section of the top-level README.

Build with any pdfTeX toolchain:

```bash
cd docs/report
latexmk -pdf main.tex
```

`main.tex` pulls in one file per section (`summary`, `introduction`, `models`
which in turn includes `dcgan` and `vae`, `algorithm`, `mnist`, `results`,
`conclusions`) and closes with the bibliography in `references.bib`. Figures
live in `assets/`.

# Paper sources

LaTeX source and figures of the paper. The scripts and data behind every figure and table are listed in the
figure and table index of the [repository README](../README.md#figure-and-table-index).

## Files

- `main.tex`: the paper, with Appendix A (pseudocode and parameters of VQPM).
- `appendix_theory.tex`: Appendix B, theoretical analysis (coherent power iteration, the VQPM update, locking, the
  Hoeffding threshold, a moment-matrix view).
- `appendix_classical.tex`: Appendix C, classical sampling of VQPM (shot-by-shot simulation, shots on a device
  against classical draws, runs at n = 20 to 100, the product-state map without and with locking).
- `main.bib`: references.
- `graphical.tex`: the overview figure after the abstract (included in `main.tex`).
- `frontmatter.tex`: highlights (not included in `main.tex`).
- `figures/`: all figures. The run figures of Sec. 2 and the cosine figure are made by the scripts in the
  repository root (`plot_qubit_vs_metrics.py`, `misc_plots.py`, `vqpm.py`); the others by the scripts in
  `theory_checks/`. The circuit of Sec. 2.1 is `figures/vqpm_circuit.tex` (qcircuit), compiled on its own to
  `figures/vqpm_circuit.pdf` (`cd figures && latexmk -pdf vqpm_circuit.tex`) and included as an image, so that the
  HTML version of the paper on arXiv shows it as well.
- `main.pdf`: the compiled paper.
- `update_figure_index.py`: writes the figure and table index into the repository README, with the figure, table
  and section numbers of the compiled paper.

## Build

```bash
cd paper
latexmk -pdf main.tex
cd ..
python3 paper/update_figure_index.py      # refresh the numbers in the README index
```

On Debian or Ubuntu this needs `texlive-latex-extra texlive-science texlive-pictures lmodern latexmk`. To check the
HTML version as arXiv makes it, run `latexmlc --format=html5 --dest=html/index.html main.tex` (package `latexml`).

# Paper

Self-contained LaTeX source for an arXiv-style write-up of the
CogARC human-analysis pipeline. Compiles with a vanilla TeX Live
installation; no custom class or bib manager required (the
bibliography is inline).

## Build

```
pdflatex paper.tex
pdflatex paper.tex   # run twice for references
```

or

```
latexmk -pdf paper.tex
```

## Contents

- `paper.tex` — the manuscript.
- `figures/` — seven PNGs, all referenced by the .tex:
  - `motor_vs_cognitive_figure.png`     (§3, Fig. 1)
  - `motor_cognitive_case_studies.png`  (§3, Fig. 2)
  - `drawing_strategy_figure.png`       (§4, Fig. 3)
  - `color_vs_cc_example.png`           (§4, Fig. 4)
  - `strategies_vs_human_figure.png`    (§5, Fig. 5)
  - `task_vs_strategy_figure.png`       (§6, Fig. 6)
  - `chunks_vs_errors_figure.png`       (§7, Fig. 7)

Each figure is produced by the corresponding pipeline script in the
repo root (`motor_vs_cognitive.py`, `plot_drawing_strategy.py`,
`plot_color_vs_cc_example.py`, `plot_strategies.py`,
`plot_task_vs_strategy.py`, `plot_chunks_vs_errors.py`). Re-running
those scripts regenerates the PNGs in `prior_analysis/`; copy them
here to refresh the paper figures.

## Placeholders

The title page uses `Author Name` / `Affiliation` placeholders. The
bibliography cites CogARC and ARC-GNN work with minimal details;
replace with real citations before submission.

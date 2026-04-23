# cogarc-human-analysis

Behavioral-data analysis of human solvers on the Cognitive Abstraction and
Reasoning Corpus (CogARC). Uses the [Boston University CogARC Data
Repository](https://zenodo.org/records/18177487) (Experiments 1 and 2, 233
participants × 75 tasks) to characterize *how* humans draw — per-task
drawing strategies, motor vs cognitive error composition, pause-segmented
cognitive chunks, attention transition graphs — and to export priors that
can be fed to the ARC-GNN solver.

## Contents at a glance

1. [Install](#install) & [Data](#data)
2. [Key findings](#key-findings) — headline results, each with its figure
3. [Module map](#module-map) — every script and what it does
4. [Running](#running) — invocation, reproduction order
5. [Papers](#papers) — LaTeX write-ups in `paper/`

## Install

```bash
pip install numpy pandas scipy matplotlib scikit-learn python-docx pyarrow
# optional: torch + networkx if you want to run human_solver.py
```

## Data

Unzip the CogARC release to `data/CogARC-dataRepository/` (gitignored)
at the repo root:

```
data/CogARC-dataRepository/
├── Behavioral data/
│   ├── Experiment 1/{Edit sequences, Submissions}/
│   └── Experiment 2/{Edit sequences, Submissions}/
├── Common solutions/
├── Task example PNGs/
├── Task JSONs/
├── Task tags.csv
└── README.rtf
```

Every module resolves this location automatically via `_paths.DATA_ROOT`
regardless of where it is invoked from. Outputs land in
`prior_analysis/` at the repo root.

## Key findings

### 1 · Humans make two qualitatively different kinds of errors

A per-edit classifier separates **motor** (pen slipped, quickly
corrected, random cell) from **cognitive** (systematic, aligns with a
canonical Top-Error, takes longer, no quick correction) errors using
five signals and bootstrap CIs. Task composition is reliable
(split-half ρ ≈ 0.77) and carries into downstream predictive analyses.

- Code: `human_error_type.py`, `motor_vs_cognitive.py`
- Figures: `prior_analysis/motor_vs_cognitive_figure.png`,
  `prior_analysis/motor_cognitive_case_studies.png`
- Methods doc: `docs/motor_vs_cognitive_methods.{txt,docx}`

### 2 · Drawing strategies are task-driven, not subject-driven

Six canonical strategies (`object_first`, `color_first`, `nn_color_first`,
`row_first`, `col_first`, `random_k3`) are scored against every
trajectory; 67/75 tasks afford ≥ 5 of them, so affordance does *not*
explain which strategy humans pick. Task-intrinsic features do:
task identity explains 4–5× more variance in strategy choice than
subject identity.

- Code: `human_vs_strategies.py`, `human_task_vs_strategy.py`
- Figures: `prior_analysis/strategies_vs_human_figure.png`,
  `task_vs_strategy_figure.png`, `drawing_strategy_figure.png`

### 3 · Chunks align with Top Errors and predict outcomes early

Edit sequences are segmented at pauses longer than
`max(2 × subject-median RT, 500 ms)`. On balanced subsets, the *first
quarter* of a trajectory's chunks predicts which Top Error a subject
will commit at 74% accuracy vs a 56% baseline. Error-proneness is a
stable subject-level trait (split-half ρ ≈ 0.35), but the particular
Top Error chosen is not (ρ ≤ 0.16) — task-level bias dominates.

- Code: `human_chunking.py`, `human_chunks_vs_errors.py`
- Figure: `prior_analysis/chunks_vs_errors_figure.png`

### 4 · Chunks really are cognitive units (not analytical artifacts)

Three independent tests on 399 k chunks × 75 tasks × 233 subjects
confirm that pause-segmented chunks reflect participants' own
segmentation, not a slicing we impose:

| test | real | null-Cut (random cuts, same count) | Δ |
|---|---|---|---|
| inter-edit RT median | **intra 217 ms / inter 1 703 ms** (7.8× gap) | single-mode long-tail | bimodal vs unimodal |
| color homogeneity | 0.989 | 0.935 | +0.054 |
| frac 4-connected | **0.774** | 0.596 | **+0.178** |
| best IoU with Success component | 0.149 | 0.130 | +0.019 |
| cross-subject ARI (3 750 pairs) | **0.524** | 0.234 | **+0.290** |

Real pause-segmented chunks have higher colour homogeneity, are more
often spatially connected, align better with object boundaries, and
crucially, **different subjects agree on how to chunk each task more
than twice as often as chance** (ARI 0.52 vs 0.23).

- Code: `chunks_are_cognitive_units.py`, `plot_chunks_are_cognitive_units.py`
- Figure: `prior_analysis/chunks_cognitive_units_figure.png`
- Stand-alone write-up: `paper/chunk_analysis.{tex,pdf}`

## Module map

### Core data + features

| File | What it does |
|---|---|
| `human_targets.py` | Loads the Success + Top Error 1..K grids per task and empirical response frequencies from Experiment 2 `_submission1.json` files. |
| `human_style_features.py` | Per-task aggregate behavioral features from the Experiment 2 edit-sequence CSVs (color-run length, adjacency rate, scan direction, extension/seed/stamp-burst rates, component stats) and a z-score-based prior recommender. |
| `human_variance.py` | Decomposes the variance of each feature into SUBJECT vs TASK components over 14 k trajectories × 233 subjects × 75 tasks. |
| `human_sequence.py` | First-edit saliency map, cell-order consistency (Spearman across subjects), color-priority sequence, and the attention transition graph per task. |
| `human_component_priority.py` | Per-task component-ordering analysis: which Success component subjects paint first, correlated against component features. |

### Errors (motor vs cognitive)

| File | What it does |
|---|---|
| `human_error_type.py` | Per-edit motor vs cognitive classifier: spatial distance to correct, in-wrong-run, matches-top-error, RT z-score, quick-correction window. Writes `prior_analysis/error_types{,_per_task}.csv`. |
| `motor_vs_cognitive.py` | Aggregates the classifier into a sorted 75-task table with bootstrap CIs + split-half reliability and builds the paper-quality overview + case-study figures. |
| `exp1_time_trend.py` | Within-subject progression of motor vs cognitive rates across the Experiment 1 fixed task order (49 subjects, up to 75 trials each). |

### Drawing strategies

| File | What it does |
|---|---|
| `human_vs_strategies.py` | Compares each human trajectory to the six canonical synthetic strategies on chunk-level features; writes `subject_best_strategy.csv`. |
| `human_task_vs_strategy.py` | Tests whether task-intrinsic features predict which strategy the population uses; writes per-task correlations + dominant-strategy summaries. |

### Chunks (pause-segmented cognitive units)

| File | What it does |
|---|---|
| `human_chunking.py` | Pause-based segmentation + per-chunk features (size, color homogeneity, connectedness, bbox, Success-component IoU, color-class IoU). |
| `human_chunks_vs_errors.py` | Asks whether chunks diagnose which Top Error a subject will commit. Early-chunk prediction, per-subject reliability, task-level η² decomposition, diagnostic-cell voting schemes. |
| `chunks_are_cognitive_units.py` | **Methodological validation.** Three tests that pause-segmented chunks are the participant's own cognitive unit: bimodal inter-edit RT, real chunks beat two null re-segmentations on structural coherence, cross-subject ARI far exceeds chance. See Key Finding #4 above. |

### Plot scripts

| File | What it does |
|---|---|
| `plot_sequence.py` | Four-task figure: Success grid + first-edit heatmap + attention transition arrows + color-priority bars. |
| `plot_propagation.py` | Histograms of the edit-propagation features + example trajectories for the three majority-error tasks. |
| `plot_error_types.py` | Stacked per-task motor/cognitive bars + pooled RT histogram (superseded by the figures in `motor_vs_cognitive.py`). |
| `plot_drawing_strategy.py` | Strategy × chunk figure with the six canonical strategies. |
| `plot_strategies.py` | Strategies vs humans distance, radar chart + case studies. |
| `plot_color_vs_cc_example.py` | Illustrates the difference between color-class IoU and connected-component IoU on one example. |
| `plot_task_vs_strategy.py` | Five-panel: affordance histogram, dominance distribution, feature × strategy-usage heatmap, three scatter plots, two concrete examples. |
| `plot_chunks_vs_errors.py` | Five-panel: early-vs-late chunk prediction, subject split-half ρ, variance decomposition, TE pairwise accuracy, accuracy by task diagnosability. |
| `plot_chunks_are_cognitive_units.py` | Five-panel: intra/inter-chunk RT distributions, real vs two nulls on three coherence metrics, cross-subject ARI. |

### GNN integration

| File | What it does |
|---|---|
| `human_solver.py` | ARCSolverModel trainer that scores predictions against the distribution of human responses, returning per-label winners (Success + each Top Error). **Requires an ARC-GNN checkout** — set `ARC_GNN_ROOT=/path/to/ARC-GNN`. |
| `run_human_analysis.py` | CLI sweeping (prior × edge rule × architecture) per task, gated by `--use_recommender` so priors come from the `human_style_features.recommend_priors` heuristic. Same `ARC_GNN_ROOT` requirement. |

### Utilities

| File | What it does |
|---|---|
| `make_methods_docx.py` | Converts any of the `docs/*_methods.txt` files to a `.docx` for pasting into a methods section. |
| `_paths.py` | Single source of truth for `DATA_ROOT` so every script works from any cwd. |

## Running

Both invocation styles work and resolve paths the same way:

```bash
# As plain scripts
python human_targets.py --task_id 00d62c1b
python motor_vs_cognitive.py
python chunks_are_cognitive_units.py
python plot_chunks_are_cognitive_units.py

# As modules
python -m human_targets --task_id 00d62c1b
python -m motor_vs_cognitive

# GNN integration (requires an ARC-GNN checkout)
export ARC_GNN_ROOT=/path/to/ARC-GNN
python run_human_analysis.py --all --use_recommender --rec_top_k 4 \
    --archs small --edge_rules none --epochs 300
```

Outputs land in `prior_analysis/` at the repo root.

## Typical reproduction path

1. `python human_style_features.py` — parses edit sequences, writes per-task feature table.
2. `python human_variance.py` — variance decomposition (subject vs task).
3. `python human_error_type.py` — classifies every wrong edit motor / cognitive / ambiguous.
4. `python motor_vs_cognitive.py` — sorted 75-task table + paper figures.
5. `python human_chunking.py` — pause-segmented chunks + per-chunk features.
6. `python human_chunks_vs_errors.py` — early-chunk Top-Error prediction + reliability analyses.
7. `python chunks_are_cognitive_units.py && python plot_chunks_are_cognitive_units.py` — methodological validation of chunks as cognitive units.
8. `python human_vs_strategies.py && python human_task_vs_strategy.py` — strategy comparison + task-feature regression.
9. `python human_sequence.py && python plot_sequence.py` — saliency + attention transitions.
10. `python exp1_time_trend.py` — within-subject time trend (Experiment 1 only, since Experiment 2 trial order is randomized and not exposed in the released data).
11. `python run_human_analysis.py --all --use_recommender` *(optional, needs `ARC_GNN_ROOT`)*.

## Papers

Self-contained LaTeX write-ups in `paper/` (plus compiled PDFs):

- `paper/paper.tex` — full arXiv-style paper bundling motor vs cognitive,
  chunks vs errors, drawing strategies vs humans, task-feature
  predictors, and common-error diagnosis findings, with GNN design
  implications.
- `paper/chunk_analysis.tex` — focused results-section proving chunks
  are participants' own cognitive units (the five-panel figure in Key
  Finding #4).

Build with `pdflatex` (run twice for references) or `latexmk -pdf`.

## Dataset citation

If you use any of these analyses please cite the CogARC dataset:

> Ahn, Caroline. *Boston University CogARC Data Repository*, Zenodo
> (2026). doi:10.5281/zenodo.18177487

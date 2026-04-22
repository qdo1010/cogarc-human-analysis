# cogarc-human-analysis

Behavioral-data analysis of human solvers on the Cognitive Abstraction and
Reasoning Corpus (CogARC). Uses the [Boston University CogARC Data
Repository](https://zenodo.org/records/18177487) (Experiments 1 and 2)
to characterize *how* humans draw — per-task drawing strategies, motor
vs cognitive error composition, saliency / attention-transition graphs —
and to export priors that can be fed to the ARC-GNN solver.

## Install

```bash
pip install numpy pandas scipy matplotlib python-docx pyarrow
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
regardless of where it is invoked from.

## Module map

| File | What it does |
|---|---|
| `human_targets.py` | Loads the Success + Top Error 1..K grids per task and empirical response frequencies from Experiment 2 `_submission1.json` files. |
| `human_style_features.py` | Per-task aggregate behavioral features from the Experiment 2 edit-sequence CSVs (color-run length, adjacency rate, scan direction, extension/seed/stamp-burst rates, component stats) and a z-score-based prior recommender. |
| `human_variance.py` | Decomposes the variance of each feature into SUBJECT vs TASK components over 14 k trajectories × 233 subjects × 75 tasks. |
| `human_sequence.py` | First-edit saliency map, cell-order consistency (Spearman across subjects), color-priority sequence, and the attention transition graph per task. |
| `human_error_type.py` | Per-edit **motor vs cognitive** classifier: spatial distance to correct, in-wrong-run, matches-top-error, RT z-score, quick-correction window. Writes `prior_analysis/error_types{,_per_task}.csv`. |
| `motor_vs_cognitive.py` | Aggregates the classifier into a sorted 75-task table with bootstrap CIs + split-half reliability and builds the paper-quality overview + case-study figures. |
| `exp1_time_trend.py` | Within-subject progression of motor vs cognitive rates across the Experiment 1 fixed task order (49 subjects, up to 75 trials each). |
| `plot_propagation.py` | Histograms of the edit-propagation features + example trajectories for the three majority-error tasks. |
| `plot_sequence.py` | Four-task figure: Success grid + first-edit heatmap + attention transition arrows + color-priority bars. |
| `plot_error_types.py` | Stacked per-task motor/cognitive bars + pooled RT histogram (superseded by the figures in `motor_vs_cognitive.py`). |
| `human_solver.py` | ARCSolverModel trainer that scores predictions against the distribution of human responses, returning per-label winners (Success + each Top Error). **Requires an ARC-GNN checkout** — set the environment variable `ARC_GNN_ROOT=/path/to/ARC-GNN`. |
| `run_human_analysis.py` | CLI that sweeps (prior × edge rule × architecture) per task, optionally gated by `--use_recommender` so priors come from the behavioral heuristic in `human_style_features.recommend_priors`. Same `ARC_GNN_ROOT` requirement as above. |
| `make_methods_docx.py` | Converts `docs/motor_vs_cognitive_methods.txt` to a `.docx` for pasting into a methods section. |
| `docs/motor_vs_cognitive_methods.{txt,docx}` | 8-section methods writeup of the motor-vs-cognitive classifier (data, signals, rule, aggregation, validation, outputs, caveats, reproduction). |

## Running

Both invocation styles work and resolve paths the same way:

```bash
# As plain scripts
python human_targets.py --task_id 00d62c1b
python motor_vs_cognitive.py
python plot_sequence.py

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
5. `python human_sequence.py` and `python plot_sequence.py` — saliency + attention transitions.
6. `python exp1_time_trend.py` — within-subject time trend (Experiment 1 only, since Experiment 2 trial order is randomized and not exposed in the released data).
7. `python run_human_analysis.py --all --use_recommender` (optional, needs `ARC_GNN_ROOT`).

## Dataset citation

If you use any of these analyses please cite the CogARC dataset:

> Ahn, Caroline. *Boston University CogARC Data Repository*, Zenodo
> (2026). doi:10.5281/zenodo.18177487

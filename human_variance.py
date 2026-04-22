"""
Decompose variance of edit-style features into SUBJECT vs TASK effects.

Builds a (subject, task) -> features table by parsing every Experiment 2
edit-sequence CSV (~15k trajectories), caches it to parquet/csv, and
reports for each behavioral feature:

    var_task   = var(E[feature | task])     # between-task
    var_subj   = var(E[feature | subject])  # between-subject
    var_total  = var(feature)                # raw across all (subj, task)
    frac_task  = var_task / var_total
    frac_subj  = var_subj / var_total

If frac_task >> frac_subj, the feature is TASK-driven (the task imposes the
drawing strategy). If frac_subj >> frac_task, the feature is a SUBJECT trait
that persists across tasks. Residual variance = everything not explained by
either main effect (an interaction + idiosyncratic trial-to-trial noise).

Also:
    - per-subject consistency: for a fixed subject, how much do their
      feature values spread across tasks (coefficient of variation)?
    - rank-stability: for each feature, do subjects who are high on task A
      tend to be high on task B too (Spearman across tasks, averaged)?
"""

from __future__ import annotations

import _paths  # noqa: F401  (sys.path bootstrap, safe under both invocations)

import os
import pickle
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from human_style_features import (
    DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR,
    _parse_trajectory, _trajectory_features,
)


CACHE_PATH = "prior_analysis/subj_task_features.parquet"
CSV_PATH   = "prior_analysis/subj_task_features.csv"

_FNAME_RE = re.compile(r"^subj_([^_]+)_trial_([0-9a-f]+)\.json\.csv$")

# Focus on the behavioral features that survived the discriminability check
FEATURES_OF_INTEREST = [
    "n_edits", "n_reset", "n_showex", "used_copy", "mean_rt_ms",
    "color_run_mean", "within_obj_rate", "adjacency_rate",
    "best_scan_corr", "unique_colors",
    "extension_rate", "new_seed_rate", "stamp_burst_rate",
    "same_color_component_count", "same_color_component_max",
]


# ---------------------------------------------------------------------------
# Build the (subject, task) feature table
# ---------------------------------------------------------------------------

def build_subj_task_table(data_root: str = DEFAULT_DATA_ROOT,
                          force: bool = False) -> pd.DataFrame:
    if not force and os.path.exists(CACHE_PATH):
        return pd.read_parquet(CACHE_PATH)

    root = os.path.join(data_root, _EXP2_EDIT_DIR)
    rows: List[Dict] = []
    task_dirs = sorted(d for d in os.listdir(root) if d.endswith(".json"))
    for i, td in enumerate(task_dirs, 1):
        task_id = td[:-5]  # strip ".json"
        task_dir = os.path.join(root, td)
        if not os.path.isdir(task_dir):
            continue
        for fname in os.listdir(task_dir):
            m = _FNAME_RE.match(fname)
            if not m:
                continue
            subj_id = m.group(1)
            tr = _parse_trajectory(os.path.join(task_dir, fname))
            if not tr["edits"]:
                continue
            f = _trajectory_features(tr)
            rec = {"task_id": task_id, "subject_id": subj_id}
            rec.update({k: f[k] for k in FEATURES_OF_INTEREST if k in f})
            rows.append(rec)
        if i % 10 == 0:
            print(f"  parsed {i}/{len(task_dirs)} tasks ({len(rows)} trajectories)")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)
    df.to_csv(CSV_PATH, index=False)
    print(f"wrote {CACHE_PATH}  ({len(df)} rows, {df['subject_id'].nunique()} subjects, "
          f"{df['task_id'].nunique()} tasks)")
    return df


# ---------------------------------------------------------------------------
# Variance decomposition
# ---------------------------------------------------------------------------

def variance_decomposition(df: pd.DataFrame) -> pd.DataFrame:
    """For each feature, report:
          var_total, var_task, var_subj, frac_task, frac_subj,
          n_subjects, n_tasks, n_obs."""
    out_rows = []
    for feat in FEATURES_OF_INTEREST:
        s = df[feat].dropna()
        if len(s) == 0 or s.std(ddof=0) == 0:
            continue
        sub = df[["task_id", "subject_id", feat]].dropna()
        var_total = sub[feat].var(ddof=0)
        task_means = sub.groupby("task_id")[feat].mean()
        subj_means = sub.groupby("subject_id")[feat].mean()
        var_task = task_means.var(ddof=0)
        var_subj = subj_means.var(ddof=0)
        out_rows.append({
            "feature": feat,
            "var_total": var_total,
            "var_task": var_task,
            "var_subj": var_subj,
            "frac_task": var_task / var_total if var_total else 0.0,
            "frac_subj": var_subj / var_total if var_total else 0.0,
            "n_obs": len(sub),
            "n_tasks": sub["task_id"].nunique(),
            "n_subjects": sub["subject_id"].nunique(),
        })
    return pd.DataFrame(out_rows).sort_values("frac_task", ascending=False)


def subject_rank_stability(df: pd.DataFrame,
                           feature: str,
                           min_tasks_per_subj: int = 15) -> Dict:
    """Average pairwise Spearman across tasks: for each pair (A, B), take
    the subjects present on both, rank them by `feature`, correlate. Then
    average over all task pairs.

    A positive correlation means: subjects who are high on task A also
    tend to be high on task B — i.e. the feature is a person trait that
    persists across tasks.
    """
    tasks = df["task_id"].unique()
    # subset to subjects that did at least min_tasks_per_subj tasks
    subj_counts = df.groupby("subject_id").size()
    kept = subj_counts[subj_counts >= min_tasks_per_subj].index
    sub = df[df["subject_id"].isin(kept)]
    tasks = sub["task_id"].unique()

    # Build subject x task matrix of this feature
    mat = sub.pivot_table(index="subject_id", columns="task_id",
                          values=feature, aggfunc="mean")
    # Pairwise Spearman across task columns
    from scipy.stats import spearmanr
    corrs = []
    cols = list(mat.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a = mat[cols[i]]
            b = mat[cols[j]]
            mask = a.notna() & b.notna()
            if mask.sum() < 10:
                continue
            rho, _ = spearmanr(a[mask], b[mask])
            if not np.isnan(rho):
                corrs.append(rho)
    return {
        "feature": feature,
        "n_task_pairs": len(corrs),
        "mean_spearman": float(np.mean(corrs)) if corrs else float("nan"),
        "median_spearman": float(np.median(corrs)) if corrs else float("nan"),
        "frac_positive": float(np.mean([c > 0 for c in corrs])) if corrs else float("nan"),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_variance_bars(decomp: pd.DataFrame, out_path: str) -> None:
    import matplotlib.pyplot as plt
    d = decomp.sort_values("frac_task").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(d))
    ax.barh(y - 0.2, d["frac_task"], 0.38, color="#1f77b4",
            label="between-task (task effect)")
    ax.barh(y + 0.2, d["frac_subj"], 0.38, color="#ff7f0e",
            label="between-subject (person trait)")
    ax.set_yticks(y)
    ax.set_yticklabels(d["feature"])
    ax.set_xlabel("fraction of total variance")
    ax.axvline(0.5, color="gray", lw=0.5, linestyle="--")
    ax.set_title("Variance decomposition per feature\n"
                 "Left = subject-driven (person trait), Right = task-driven",
                 fontsize=11)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


def plot_subject_task_heatmap(df: pd.DataFrame, feature: str,
                              out_path: str,
                              min_tasks_per_subj: int = 15) -> None:
    import matplotlib.pyplot as plt
    subj_counts = df.groupby("subject_id").size()
    kept = subj_counts[subj_counts >= min_tasks_per_subj].index
    sub = df[df["subject_id"].isin(kept)]
    mat = sub.pivot_table(index="subject_id", columns="task_id",
                          values=feature, aggfunc="mean")

    # Order subjects by their mean on the feature; tasks by their mean.
    mat = mat.loc[mat.mean(axis=1).sort_values().index]
    mat = mat.reindex(columns=mat.mean().sort_values().index)

    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(mat.values, aspect="auto", cmap="viridis",
                   interpolation="nearest")
    ax.set_xlabel(f"tasks sorted by mean {feature}")
    ax.set_ylabel(f"subjects sorted by mean {feature}  "
                  f"(n={mat.shape[0]} with >={min_tasks_per_subj} tasks)")
    ax.set_title(
        f"{feature}: per-(subject, task) matrix.\n"
        "Pure column-stripes = task effect dominates. "
        "Pure row-stripes = subject effect dominates.",
        fontsize=10,
    )
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, label=feature)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="Re-parse all CSVs even if cache exists.")
    ap.add_argument("--min_tasks_per_subj", type=int, default=15)
    args = ap.parse_args()

    df = build_subj_task_table(force=args.rebuild)
    print(f"\nTable: {len(df)} rows, {df['subject_id'].nunique()} subjects, "
          f"{df['task_id'].nunique()} tasks")

    decomp = variance_decomposition(df)
    os.makedirs("prior_analysis", exist_ok=True)
    decomp.to_csv("prior_analysis/variance_decomposition.csv", index=False)
    print("\n--- Variance decomposition (sorted by task fraction) ---")
    print(decomp[["feature", "frac_task", "frac_subj", "n_obs",
                  "n_tasks", "n_subjects"]].round(3).to_string(index=False))

    print("\n--- Subject rank-stability across tasks ---")
    stab_rows = []
    for feat in FEATURES_OF_INTEREST:
        stab_rows.append(subject_rank_stability(
            df, feat, min_tasks_per_subj=args.min_tasks_per_subj))
    stab = pd.DataFrame(stab_rows).sort_values("mean_spearman",
                                               ascending=False)
    stab.to_csv("prior_analysis/subject_rank_stability.csv", index=False)
    print(stab.round(3).to_string(index=False))

    plot_variance_bars(decomp, "prior_analysis/variance_decomposition.png")
    for feat in ["extension_rate", "n_edits", "stamp_burst_rate"]:
        plot_subject_task_heatmap(
            df, feat,
            f"prior_analysis/heatmap_{feat}.png",
            min_tasks_per_subj=args.min_tasks_per_subj,
        )


if __name__ == "__main__":
    main()

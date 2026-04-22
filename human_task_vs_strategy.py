"""
What about a TASK forces humans into a strategy?

Two hypotheses to separate:
  (H1) The task affords only one plausible strategy, so every human
       picks it because there's nothing else to pick.
  (H2) The task affords several strategies and humans genuinely
       coordinate on one.

Operationalisation:

  * affordance(task, strategy) = True iff the synthetic strategy
    produces >= 2 chunks on the task's Success grid. With 1 chunk
    the strategy is degenerate (the whole answer is one unit).

  * n_afforded(task) = number of afforded strategies (0..6).

  * human_entropy(task) = Shannon entropy (base 2) over the
    distribution of subjects' best-matching strategies on that
    task. 0 = perfect agreement, log2(n_afforded) = maximum spread.

  * dominant_frac(task) = fraction of subjects choosing the modal
    strategy. 1 = everyone agrees, 1/n_afforded = chance.

Then we relate these to task-intrinsic features computed straight
from the Success grid + test input and to Experiment-2 interface
usage (copy / showex / reset rates), WITHOUT any reliance on the
Obj/Geo/Num/GoD tags.

Task features:
    grid_h, grid_w, n_cells, frac_nonbg
    n_colors_output, n_colors_input, n_new_colors (output - input)
    n_success_cc, mean_cc_area, max_cc_area
    input_to_output_size_ratio
    symmetry_h / symmetry_v / symmetry_diag (pixel-match under flip)
    used_copy_mean, n_showex_mean, n_reset_mean, n_edits_mean
      (from prior_analysis/style_features_all75.csv)

Outputs:
    prior_analysis/task_features_for_strategy.csv
    prior_analysis/task_vs_strategy_correlations.csv
"""

from __future__ import annotations


import _paths  # noqa: F401

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.ndimage import label as cc_label
from scipy.stats import entropy, spearmanr

from human_chunking import (
    best_iou_with_success, chunk_features, color_class_iou,
    n_success_cc_spanned, _success_components,
)
from human_vs_strategies import (
    FEATURES_OF_INTEREST, STRATEGIES, _success_grid,
)
from human_targets import DEFAULT_DATA_ROOT, available_task_ids, human_targets


# ---------------------------------------------------------------------------
# Task-intrinsic features
# ---------------------------------------------------------------------------

def _background(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _symmetry_score(grid: np.ndarray, axis: str) -> float:
    """Fraction of non-background cells whose reflection is the same color."""
    if grid.size == 0:
        return float("nan")
    if axis == "h":
        flipped = np.fliplr(grid)
    elif axis == "v":
        flipped = np.flipud(grid)
    elif axis == "d":
        flipped = grid.T if grid.shape[0] == grid.shape[1] else grid
    else:
        raise ValueError(axis)
    mask = grid != _background(grid)
    if not mask.any():
        return 0.0
    return float((grid[mask] == flipped[mask]).mean())


def task_features(task_id: str) -> Dict:
    task = json.load(open(
        os.path.join(DEFAULT_DATA_ROOT, "Task JSONs", f"{task_id}.json")))
    out = np.array(task["test"][0]["output"])
    inp = np.array(task["test"][0]["input"])
    bg = _background(out)
    H, W = out.shape
    mask = out != bg
    n_cells = int(mask.sum())

    # Connected components
    labels, n_cc = cc_label(mask, structure=np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    cc_areas = [int((labels == k).sum()) for k in range(1, n_cc + 1)]

    colors_out = set(int(c) for c in np.unique(out) if int(c) != bg)
    colors_in = set(int(c) for c in np.unique(inp) if int(c) != _background(inp))
    new_colors = colors_out - colors_in

    return {
        "task_id": task_id,
        "grid_h": H, "grid_w": W,
        "n_cells_total": H * W,
        "n_cells_nonbg": n_cells,
        "frac_nonbg": n_cells / (H * W),
        "n_colors_output": len(colors_out),
        "n_colors_input": len(colors_in),
        "n_new_colors": len(new_colors),
        "n_success_cc": n_cc,
        "mean_cc_area": float(np.mean(cc_areas)) if cc_areas else 0.0,
        "max_cc_area": int(max(cc_areas)) if cc_areas else 0,
        "input_out_size_ratio": (
            (inp.shape[0] * inp.shape[1]) / max(H * W, 1)
        ),
        "symmetry_h": _symmetry_score(out, "h"),
        "symmetry_v": _symmetry_score(out, "v"),
        "symmetry_d": _symmetry_score(out, "d"),
    }


# ---------------------------------------------------------------------------
# Affordance per strategy
# ---------------------------------------------------------------------------

def affordance(task_id: str) -> Dict:
    """For each strategy, record the number of chunks it would produce
    on this task. >=2 means the strategy is 'afforded' (non-degenerate)."""
    success = _success_grid(task_id)
    out = {"task_id": task_id}
    for sname, fn in STRATEGIES.items():
        chunks = fn(success)
        out[f"n_chunks_{sname}"] = len(chunks)
        out[f"afforded_{sname}"] = int(len(chunks) >= 2)
    out["n_afforded"] = sum(out[f"afforded_{s}"] for s in STRATEGIES)
    return out


# ---------------------------------------------------------------------------
# Per-subject best strategy per task
# ---------------------------------------------------------------------------

def subject_level_best_strategy(chunks_csv: str,
                                strategy_csv: str) -> pd.DataFrame:
    """For every (subject, task) pair, compute the subject's per-chunk
    feature fingerprint and find the canonical strategy whose pooled
    fingerprint (from strategy_chunks_per_task.csv) is closest."""
    chunks = pd.read_csv(chunks_csv)
    strat = pd.read_csv(strategy_csv)
    # strat has one row per (task, strategy). We want per-task-strategy
    # feature vectors for the 6 canonical strategies only.
    strat_s = strat[strat.strategy.isin(STRATEGIES.keys())]
    feat_cols = [
        "frac_single_color", "frac_connected", "frac_same_row",
        "frac_same_col", "frac_multi_cc_spanned",
        "frac_multi_cc_same_color", "frac_one_cc",
        "mean_iou_cc", "mean_iou_color_class",
    ]
    # Z-score each feature column across the 75 tasks × 6 strategies pool
    # so the distance metric is comparable to the task-level comparison.
    means = strat_s[feat_cols].mean()
    stds = strat_s[feat_cols].std(ddof=0) + 1e-9

    def _z(v):
        return (v - means) / stds

    # Aggregate human chunks per (subject, task)
    def _subj_vec(sub: pd.DataFrame) -> Dict:
        n_cc = sub["n_success_cc_spanned"].values
        homo = sub["color_homogeneity"].values
        return {
            "frac_single_color": float((homo >= 0.95).mean()),
            "frac_connected": float(sub["is_connected"].mean()),
            "frac_same_row": float(sub["same_row"].mean()),
            "frac_same_col": float(sub["same_col"].mean()),
            "frac_multi_cc_spanned": float((n_cc >= 2).mean()),
            "frac_multi_cc_same_color": float(
                ((n_cc >= 2) & (homo >= 0.95)).mean()),
            "frac_one_cc": float((n_cc == 1).mean()),
            "mean_iou_cc": float(sub["success_iou_best"].mean()),
            "mean_iou_color_class": float(
                sub["success_iou_color_class"].mean()),
        }

    rows = []
    for (tid, subj), sub in chunks.groupby(["task_id", "subject_id"]):
        if len(sub) < 2:
            continue
        subj_vec = _subj_vec(sub)
        subj_z = _z(pd.Series(subj_vec))
        # For this task find closest strategy
        t_strats = strat_s[strat_s.task_id == tid]
        best_name, best_dist = None, np.inf
        for _, row in t_strats.iterrows():
            s_vec = _z(row[feat_cols])
            dist = float(np.sqrt(((subj_z - s_vec) ** 2).sum()))
            if dist < best_dist:
                best_dist = dist; best_name = row.strategy
        rows.append({
            "task_id": tid, "subject_id": subj,
            "best_strategy": best_name,
            "distance": best_dist,
            "n_chunks_subj": int(len(sub)),
        })
    return pd.DataFrame(rows)


def per_task_entropy(subj_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tid, sub in subj_df.groupby("task_id"):
        counts = sub["best_strategy"].value_counts()
        probs = counts.values / counts.sum()
        ent = float(entropy(probs, base=2))
        dom = counts.iloc[0]
        rows.append({
            "task_id": tid,
            "n_subjects": int(counts.sum()),
            "n_distinct_strategies_chosen": int(len(counts)),
            "human_entropy_bits": ent,
            "dominant_frac": float(dom / counts.sum()),
            "dominant_strategy": str(counts.idxmax()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    os.makedirs("prior_analysis", exist_ok=True)

    # 1. Task-intrinsic features
    rows = [task_features(tid) for tid in available_task_ids()]
    tfeat = pd.DataFrame(rows)
    # Merge in style features (used_copy, n_showex, n_reset, n_edits)
    if os.path.exists("prior_analysis/style_features_all75.csv"):
        style = pd.read_csv("prior_analysis/style_features_all75.csv")
        keep = ["task_id", "used_copy_mean", "n_showex_mean",
                "n_reset_mean", "n_edits_mean"]
        tfeat = tfeat.merge(style[keep], on="task_id", how="left")

    # 2. Affordance per task
    aff_rows = [affordance(tid) for tid in available_task_ids()]
    aff = pd.DataFrame(aff_rows)

    # 3. Subject-level best strategy + per-task entropy
    subj_df = subject_level_best_strategy(
        "prior_analysis/chunks_per_trajectory.csv",
        "prior_analysis/strategy_chunks_per_task.csv",
    )
    ent = per_task_entropy(subj_df)

    merged = tfeat.merge(aff, on="task_id").merge(ent, on="task_id", how="left")
    merged.to_csv("prior_analysis/task_features_for_strategy.csv", index=False)
    subj_df.to_csv("prior_analysis/subject_best_strategy.csv", index=False)
    print(f"wrote task_features_for_strategy.csv  ({len(merged)} tasks)")
    print(f"wrote subject_best_strategy.csv       "
          f"({len(subj_df)} subject-task pairs)")

    # 4. Correlate task features with human entropy and dominant strategy
    predictors = [
        "grid_h", "grid_w", "n_cells_nonbg", "frac_nonbg",
        "n_colors_output", "n_colors_input", "n_new_colors",
        "n_success_cc", "mean_cc_area", "max_cc_area",
        "input_out_size_ratio", "symmetry_h", "symmetry_v",
        "used_copy_mean", "n_showex_mean", "n_reset_mean",
        "n_edits_mean", "n_afforded",
    ]
    targets = ["human_entropy_bits", "dominant_frac"]
    corr_rows = []
    for p in predictors:
        if p not in merged.columns:
            continue
        for t in targets:
            sub = merged.dropna(subset=[p, t])
            if len(sub) < 10:
                continue
            rho, pval = spearmanr(sub[p], sub[t])
            corr_rows.append({
                "predictor": p, "target": t,
                "spearman_rho": float(rho), "p_value": float(pval),
                "n": int(len(sub)),
            })
    corr = pd.DataFrame(corr_rows).sort_values(
        ["target", "spearman_rho"], key=lambda s: s.abs() if s.name == "spearman_rho" else s,
        ascending=[True, False])
    corr.to_csv("prior_analysis/task_vs_strategy_correlations.csv",
                index=False)
    print("\nTop task features predicting HUMAN STRATEGY ENTROPY (higher "
          "entropy = subjects disagree):")
    print(corr[corr.target == "human_entropy_bits"]
          .sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False)
          .head(10).round(3).to_string(index=False))
    print("\nTop task features predicting DOMINANT_FRAC (subjects converge):")
    print(corr[corr.target == "dominant_frac"]
          .sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False)
          .head(10).round(3).to_string(index=False))

    # 5. Per-strategy usage share per task, then correlate with task features.
    #    usage_share[task, s] = fraction of subjects on that task whose
    #    closest canonical strategy is s.
    usage = (subj_df.groupby(["task_id", "best_strategy"]).size()
             .unstack(fill_value=0))
    usage = usage.div(usage.sum(axis=1), axis=0)   # normalise per task
    usage = usage.reset_index()
    usage.columns.name = None
    usage = usage.rename(columns={s: f"usage_{s}" for s in STRATEGIES.keys()
                                  if s in usage.columns})
    merged2 = tfeat.merge(aff, on="task_id").merge(usage, on="task_id",
                                                   how="left")

    usage_cols = [c for c in merged2.columns if c.startswith("usage_")]
    per_strat_rows = []
    for p in predictors:
        if p not in merged2.columns:
            continue
        for u in usage_cols:
            sub = merged2.dropna(subset=[p, u])
            if len(sub) < 10:
                continue
            rho, pval = spearmanr(sub[p], sub[u])
            per_strat_rows.append({
                "predictor": p, "strategy_usage": u,
                "spearman_rho": float(rho), "p_value": float(pval),
                "n": int(len(sub)),
            })
    per_strat = pd.DataFrame(per_strat_rows)
    per_strat.to_csv(
        "prior_analysis/task_vs_strategy_usage_correlations.csv",
        index=False,
    )

    # Pretty-print: per strategy, the top |rho| predictor.
    print("\nStrongest task-feature predictors of SUBJECT USAGE of each "
          "canonical strategy:")
    for u in usage_cols:
        sub = per_strat[per_strat.strategy_usage == u].copy()
        sub["abs_rho"] = sub["spearman_rho"].abs()
        sub = sub.sort_values("abs_rho", ascending=False).head(5)
        print(f"\n-- {u} --")
        print(sub[["predictor", "spearman_rho", "p_value", "n"]]
              .round(3).to_string(index=False))


if __name__ == "__main__":
    main()

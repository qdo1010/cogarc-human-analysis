"""
Do humans prioritize certain objects in a task?

For each CogARC task we segment the Success grid into 4-connected
same-color components (ignoring the background color), then walk every
Experiment 2 participant's edit sequence to record the rank at which
each component is *first touched* by that subject. Averaging across
subjects gives a per-component "touch priority" between 0 (first) and 1
(last).

We then regress touch priority on component features (area, centroid
row/col, color frequency in the task, boundary-to-interior ratio,
distance to grid center) with task fixed effects to identify which
object attributes consistently attract early attention.

Outputs:
    prior_analysis/component_priority_per_task.csv
    prior_analysis/component_priority_feature_regression.csv
"""

from __future__ import annotations


import _paths  # noqa: F401
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import label as cc_label
from scipy.stats import spearmanr

from human_style_features import (
    DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR, _parse_trajectory,
)
from human_targets import human_targets


_FNAME_RE = re.compile(r"^subj_([^_]+)_trial_([0-9a-f]+)\.json\.csv$")


# ---------------------------------------------------------------------------
# Component segmentation + feature extraction on the Success grid
# ---------------------------------------------------------------------------

def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(grid: np.ndarray) -> List[Dict]:
    """Return per-component dicts: pixels (set of (r, c)), bbox, color, area."""
    bg = _background_color(grid)
    H, W = grid.shape
    comps = []
    for color in np.unique(grid):
        if int(color) == bg:
            continue
        mask = (grid == color)
        labels, n = cc_label(mask, structure=np.array(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        for cid in range(1, n + 1):
            coords = np.argwhere(labels == cid)
            if len(coords) < 1:
                continue
            rows, cols = coords[:, 0], coords[:, 1]
            comps.append({
                "color": int(color),
                "pixels": set(map(tuple, coords.tolist())),
                "area": int(len(coords)),
                "centroid_r": float(rows.mean()),
                "centroid_c": float(cols.mean()),
                "bbox_r0": int(rows.min()), "bbox_r1": int(rows.max()),
                "bbox_c0": int(cols.min()), "bbox_c1": int(cols.max()),
                "grid_h": H, "grid_w": W, "bg": bg,
            })
    return comps


def _component_features(comp: Dict, grid: np.ndarray) -> Dict:
    H, W = grid.shape
    cr, cc = comp["centroid_r"], comp["centroid_c"]
    center_r, center_c = (H - 1) / 2.0, (W - 1) / 2.0
    dist_center = float(np.hypot(cr - center_r, cc - center_c))
    # Boundary: fraction of pixels whose 4-neighbourhood leaves the component.
    pixels = comp["pixels"]
    n = len(pixels)
    boundary = 0
    for (r, c) in pixels:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (r + dr, c + dc) not in pixels:
                boundary += 1
                break
    # Color rarity (how rare is this color within the grid)?
    color_count = int((grid == comp["color"]).sum())
    total_non_bg = int((grid != comp["bg"]).sum())
    color_frac = color_count / max(total_non_bg, 1)

    bbox_h = comp["bbox_r1"] - comp["bbox_r0"] + 1
    bbox_w = comp["bbox_c1"] - comp["bbox_c0"] + 1
    fill_ratio = n / max(bbox_h * bbox_w, 1)

    return {
        "area": n,
        "centroid_r": cr, "centroid_c": cc,
        "dist_center": dist_center,
        "boundary_frac": boundary / max(n, 1),
        "color_frac_among_nonbg": color_frac,
        "is_rarest_color": 0.0,  # set below, after seeing all comps
        "is_unique_color": 0.0,  # set below
        "bbox_h": bbox_h, "bbox_w": bbox_w,
        "bbox_aspect": bbox_h / max(bbox_w, 1),
        "fill_ratio": fill_ratio,
        "color": comp["color"],
        "n_pixels_same_color_grid": color_count,
    }


# ---------------------------------------------------------------------------
# First-touch rank per component per subject
# ---------------------------------------------------------------------------

def _first_touch_ranks(comps: List[Dict], edits: List[Dict]) -> List[int]:
    """Returns the 0-based rank (in the edit sequence) at which each
    component's interior is first touched. -1 if never touched."""
    ranks = [-1] * len(comps)
    for r, e in enumerate(edits):
        xy = (e["y"], e["x"])
        for i, c in enumerate(comps):
            if ranks[i] == -1 and xy in c["pixels"]:
                ranks[i] = r
    return ranks


def component_priority(task_id: str,
                       data_root: str = DEFAULT_DATA_ROOT) -> pd.DataFrame:
    """Return one row per component for `task_id`, with features +
    mean touch priority (normalized rank in [0,1], averaged across Exp 2
    subjects; smaller = earlier touched)."""
    t = human_targets(task_id, data_root=data_root)
    success = None
    for lbl, g in zip(t["labels"], t["grids"]):
        if lbl == "Success":
            success = g; break
    if success is None:
        return pd.DataFrame()
    comps = _components(success)
    if not comps:
        return pd.DataFrame()

    feats = [_component_features(c, success) for c in comps]
    # Flag rarest / unique colors
    color_counts = defaultdict(int)
    for c in comps:
        color_counts[c["color"]] += 1
    min_count = min(color_counts.values())
    for f, c in zip(feats, comps):
        f["is_unique_color"] = float(color_counts[c["color"]] == 1)
        f["is_rarest_color"] = float(color_counts[c["color"]] == min_count)

    # Walk trajectories
    traj_dir = os.path.join(data_root, _EXP2_EDIT_DIR, f"{task_id}.json")
    per_comp_ranks: List[List[int]] = [[] for _ in comps]
    n_subj = 0
    if os.path.isdir(traj_dir):
        for fname in os.listdir(traj_dir):
            m = _FNAME_RE.match(fname)
            if not m:
                continue
            tr = _parse_trajectory(os.path.join(traj_dir, fname))
            if not tr["edits"]:
                continue
            n_subj += 1
            ranks = _first_touch_ranks(comps, tr["edits"])
            total = len(tr["edits"])
            # Normalize so rank 0 -> 0.0, last-possible -> 1.0.
            for i, r in enumerate(ranks):
                if r >= 0 and total > 1:
                    per_comp_ranks[i].append(r / (total - 1))
                elif r == 0:
                    per_comp_ranks[i].append(0.0)

    rows = []
    for i, (c, f) in enumerate(zip(comps, feats)):
        vals = per_comp_ranks[i]
        f = dict(f)
        f.update({
            "task_id": task_id,
            "component_id": i,
            "n_subjects_touched": len(vals),
            "n_subjects_total": n_subj,
            "mean_touch_priority": float(np.mean(vals)) if vals else float("nan"),
            "median_touch_priority": float(np.median(vals)) if vals else float("nan"),
        })
        rows.append(f)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregate regression: which features predict early touch?
# ---------------------------------------------------------------------------

def regress_priority_on_features(all_df: pd.DataFrame) -> pd.DataFrame:
    """OLS of mean_touch_priority on z-scored features WITH task fixed
    effects. We report the per-feature coefficient, Spearman rho with
    priority (non-parametric), and coverage stats."""
    df = all_df.dropna(subset=["mean_touch_priority"]).copy()
    if df.empty:
        return pd.DataFrame()

    feat_cols = [
        "area", "dist_center", "boundary_frac",
        "color_frac_among_nonbg", "is_rarest_color", "is_unique_color",
        "bbox_h", "bbox_w", "bbox_aspect", "fill_ratio",
    ]
    # z-score within-task so we can pool fairly
    for c in feat_cols:
        df[c + "_z"] = df.groupby("task_id")[c].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))
    # De-mean the target per task (task fixed effect)
    df["priority_z"] = df.groupby("task_id")["mean_touch_priority"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) + 1e-9))

    rows = []
    for c in feat_cols:
        mask = df[c + "_z"].notna() & df["priority_z"].notna()
        if mask.sum() < 10:
            continue
        # Simple OLS slope (intercept is 0 by construction after z-score)
        x = df.loc[mask, c + "_z"].values
        y = df.loc[mask, "priority_z"].values
        beta = float((x @ y) / (x @ x)) if (x @ x) > 0 else float("nan")
        # Spearman for robustness
        rho, p = spearmanr(x, y)
        rows.append({
            "feature": c,
            "beta_within_task": beta,
            "spearman_rho": float(rho), "spearman_p": float(p),
            "n": int(mask.sum()),
        })
    out = pd.DataFrame(rows).sort_values("beta_within_task",
                                         key=lambda s: s.abs(),
                                         ascending=False)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    from human_targets import available_task_ids
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task_ids", nargs="+", default=None)
    ap.add_argument("--out_dir", default="prior_analysis")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ids = args.task_ids or available_task_ids()

    frames = []
    for i, tid in enumerate(ids, 1):
        df = component_priority(tid)
        if not df.empty:
            frames.append(df)
        if i % 10 == 0:
            print(f"  processed {i}/{len(ids)} tasks")
    if not frames:
        print("no components recovered"); return
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(os.path.join(args.out_dir,
                               "component_priority_per_task.csv"), index=False)
    print(f"wrote component_priority_per_task.csv  "
          f"({len(all_df)} components across {all_df['task_id'].nunique()} tasks)")

    reg = regress_priority_on_features(all_df)
    reg.to_csv(os.path.join(args.out_dir,
                            "component_priority_feature_regression.csv"),
               index=False)
    print("\n--- Which features predict EARLY touch?\n"
          "    (negative beta = higher value -> earlier; positive = later) ---")
    print(reg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

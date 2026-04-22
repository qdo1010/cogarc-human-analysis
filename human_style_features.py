"""
Compute per-task behavioral style features from CogARC Experiment 2
edit-sequence CSVs. The features summarize *how* humans drew the
submission — not just the final grid — so we can heuristically pick
graph priors before running a full optimization sweep.

Public API:
    style_features(task_id, data_root=DEFAULT_DATA_ROOT,
                   first_attempt_only=True) -> dict

    all_style_features(task_ids=None) -> pd.DataFrame  # one row per task

    recommend_priors(features_row, top_k=3) -> List[str]
        Heuristic mapping from features to candidate graph priors
        (see heuristic details in the function docstring).
"""

from __future__ import annotations

import _paths  # noqa: F401  (sys.path bootstrap, safe under both invocations)

import csv
import math
import os
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from human_targets import DEFAULT_DATA_ROOT, available_task_ids


_EXP2_EDIT_DIR = os.path.join("Behavioral data", "Experiment 2", "Edit sequences")


# ---------------------------------------------------------------------------
# Parsing a single trajectory
# ---------------------------------------------------------------------------

def _parse_trajectory(csv_path: str) -> Dict:
    """
    Parse one edit-sequence CSV. Returns a dict with the sequence split
    by action type plus the raw edit list in temporal order.
    """
    edits: List[Dict] = []
    n_reset = 0
    n_showex = 0
    n_hideex = 0
    n_submit = 0
    used_copy = False
    first_action_after_new = None
    times: List[float] = []
    rts: List[float] = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        seen_new = False
        for row in reader:
            a = (row.get("action") or "").strip()
            if not a:
                continue
            if a == "new":
                seen_new = True
                continue
            if seen_new and first_action_after_new is None:
                first_action_after_new = a
            if a == "reset":
                n_reset += 1
            elif a == "showex":
                n_showex += 1
            elif a == "hideex":
                n_hideex += 1
            elif a == "copy":
                used_copy = True
            elif a == "submit":
                n_submit += 1
            elif a == "edit":
                try:
                    x = int(float(row["x"]))
                    y = int(float(row["y"]))
                    c = int(float(row["color"]))
                    t = float(row["time"])
                    rt_s = row.get("rt", "").strip()
                    rt = float(rt_s) if rt_s else float("nan")
                except (ValueError, TypeError, KeyError):
                    continue
                edits.append({"x": x, "y": y, "color": c, "time": t, "rt": rt})
                times.append(t)
                if not math.isnan(rt):
                    rts.append(rt)

    return {
        "edits": edits,
        "n_reset": n_reset,
        "n_showex": n_showex,
        "n_hideex": n_hideex,
        "n_submit": n_submit,
        "used_copy": used_copy,
        "first_action_after_new": first_action_after_new,
        "times": times,
        "rts": rts,
    }


def _run_lengths(seq: List) -> List[int]:
    """Lengths of maximal runs of equal values."""
    if not seq:
        return []
    out = []
    cur_len = 1
    for i in range(1, len(seq)):
        if seq[i] == seq[i - 1]:
            cur_len += 1
        else:
            out.append(cur_len)
            cur_len = 1
    out.append(cur_len)
    return out


def _scan_order_score(edits: List[Dict]) -> Dict[str, float]:
    """
    Spearman-like correlation between edit order and several 1-D indices:
      row_major = y*W + x, col_major = x*H + y.
    Returns a dict with the best-matching name and its rank correlation.
    """
    if len(edits) < 4:
        return {"row_major_corr": 0.0, "col_major_corr": 0.0,
                "best_scan": "none", "best_scan_corr": 0.0}
    xs = np.array([e["x"] for e in edits])
    ys = np.array([e["y"] for e in edits])
    order = np.arange(len(edits))
    W = max(xs.max() - xs.min() + 1, 1)
    H = max(ys.max() - ys.min() + 1, 1)
    rm = ys * W + xs
    cm = xs * H + ys

    def _rank_corr(a, b):
        ra = np.argsort(np.argsort(a))
        rb = np.argsort(np.argsort(b))
        if np.std(ra) == 0 or np.std(rb) == 0:
            return 0.0
        return float(np.corrcoef(ra, rb)[0, 1])

    row_corr = _rank_corr(order, rm)
    col_corr = _rank_corr(order, cm)
    if abs(row_corr) >= abs(col_corr):
        best, best_v = "row_major", row_corr
    else:
        best, best_v = "col_major", col_corr
    return {
        "row_major_corr": row_corr,
        "col_major_corr": col_corr,
        "best_scan": best,
        "best_scan_corr": best_v,
    }


def _trajectory_features(traj: Dict) -> Dict:
    edits = traj["edits"]
    f: Dict[str, float] = {
        "n_edits":    len(edits),
        "n_reset":    traj["n_reset"],
        "n_showex":   traj["n_showex"],
        "n_hideex":   traj["n_hideex"],
        "used_copy":  float(traj["used_copy"]),
        "mean_rt_ms": float(np.mean(traj["rts"])) if traj["rts"] else float("nan"),
        "total_time_ms": (edits[-1]["time"] - edits[0]["time"]) if len(edits) > 1 else 0.0,
    }

    if not edits:
        f.update({
            "color_run_mean": 0.0, "color_run_max": 0.0,
            "spatial_run_mean": 0.0, "adjacency_rate": 0.0, "mean_step": 0.0,
            "unique_colors": 0, "row_major_corr": 0.0, "col_major_corr": 0.0,
            "best_scan": "none", "best_scan_corr": 0.0,
            "within_obj_rate": 0.0,
            "extension_rate": 0.0, "new_seed_rate": 0.0,
            "stamp_burst_rate": 0.0, "stamp_burst_cells": 0.0,
            "same_color_component_count": 0, "same_color_component_mean": 0.0,
            "same_color_component_max": 0.0,
        })
        return f

    colors = [e["color"] for e in edits]
    runs_color = _run_lengths(colors)
    f["color_run_mean"] = float(np.mean(runs_color))
    f["color_run_max"] = float(np.max(runs_color))
    f["unique_colors"] = len(set(colors))

    # Spatial adjacency between consecutive edits
    adj = 0
    steps = []
    spatial_run_start = 0
    spatial_runs = []
    for i in range(1, len(edits)):
        dx = abs(edits[i]["x"] - edits[i - 1]["x"])
        dy = abs(edits[i]["y"] - edits[i - 1]["y"])
        step = max(dx, dy)  # Chebyshev (8-connectivity distance)
        steps.append(step)
        if step <= 1:
            adj += 1
        else:
            spatial_runs.append(i - spatial_run_start)
            spatial_run_start = i
    spatial_runs.append(len(edits) - spatial_run_start)
    f["adjacency_rate"] = adj / max(len(edits) - 1, 1)
    f["mean_step"] = float(np.mean(steps)) if steps else 0.0
    f["spatial_run_mean"] = float(np.mean(spatial_runs))

    # Rate of consecutive edits where BOTH color and spatial are continuous —
    # a proxy for "paint inside an object then move on"
    within = 0
    for i in range(1, len(edits)):
        dx = abs(edits[i]["x"] - edits[i - 1]["x"])
        dy = abs(edits[i]["y"] - edits[i - 1]["y"])
        if max(dx, dy) <= 1 and edits[i]["color"] == edits[i - 1]["color"]:
            within += 1
    f["within_obj_rate"] = within / max(len(edits) - 1, 1)

    # --- Edit-propagation features ---
    # For each edit, decide whether it EXTENDS an existing same-color region
    # (4-adjacent to any previously-edited same-color cell), or SEEDS a new
    # disjoint region. Also count same-color connected components formed by
    # all edits together (a stable "stamp count" proxy).
    per_color_cells: Dict[int, set] = {}
    extensions = 0
    seeds = 0
    for e in edits:
        c, x, y = e["color"], e["x"], e["y"]
        cells = per_color_cells.setdefault(c, set())
        if not cells:
            seeds += 1
        else:
            # 4-adjacency: any of (x+-1,y) or (x,y+-1) already painted same color?
            if ((x - 1, y) in cells or (x + 1, y) in cells
                    or (x, y - 1) in cells or (x, y + 1) in cells):
                extensions += 1
            elif (x, y) not in cells:
                seeds += 1
            # if (x,y) already in cells the participant is re-coloring; count as
            # neither extension nor seed.
        cells.add((x, y))

    denom = max(extensions + seeds, 1)
    f["extension_rate"] = extensions / denom
    f["new_seed_rate"]  = seeds / denom

    # Same-color connected-component stats over the FINAL painted cells
    comp_sizes: List[int] = []
    for c, cells in per_color_cells.items():
        remaining = set(cells)
        while remaining:
            stack = [remaining.pop()]
            size = 1
            while stack:
                x, y = stack.pop()
                for nb in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                    if nb in remaining:
                        remaining.remove(nb)
                        stack.append(nb)
                        size += 1
            comp_sizes.append(size)
    f["same_color_component_count"] = len(comp_sizes)
    f["same_color_component_mean"] = float(np.mean(comp_sizes)) if comp_sizes else 0.0
    f["same_color_component_max"]  = float(np.max(comp_sizes)) if comp_sizes else 0.0

    # Stamp-burst: consecutive edits within a tight RT window of each other
    # AND same-color AND spatially adjacent. Bursts of length >=3 count as a
    # "stamp". Proxy for quickly painting a pre-mentalized shape.
    burst_cells = 0
    n_bursts = 0
    cur_burst = 1
    BURST_RT_MS = 500.0
    for i in range(1, len(edits)):
        dt = edits[i]["time"] - edits[i - 1]["time"]
        dx = abs(edits[i]["x"] - edits[i - 1]["x"])
        dy = abs(edits[i]["y"] - edits[i - 1]["y"])
        same_color = edits[i]["color"] == edits[i - 1]["color"]
        if dt <= BURST_RT_MS and max(dx, dy) <= 1 and same_color:
            cur_burst += 1
        else:
            if cur_burst >= 3:
                burst_cells += cur_burst
                n_bursts += 1
            cur_burst = 1
    if cur_burst >= 3:
        burst_cells += cur_burst
        n_bursts += 1
    f["stamp_burst_cells"] = float(burst_cells)
    f["stamp_burst_rate"]  = burst_cells / max(len(edits), 1)

    # Scan-order signature
    f.update(_scan_order_score(edits))

    return f


# ---------------------------------------------------------------------------
# Per-task aggregation
# ---------------------------------------------------------------------------

def _list_trajectory_files(task_id: str, data_root: str,
                           first_attempt_only: bool) -> List[str]:
    root = os.path.join(data_root, _EXP2_EDIT_DIR, f"{task_id}.json")
    if not os.path.isdir(root):
        return []
    files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".csv")]
    if first_attempt_only:
        # subj_<id>_trial_<task_id>.json.csv  — all files ARE the trial for this
        # task; Exp 2 typically logs one trial per subject-task. Keep as is.
        pass
    return sorted(files)


@lru_cache(maxsize=None)
def style_features(task_id: str,
                   data_root: str = DEFAULT_DATA_ROOT,
                   first_attempt_only: bool = True) -> Dict:
    """
    Aggregate per-subject trajectory features for one task into a single
    row of population-level features (means, and a few categorical rates).
    """
    files = _list_trajectory_files(task_id, data_root, first_attempt_only)
    if not files:
        return {"task_id": task_id, "n_subjects": 0}

    per_subj = [_trajectory_features(_parse_trajectory(p)) for p in files]

    agg: Dict[str, float] = {"task_id": task_id, "n_subjects": len(per_subj)}
    numeric_keys = [
        "n_edits", "n_reset", "n_showex", "n_hideex", "used_copy", "mean_rt_ms",
        "total_time_ms", "color_run_mean", "color_run_max", "spatial_run_mean",
        "adjacency_rate", "mean_step", "within_obj_rate", "unique_colors",
        "row_major_corr", "col_major_corr", "best_scan_corr",
        "extension_rate", "new_seed_rate", "stamp_burst_rate", "stamp_burst_cells",
        "same_color_component_count", "same_color_component_mean",
        "same_color_component_max",
    ]
    for k in numeric_keys:
        vals = [r[k] for r in per_subj if k in r and not (isinstance(r[k], float) and math.isnan(r[k]))]
        agg[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")

    scan_counts = Counter(r.get("best_scan", "none") for r in per_subj)
    agg["scan_row_major_rate"] = scan_counts.get("row_major", 0) / len(per_subj)
    agg["scan_col_major_rate"] = scan_counts.get("col_major", 0) / len(per_subj)

    first_actions = Counter(r.get("first_action_after_new")
                            for r in [_parse_trajectory(p) for p in files])
    agg["first_action_copy_rate"] = first_actions.get("copy", 0) / len(per_subj)
    agg["first_action_reset_rate"] = first_actions.get("reset", 0) / len(per_subj)
    agg["first_action_showex_rate"] = first_actions.get("showex", 0) / len(per_subj)

    return agg


def all_style_features(task_ids: Optional[Iterable[str]] = None,
                       data_root: str = DEFAULT_DATA_ROOT) -> pd.DataFrame:
    ids = list(task_ids) if task_ids is not None else available_task_ids(data_root)
    rows = [style_features(tid, data_root=data_root) for tid in ids]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Heuristic: features -> ranked candidate priors
# ---------------------------------------------------------------------------

def _task_zscores(row: Dict, pop: pd.DataFrame) -> Dict[str, float]:
    """Z-score this task's features against the 75-task population."""
    z = {}
    for col in ["n_edits_mean", "n_reset_mean", "n_showex_mean",
                "color_run_mean_mean", "within_obj_rate_mean",
                "adjacency_rate_mean", "best_scan_corr_mean",
                "unique_colors_mean", "used_copy_mean",
                "extension_rate_mean", "new_seed_rate_mean",
                "stamp_burst_rate_mean", "stamp_burst_cells_mean",
                "same_color_component_count_mean",
                "same_color_component_mean_mean",
                "same_color_component_max_mean"]:
        if col not in pop.columns:
            continue
        mu = pop[col].mean()
        sd = pop[col].std(ddof=0)
        v = row.get(col, mu)
        z[col] = 0.0 if sd == 0 else float((v - mu) / sd)
    return z


def recommend_priors(row: Dict,
                     population: Optional[pd.DataFrame] = None,
                     top_k: int = 4) -> List[str]:
    """
    Rank candidate graph priors for one task from its behavioral style
    features, using *relative* (z-score) comparisons against the 75-task
    population so that tasks differ meaningfully instead of all saturating
    the same absolute thresholds.

    If `population` is None the 75-task table is computed lazily.

    Mapping (each signal is a soft vote):
      - edits-per-subject >> population       : "pixel-by-pixel laborious"
                                              -> growth, containment, color_map
      - color_run >> pop (while edits moderate): "by-color reasoning"
                                              -> color_adjacency, color_map
      - within_obj_rate >> pop                : "paint inside one object"
                                              -> hierarchical, containment
      - scan_corr >> pop                      : "row/col scan"
                                              -> symmetry, grid_partition, projection
      - showex >> pop                         : "heavy example-viewing"
                                              -> shape_similarity, stamp
      - used_copy << pop                      : "rarely uses input grid"
                                              -> symmetry, grid_partition (from-scratch)
      - unique_colors >> pop                  : "many colors"
                                              -> color_adjacency, color_map
      - otherwise                             : fallback hierarchical
    """
    scores: Dict[str, float] = defaultdict(float)

    if population is None:
        population = all_style_features()
    z = _task_zscores(row, population)

    # Vote weights deliberately close to 1 so two coinciding signals dominate
    if z.get("n_edits_mean", 0) > 1.0:
        scores["growth"] += 1.0
        scores["containment"] += 0.6
        scores["color_map"] += 0.4

    if z.get("color_run_mean_mean", 0) > 1.0 and z.get("n_edits_mean", 0) <= 1.0:
        scores["color_adjacency"] += 1.0
        scores["color_map"] += 0.6

    if z.get("within_obj_rate_mean", 0) > 0.5:
        scores["hierarchical"] += 1.0
        scores["containment"] += 0.5

    if z.get("best_scan_corr_mean", 0) > 0.8:
        scores["symmetry"] += 1.0
        scores["grid_partition"] += 0.7
        scores["projection"] += 0.4

    if z.get("n_showex_mean", 0) > 1.0:
        scores["shape_similarity"] += 1.0
        scores["stamp"] += 0.4

    if z.get("used_copy_mean", 0) < -1.0:
        scores["symmetry"] += 0.5
        scores["grid_partition"] += 0.5

    if z.get("unique_colors_mean", 0) > 1.0:
        scores["color_adjacency"] += 0.7
        scores["color_map"] += 0.5

    # --- Edit-propagation signals (added to catch growth/stamp) ---
    # High extension_rate (cell extends an existing same-color region) -> growth.
    if z.get("extension_rate_mean", 0) > 0.5:
        scores["growth"] += 1.0
        scores["containment"] += 0.3
    # Low extension_rate (seeding many disjoint regions) with many components
    # OR high stamp-burst rate -> stamp-like painting of pre-mentalized shapes.
    if (z.get("new_seed_rate_mean", 0) > 0.5
            and z.get("same_color_component_count_mean", 0) > 0.5):
        scores["stamp"] += 1.0
        scores["shape_similarity"] += 0.4
    if z.get("stamp_burst_rate_mean", 0) > 0.8:
        scores["stamp"] += 0.8
        scores["growth"] += 0.3
    # Very large single same-color components -> dense flood-filled regions ->
    # growth AND containment (an object being filled in place).
    if z.get("same_color_component_max_mean", 0) > 1.0:
        scores["growth"] += 0.7
        scores["containment"] += 0.4

    # Hierarchical was the empirical runaway winner on the 3 labeled tasks;
    # keep a small always-on vote so it is never omitted from the top-k.
    scores["hierarchical"] += 0.2

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [name for name, _ in ranked[:top_k]]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_ids", nargs="+", default=None)
    ap.add_argument("--out_csv", default=None, help="If given, write full feature table to CSV.")
    ap.add_argument("--recommend", action="store_true",
                    help="Print recommended priors per task alongside features.")
    args = ap.parse_args()

    df = all_style_features(args.task_ids)

    cols_short = ["task_id", "n_subjects",
                  "n_edits_mean", "n_reset_mean", "n_showex_mean",
                  "color_run_mean_mean", "within_obj_rate_mean",
                  "adjacency_rate_mean", "best_scan_corr_mean",
                  "first_action_copy_rate"]
    print(df[cols_short].to_string(index=False))

    if args.recommend:
        print("\nRecommended priors (top 4) per task:")
        for _, row in df.iterrows():
            rp = recommend_priors(row.to_dict(), top_k=4)
            print(f"  {row['task_id']}  -> {rp}")

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\nFull features -> {args.out_csv}")


if __name__ == "__main__":
    main()

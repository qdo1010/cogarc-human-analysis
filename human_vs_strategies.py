"""
Test whether human chunking matches any canonical "drawing strategy",
or whether the observed chunk pattern is just what the task rule forces.

For each of the 75 CogARC Success grids we generate synthetic drawing
sequences under six strategies:

    random_k3        — cells in random order, chunks of size 3
    color_first      — paint all of colour A, then colour B, ...; one
                       chunk per colour (no intra-colour order)
    nn_color_first   — same, but within each colour paint in
                       nearest-neighbour order starting top-left
    object_first     — one 4-connected component = one chunk,
                       painted in nearest-neighbour order
    row_first        — one row of non-background cells = one chunk
    col_first        — one column of non-background cells = one chunk

Each strategy produces a list of chunks. We run the same chunk-feature
pipeline used for humans (color_homogeneity, is_connected, cc_count,
same_row, same_col, nn_chain_rate, IoU with Success components, number
of CCs spanned) and aggregate per task.

Then we compare each strategy's per-task feature vector against the
humans' per-task feature vector. If humans match one strategy much
more closely than the others (or more closely than task-independent
baselines), that strategy is a plausible model of their chunking.

Outputs:
    prior_analysis/strategy_chunks_per_task.csv
        One row per (task, strategy) with chunk-level summaries.
    prior_analysis/strategy_vs_human_distance.csv
        Per-task Euclidean distance between human and each strategy's
        feature vector (normalised by feature standard deviations).
"""

from __future__ import annotations


import _paths  # noqa: F401

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import label as cc_label

from human_chunking import (
    _success_components, best_iou_with_success, chunk_features,
    color_class_iou, n_success_cc_spanned,
)
from human_targets import available_task_ids, human_targets


# ---------------------------------------------------------------------------
# Success-grid utilities
# ---------------------------------------------------------------------------

def _success_grid(task_id: str) -> np.ndarray:
    t = human_targets(task_id)
    for lbl, g in zip(t["labels"], t["grids"]):
        if lbl == "Success":
            return g
    return np.array([])


def _non_bg_cells(grid: np.ndarray) -> List[Tuple[int, int, int]]:
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    out = []
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            c = int(grid[y, x])
            if c != bg:
                out.append((y, x, c))
    return out


def _to_edits(yxc: List[Tuple[int, int, int]], t0: float = 0.0) -> List[Dict]:
    """Convert (y, x, color) triples into edit dicts with fake time/rt."""
    out = []
    for i, (y, x, c) in enumerate(yxc):
        out.append({"x": int(x), "y": int(y), "color": int(c),
                    "time": t0 + i * 500.0, "rt": 500.0})
    return out


def _nn_order(cells: List[Tuple[int, int, int]],
              start: Tuple[int, int]) -> List[Tuple[int, int, int]]:
    """Greedy nearest-neighbour traversal. start = (y, x) to begin near."""
    if not cells:
        return []
    remaining = list(cells)
    # pick the cell closest to `start`
    dists = [abs(c[0] - start[0]) + abs(c[1] - start[1]) for c in remaining]
    idx = int(np.argmin(dists))
    ordered = [remaining.pop(idx)]
    while remaining:
        cy, cx = ordered[-1][0], ordered[-1][1]
        dists = [abs(c[0] - cy) + abs(c[1] - cx) for c in remaining]
        idx = int(np.argmin(dists))
        ordered.append(remaining.pop(idx))
    return ordered


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def strategy_random_k3(grid: np.ndarray, seed: int = 0) -> List[List[Dict]]:
    cells = _non_bg_cells(grid)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(cells))
    cells = [cells[i] for i in perm]
    edits = _to_edits(cells)
    return [edits[i:i + 3] for i in range(0, len(edits), 3)]


def strategy_color_first(grid: np.ndarray) -> List[List[Dict]]:
    cells = _non_bg_cells(grid)
    by_color: Dict[int, List] = {}
    for c in cells:
        by_color.setdefault(c[2], []).append(c)
    chunks = []
    for color, cs in sorted(by_color.items()):
        chunks.append(_to_edits(cs))
    return chunks


def strategy_nn_color_first(grid: np.ndarray) -> List[List[Dict]]:
    cells = _non_bg_cells(grid)
    by_color: Dict[int, List] = {}
    for c in cells:
        by_color.setdefault(c[2], []).append(c)
    chunks = []
    for color, cs in sorted(by_color.items()):
        chunks.append(_to_edits(_nn_order(cs, start=(0, 0))))
    return chunks


def strategy_object_first(grid: np.ndarray) -> List[List[Dict]]:
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    chunks = []
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    for color in np.unique(grid):
        if int(color) == bg:
            continue
        mask = (grid == color)
        labels, n = cc_label(mask, structure=struct)
        for cid in range(1, n + 1):
            coords = np.argwhere(labels == cid)
            cells = [(int(r), int(c), int(color)) for (r, c) in coords]
            chunks.append(_to_edits(_nn_order(cells, start=(0, 0))))
    return chunks


def strategy_row_first(grid: np.ndarray) -> List[List[Dict]]:
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    chunks = []
    for y in range(grid.shape[0]):
        row = [(y, x, int(grid[y, x]))
               for x in range(grid.shape[1]) if int(grid[y, x]) != bg]
        if row:
            chunks.append(_to_edits(row))
    return chunks


def strategy_col_first(grid: np.ndarray) -> List[List[Dict]]:
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    chunks = []
    for x in range(grid.shape[1]):
        col = [(y, x, int(grid[y, x]))
               for y in range(grid.shape[0]) if int(grid[y, x]) != bg]
        if col:
            chunks.append(_to_edits(col))
    return chunks


STRATEGIES = {
    "random_k3":        strategy_random_k3,
    "color_first":      strategy_color_first,
    "nn_color_first":   strategy_nn_color_first,
    "object_first":     strategy_object_first,
    "row_first":        strategy_row_first,
    "col_first":        strategy_col_first,
}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

FEATURES_OF_INTEREST = [
    "mean_chunk_size",
    "frac_single_color",
    "frac_connected",
    "frac_same_row",
    "frac_same_col",
    "frac_multi_cc_spanned",
    "frac_multi_cc_same_color",
    "frac_one_cc",
    "mean_iou_cc",
    "mean_iou_color_class",
]


def _summarize_chunks(chunks: List[List[Dict]],
                      comp_masks: List[set],
                      color_sets: Dict[int, set]) -> Dict:
    if not chunks:
        return {k: float("nan") for k in FEATURES_OF_INTEREST}
    sizes = []
    homo = []
    conn = []
    row = []
    col = []
    ious_cc = []
    ious_col = []
    n_cc_spanned_list = []
    for ch in chunks:
        feats = chunk_features(ch)
        sizes.append(feats["size"])
        homo.append(feats["color_homogeneity"])
        conn.append(int(feats["is_connected"]))
        row.append(int(feats["same_row"]))
        col.append(int(feats["same_col"]))
        iou_cc, _ = best_iou_with_success(ch, comp_masks)
        iou_color, _ = color_class_iou(ch, color_sets)
        ious_cc.append(iou_cc)
        ious_col.append(iou_color)
        n_cc_spanned_list.append(n_success_cc_spanned(ch, comp_masks))

    n_cc_spanned_arr = np.array(n_cc_spanned_list)
    return {
        "mean_chunk_size": float(np.mean(sizes)),
        "frac_single_color": float(np.mean(np.array(homo) >= 0.95)),
        "frac_connected": float(np.mean(conn)),
        "frac_same_row": float(np.mean(row)),
        "frac_same_col": float(np.mean(col)),
        "frac_multi_cc_spanned": float(np.mean(n_cc_spanned_arr >= 2)),
        "frac_multi_cc_same_color": float(np.mean(
            (n_cc_spanned_arr >= 2) & (np.array(homo) >= 0.95))),
        "frac_one_cc": float(np.mean(n_cc_spanned_arr == 1)),
        "mean_iou_cc": float(np.mean(ious_cc)),
        "mean_iou_color_class": float(np.mean(ious_col)),
    }


def run() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # --- Per-strategy per-task summaries ---
    rows = []
    ids = available_task_ids()
    for i, tid in enumerate(ids, 1):
        success = _success_grid(tid)
        if success.size == 0:
            continue
        _, comp_masks, color_sets = _success_components(tid)
        for sname, fn in STRATEGIES.items():
            chunks = fn(success)
            summary = _summarize_chunks(chunks, comp_masks, color_sets)
            summary["task_id"] = tid
            summary["strategy"] = sname
            rows.append(summary)
        if i % 15 == 0:
            print(f"  processed {i}/{len(ids)} tasks")
    strat_df = pd.DataFrame(rows)

    # --- Per-task human summaries from the chunks CSV ---
    human_df = pd.read_csv("prior_analysis/chunks_per_trajectory.csv")
    per_task_human_rows = []
    for tid, sub in human_df.groupby("task_id"):
        n_cc = sub["n_success_cc_spanned"].values
        homo = sub["color_homogeneity"].values
        per_task_human_rows.append({
            "task_id": tid,
            "strategy": "human",
            "mean_chunk_size": float(sub["size"].mean()),
            "frac_single_color": float((homo >= 0.95).mean()),
            "frac_connected": float(sub["is_connected"].mean()),
            "frac_same_row": float(sub["same_row"].mean()),
            "frac_same_col": float(sub["same_col"].mean()),
            "frac_multi_cc_spanned": float((n_cc >= 2).mean()),
            "frac_multi_cc_same_color": float(((n_cc >= 2) & (homo >= 0.95)).mean()),
            "frac_one_cc": float((n_cc == 1).mean()),
            "mean_iou_cc": float(sub["success_iou_best"].mean()),
            "mean_iou_color_class": float(sub["success_iou_color_class"].mean()),
        })
    human_df_summary = pd.DataFrame(per_task_human_rows)
    all_df = pd.concat([strat_df, human_df_summary], ignore_index=True)

    all_df.to_csv("prior_analysis/strategy_chunks_per_task.csv", index=False)
    print(f"wrote prior_analysis/strategy_chunks_per_task.csv  "
          f"({len(all_df)} rows = 75 tasks × {len(STRATEGIES)+1} strategies)")

    # --- Distance (per task) between each strategy's feature vector and humans ---
    pivot = all_df.pivot_table(
        index="task_id", columns="strategy", values=FEATURES_OF_INTEREST
    )
    # Column-level z-scoring over the 75 tasks so features with wider
    # ranges don't dominate the Euclidean distance.
    pivot_z = pivot.apply(
        lambda col: (col - col.mean()) / (col.std(ddof=0) + 1e-9),
        axis=0,
    )
    distances = []
    for tid in pivot.index:
        if "human" not in pivot.columns.get_level_values("strategy"):
            continue
        for feat in FEATURES_OF_INTEREST:
            pass  # zgetter just below
        for sname in STRATEGIES:
            diffs = []
            for feat in FEATURES_OF_INTEREST:
                h = pivot_z.loc[tid, (feat, "human")]
                s = pivot_z.loc[tid, (feat, sname)]
                if np.isnan(h) or np.isnan(s):
                    continue
                diffs.append((h - s) ** 2)
            dist = float(np.sqrt(np.sum(diffs))) if diffs else float("nan")
            distances.append({"task_id": tid, "strategy": sname, "distance": dist})
    dist_df = pd.DataFrame(distances)
    dist_df.to_csv("prior_analysis/strategy_vs_human_distance.csv", index=False)
    print(f"wrote prior_analysis/strategy_vs_human_distance.csv")

    print("\n--- Mean distance (across 75 tasks) from humans ---")
    print(dist_df.groupby("strategy")["distance"].agg(["mean", "median", "std"])
          .round(3).sort_values("mean").to_string())

    return all_df, dist_df


if __name__ == "__main__":
    run()

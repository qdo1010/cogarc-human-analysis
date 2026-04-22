"""
Per-task sequence / attention analysis from Experiment 2 edit trajectories.

For each of the 75 CogARC tasks we compute four signals about HOW the
drawing is ordered:

  1. first_edit_heatmap(task_id) -> (H, W) float32
     For each cell, the fraction of subjects whose FIRST edit on that
     task landed on that cell. High-value cells = where people start,
     i.e. an empirical saliency/attention map for the task.

  2. order_consistency(task_id) -> scalar in [-1, 1]
     Average pairwise Spearman between subjects' cell-visit orderings.
     High = participants agree on the order in which cells are touched
     (canonical strategy). Low or near-zero = participants draw in
     idiosyncratic/random orders.

  3. color_priority(task_id) -> list of (color, rank_sum, n_subjects)
     For each color, the mean 1-based rank at which subjects first used
     it. Ordering this list gives the pooled "color priority sequence".

  4. attention_graph(task_id, threshold=3) -> sparse (H, W, H, W)-ish
     transition counts. Per subject, each pair of consecutive edits
     (cell_t -> cell_{t+1}) increments one entry. Summed across subjects
     this is a weighted directed graph over cells; thresholding and
     normalizing gives an empirical prior over cell-to-cell edges that
     the GNN can use as an edge prior.

Public CLI: computes all four for every task, writes
    prior_analysis/sequence_features.csv            (scalars per task)
    prior_analysis/first_edit_heatmaps.npz          (HxW map per task)
    prior_analysis/attention_graphs.npz             (edge counts per task)
"""

from __future__ import annotations

import _paths  # noqa: F401  (sys.path bootstrap, safe under both invocations)

import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from human_style_features import (
    DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR, _parse_trajectory,
)
from human_targets import human_targets


_FNAME_RE = re.compile(r"^subj_([^_]+)_trial_([0-9a-f]+)\.json\.csv$")


def _task_traj_paths(task_id: str, data_root: str = DEFAULT_DATA_ROOT
                     ) -> List[Tuple[str, str]]:
    root = os.path.join(data_root, _EXP2_EDIT_DIR, f"{task_id}.json")
    out = []
    if not os.path.isdir(root):
        return out
    for f in os.listdir(root):
        m = _FNAME_RE.match(f)
        if m:
            out.append((m.group(1), os.path.join(root, f)))
    return out


def _load_task_shape(task_id: str) -> Tuple[int, int]:
    tgt = human_targets(task_id)
    for lbl, g in zip(tgt["labels"], tgt["grids"]):
        if lbl == "Success":
            return g.shape
    return (10, 10)  # fallback


# ---------------------------------------------------------------------------
# 1. First-edit saliency map
# ---------------------------------------------------------------------------

def first_edit_heatmap(task_id: str) -> np.ndarray:
    H, W = _load_task_shape(task_id)
    heat = np.zeros((H, W), dtype=np.float32)
    n = 0
    for subj, p in _task_traj_paths(task_id):
        tr = _parse_trajectory(p)
        if not tr["edits"]:
            continue
        x, y = tr["edits"][0]["x"], tr["edits"][0]["y"]
        if 0 <= y < H and 0 <= x < W:
            heat[y, x] += 1.0
            n += 1
    if n > 0:
        heat /= n
    return heat


# ---------------------------------------------------------------------------
# 2. Order consistency across subjects
# ---------------------------------------------------------------------------

def order_consistency(task_id: str,
                      min_overlap_cells: int = 4,
                      max_pairs: int = 200) -> float:
    """Pairwise Spearman of subjects' cell-visit rank maps. We need a
    cell that *both* subjects edited to contribute a rank, so a random
    subset of subject pairs is used to keep the cost bounded."""
    subjects = []
    for subj, p in _task_traj_paths(task_id):
        tr = _parse_trajectory(p)
        if not tr["edits"]:
            continue
        # First-visit rank per cell
        first_rank: Dict[Tuple[int, int], int] = {}
        for rank, e in enumerate(tr["edits"]):
            k = (e["x"], e["y"])
            if k not in first_rank:
                first_rank[k] = rank
        subjects.append(first_rank)

    if len(subjects) < 2:
        return float("nan")

    rng = np.random.default_rng(0)
    n = len(subjects)
    idx_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(idx_pairs) > max_pairs:
        pick = rng.choice(len(idx_pairs), max_pairs, replace=False)
        idx_pairs = [idx_pairs[k] for k in pick]

    rhos = []
    for i, j in idx_pairs:
        a, b = subjects[i], subjects[j]
        common = set(a) & set(b)
        if len(common) < min_overlap_cells:
            continue
        va = np.array([a[k] for k in common])
        vb = np.array([b[k] for k in common])
        rho, _ = spearmanr(va, vb)
        if not np.isnan(rho):
            rhos.append(rho)
    return float(np.mean(rhos)) if rhos else float("nan")


# ---------------------------------------------------------------------------
# 3. Color-priority sequence
# ---------------------------------------------------------------------------

def color_priority(task_id: str) -> Dict:
    """For each color used by a subject, record the 1-based rank at which
    that color was first placed. Average across subjects. Lower mean rank
    = earlier priority."""
    per_color_ranks: Dict[int, List[int]] = defaultdict(list)
    n_subjects = 0
    for subj, p in _task_traj_paths(task_id):
        tr = _parse_trajectory(p)
        if not tr["edits"]:
            continue
        n_subjects += 1
        seen: Dict[int, int] = {}
        for rank, e in enumerate(tr["edits"], start=1):
            c = int(e["color"])
            if c not in seen:
                seen[c] = rank
        for c, r in seen.items():
            per_color_ranks[c].append(r)

    rows = []
    for c, ranks in per_color_ranks.items():
        rows.append({
            "color": c,
            "mean_first_rank": float(np.mean(ranks)),
            "n_subjects_used": len(ranks),
            "frac_subjects_used": len(ranks) / max(n_subjects, 1),
        })
    rows.sort(key=lambda r: (r["mean_first_rank"], -r["frac_subjects_used"]))
    return {"task_id": task_id, "n_subjects": n_subjects, "colors": rows}


# ---------------------------------------------------------------------------
# 4. Attention transition graph
# ---------------------------------------------------------------------------

def attention_graph(task_id: str) -> Dict:
    """Build the cell->cell transition count across all subjects.

    Returns:
        {"task_id", "shape": (H, W),
         "edges": np.array([[x_from, y_from, x_to, y_to, count], ...]),
         "node_visits": (H, W) count of total edit visits per cell}
    """
    H, W = _load_task_shape(task_id)
    trans: Dict[Tuple[int, int, int, int], int] = defaultdict(int)
    visits = np.zeros((H, W), dtype=np.int64)
    for subj, p in _task_traj_paths(task_id):
        tr = _parse_trajectory(p)
        edits = tr["edits"]
        for i, e in enumerate(edits):
            x, y = e["x"], e["y"]
            if 0 <= y < H and 0 <= x < W:
                visits[y, x] += 1
            if i == 0:
                continue
            px, py = edits[i - 1]["x"], edits[i - 1]["y"]
            if not (0 <= py < H and 0 <= px < W and 0 <= y < H and 0 <= x < W):
                continue
            if (px, py) == (x, y):
                continue
            trans[(px, py, x, y)] += 1

    if trans:
        edges = np.array(
            [(k[0], k[1], k[2], k[3], v) for k, v in trans.items()],
            dtype=np.int64,
        )
    else:
        edges = np.zeros((0, 5), dtype=np.int64)
    return {"task_id": task_id, "shape": (H, W),
            "edges": edges, "node_visits": visits}


# ---------------------------------------------------------------------------
# Batch computation for all tasks + persistence
# ---------------------------------------------------------------------------

def compute_all(out_dir: str = "prior_analysis") -> pd.DataFrame:
    from human_targets import available_task_ids
    os.makedirs(out_dir, exist_ok=True)
    ids = available_task_ids()

    rows = []
    heatmaps: Dict[str, np.ndarray] = {}
    graphs_edges: Dict[str, np.ndarray] = {}
    graphs_visits: Dict[str, np.ndarray] = {}

    for i, tid in enumerate(ids, 1):
        heat = first_edit_heatmap(tid)
        heatmaps[tid] = heat
        cons = order_consistency(tid)
        cp = color_priority(tid)
        ag = attention_graph(tid)
        graphs_edges[tid] = ag["edges"]
        graphs_visits[tid] = ag["node_visits"]

        top_colors = "|".join(f"{c['color']}({c['mean_first_rank']:.1f})"
                              for c in cp["colors"][:4])
        rows.append({
            "task_id": tid,
            "n_subjects": cp["n_subjects"],
            "order_consistency": cons,
            "n_unique_first_edit_cells": int((heat > 0).sum()),
            "max_first_edit_prob": float(heat.max()),
            "entropy_first_edit": float(
                -(heat[heat > 0] * np.log2(heat[heat > 0])).sum()
            ) if (heat > 0).any() else 0.0,
            "n_transition_edges": int(ag["edges"].shape[0]),
            "top_colors_by_priority": top_colors,
        })
        if i % 10 == 0:
            print(f"  computed {i}/{len(ids)} tasks")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "sequence_features.csv"), index=False)
    np.savez_compressed(os.path.join(out_dir, "first_edit_heatmaps.npz"),
                        **heatmaps)
    np.savez_compressed(os.path.join(out_dir, "attention_graphs_edges.npz"),
                        **graphs_edges)
    np.savez_compressed(os.path.join(out_dir, "attention_graphs_visits.npz"),
                        **graphs_visits)
    print(f"wrote prior_analysis/sequence_features.csv  ({len(df)} tasks)")
    print(f"      prior_analysis/first_edit_heatmaps.npz")
    print(f"      prior_analysis/attention_graphs_edges.npz")
    print(f"      prior_analysis/attention_graphs_visits.npz")
    return df


if __name__ == "__main__":
    compute_all()

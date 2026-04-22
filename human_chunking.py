"""
Drawing chunks: pause-segmented cognitive units.

For each Experiment 2 trajectory we split the edit sequence at pauses —
inter-edit reaction times longer than max(pause_factor * subject_median_rt,
min_pause_ms). The segments between pauses are chunks. A chunk is the
participant's self-defined cognitive unit: "draw this one thing, then
pause and decide what to do next".

Each chunk has:
    size             — number of edits in the chunk
    n_cells          — distinct cells touched
    n_colors         — distinct colors placed
    dominant_color   — the color used most within the chunk
    color_homogeneity — fraction of edits using the dominant color
    bbox             — bounding box of the edits in the chunk
    bbox_area        — bbox height × width
    is_connected     — whether edited cells form one 4-connected component
    cc_count         — number of 4-connected components of edited cells

Per task we also ask: do chunks align with Success-grid components? For
each chunk we find the Success component with maximum pixel IoU and
record both (chunk_iou_best, best_component_id).

Outputs:
    prior_analysis/chunks_per_trajectory.csv      (one row per chunk)
    prior_analysis/chunk_task_summary.csv         (per-task aggregates)
"""

from __future__ import annotations


import _paths  # noqa: F401
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import label as cc_label

from human_style_features import (
    DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR, _parse_trajectory,
)
from human_targets import human_targets


_FNAME_RE = re.compile(r"^subj_([^_]+)_trial_([0-9a-f]+)\.json\.csv$")


# ---------------------------------------------------------------------------
# Pause-based segmentation
# ---------------------------------------------------------------------------

def identify_chunks(edits: List[Dict],
                    pause_factor: float = 2.0,
                    min_pause_ms: float = 500.0) -> List[List[Dict]]:
    """Return a list of chunks (each a list of edits in order).

    Pause threshold per trajectory = max(pause_factor * median_rt, min_pause_ms).
    An edit is separated from the previous one when its rt exceeds the
    threshold.
    """
    if not edits:
        return []
    rts = np.array([e["rt"] for e in edits if not np.isnan(e.get("rt", float("nan")))])
    median_rt = float(np.median(rts)) if rts.size else 0.0
    thresh = max(pause_factor * median_rt, min_pause_ms)

    chunks: List[List[Dict]] = []
    current: List[Dict] = [edits[0]]
    for i in range(1, len(edits)):
        rt = edits[i].get("rt", float("nan"))
        if not np.isnan(rt) and rt > thresh:
            chunks.append(current)
            current = [edits[i]]
        else:
            current.append(edits[i])
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Per-chunk features
# ---------------------------------------------------------------------------

def chunk_features(chunk: List[Dict]) -> Dict:
    xs = np.array([e["x"] for e in chunk])
    ys = np.array([e["y"] for e in chunk])
    colors = [int(e["color"]) for e in chunk]
    ctr = Counter(colors)
    dom, dom_n = ctr.most_common(1)[0]

    # 4-connectivity of distinct cells touched
    cells = list({(e["y"], e["x"]) for e in chunk})
    if cells:
        ymin, xmin = min(c[0] for c in cells), min(c[1] for c in cells)
        H = max(c[0] for c in cells) - ymin + 1
        W = max(c[1] for c in cells) - xmin + 1
        mask = np.zeros((H, W), dtype=bool)
        for (y, x) in cells:
            mask[y - ymin, x - xmin] = True
        _, n_cc = cc_label(mask,
                           structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    else:
        n_cc = 0

    return {
        "size": len(chunk),
        "n_cells": len(cells),
        "n_colors": len(ctr),
        "dominant_color": int(dom),
        "color_homogeneity": dom_n / len(chunk),
        "bbox_ymin": int(ys.min()), "bbox_ymax": int(ys.max()),
        "bbox_xmin": int(xs.min()), "bbox_xmax": int(xs.max()),
        "bbox_area": int((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)),
        "is_connected": bool(n_cc == 1),
        "cc_count": int(n_cc),
        "t_start": float(chunk[0].get("time", 0.0)),
        "t_end": float(chunk[-1].get("time", 0.0)),
    }


# ---------------------------------------------------------------------------
# Success-component IoU (do chunks align with Success components?)
# ---------------------------------------------------------------------------

def _success_components(task_id: str) -> Tuple[np.ndarray, List[set], Dict[int, set]]:
    """Returns (success_grid, list_of_cc_masks, color_to_cellset).

    The color_to_cellset dict groups all Success-grid cells by color (a
    single key per color, regardless of 4-connectivity). This is the
    target set for the ``chunk vs color class`` comparison.
    """
    t = human_targets(task_id)
    success = None
    for lbl, g in zip(t["labels"], t["grids"]):
        if lbl == "Success":
            success = g; break
    if success is None:
        return np.array([]), [], {}
    bg_vals, bg_counts = np.unique(success, return_counts=True)
    bg = int(bg_vals[np.argmax(bg_counts)])
    comp_masks: List[set] = []
    color_sets: Dict[int, set] = {}
    for color in np.unique(success):
        if int(color) == bg:
            continue
        mask = (success == color)
        color_sets[int(color)] = set(
            map(tuple, np.argwhere(mask).tolist())
        )
        labels, n = cc_label(mask, structure=np.array(
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
        for cid in range(1, n + 1):
            coords = np.argwhere(labels == cid)
            comp_masks.append(set(map(tuple, coords.tolist())))
    return success, comp_masks, color_sets


def best_iou_with_success(chunk: List[Dict], comp_masks: List[set]) -> Tuple[float, int]:
    cells = {(e["y"], e["x"]) for e in chunk}
    if not cells or not comp_masks:
        return 0.0, -1
    best, best_id = 0.0, -1
    for i, comp in enumerate(comp_masks):
        inter = len(cells & comp)
        if inter == 0:
            continue
        union = len(cells | comp)
        iou = inter / union
        if iou > best:
            best, best_id = iou, i
    return float(best), int(best_id)


def color_class_iou(chunk: List[Dict],
                    color_sets: Dict[int, set]) -> Tuple[float, int]:
    """IoU of the chunk's cells against the Success color-class that gives
    the best match (one target per color, regardless of 4-connectivity)."""
    cells = {(e["y"], e["x"]) for e in chunk}
    if not cells or not color_sets:
        return 0.0, -1
    best, best_color = 0.0, -1
    for color, target in color_sets.items():
        inter = len(cells & target)
        if inter == 0:
            continue
        iou = inter / len(cells | target)
        if iou > best:
            best, best_color = iou, color
    return float(best), int(best_color)


def n_success_cc_spanned(chunk: List[Dict], comp_masks: List[set]) -> int:
    cells = {(e["y"], e["x"]) for e in chunk}
    return sum(1 for c in comp_masks if cells & c)


# ---------------------------------------------------------------------------
# Batch computation
# ---------------------------------------------------------------------------

def chunks_for_task(task_id: str) -> List[Dict]:
    _, comp_masks, color_sets = _success_components(task_id)
    traj_dir = os.path.join(DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR,
                            f"{task_id}.json")
    out: List[Dict] = []
    if not os.path.isdir(traj_dir):
        return out
    for fname in os.listdir(traj_dir):
        m = _FNAME_RE.match(fname)
        if not m:
            continue
        subj = m.group(1)
        tr = _parse_trajectory(os.path.join(traj_dir, fname))
        if not tr["edits"]:
            continue
        chunks = identify_chunks(tr["edits"])
        for k, ch in enumerate(chunks):
            feats = chunk_features(ch)
            iou_cc, best_cc = best_iou_with_success(ch, comp_masks)
            iou_color, best_color = color_class_iou(ch, color_sets)
            n_cc = n_success_cc_spanned(ch, comp_masks)
            row = {
                "task_id": task_id,
                "subject_id": subj,
                "chunk_index": k,
                "n_chunks_total": len(chunks),
                **feats,
                "success_iou_best": iou_cc,
                "success_component_matched": best_cc,
                "n_success_components": len(comp_masks),
                "success_iou_color_class": iou_color,
                "success_color_class_matched": best_color,
                "n_success_cc_spanned": n_cc,
            }
            out.append(row)
    return out


def main():
    import argparse
    from human_targets import available_task_ids
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task_ids", nargs="+", default=None)
    ap.add_argument("--out_dir", default="prior_analysis")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ids = args.task_ids or available_task_ids()

    all_rows = []
    for i, tid in enumerate(ids, 1):
        all_rows.extend(chunks_for_task(tid))
        if i % 10 == 0:
            print(f"  processed {i}/{len(ids)} tasks")
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(args.out_dir, "chunks_per_trajectory.csv"), index=False)
    print(f"wrote chunks_per_trajectory.csv  ({len(df)} chunks across "
          f"{df['task_id'].nunique()} tasks, "
          f"{df['subject_id'].nunique()} subjects)")

    # Per-task summary
    task_sum = df.groupby("task_id").agg(
        n_subjects=("subject_id", "nunique"),
        mean_chunks_per_subj=("chunk_index",
                              lambda s: (s + 1).groupby(df.loc[s.index, "subject_id"]).max().mean()),
        mean_chunk_size=("size", "mean"),
        median_chunk_size=("size", "median"),
        mean_color_homogeneity=("color_homogeneity", "mean"),
        frac_connected_chunks=("is_connected", "mean"),
        mean_success_iou_best=("success_iou_best", "mean"),
        n_success_components=("n_success_components", "first"),
    ).round(3).reset_index()
    task_sum.to_csv(os.path.join(args.out_dir, "chunk_task_summary.csv"),
                    index=False)
    print(f"wrote chunk_task_summary.csv  ({len(task_sum)} tasks)")
    print("\n--- chunking summary (first 10 tasks) ---")
    print(task_sum.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

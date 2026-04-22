"""
Classify edit-level errors as MOTOR SLIP vs COGNITIVE ERROR.

Per-edit signals we combine:
  - spatial_dist_to_correct: Manhattan distance from the edited cell to the
    nearest cell whose Success-grid color equals the edited color. Small =
    "I picked the right color but clicked one cell off" = motor.
  - quickly_corrected: did a later edit on the same cell restore the Success
    color within either a short time window (<= 2000 ms) or a short edit
    window (<= 3 subsequent edits)? Motor slips are immediately noticed.
  - rt_z: this edit's RT standardized against the subject's median RT on
    this trial. Large positive z = the participant paused = more likely a
    cognitive decision (deliberate placement of a wrong cell).
  - matches_top_error: at this (x, y), does the edited color equal what a
    Top Error grid would have here? Systematic matches => cognitive.
  - in_wrong_run: is this edit inside a contiguous same-color streak of
    wrong edits? Motor slips are isolated; cognitive errors tend to come
    as coherent wrong shapes.

Decision:
  flag motor if  spatial_dist_to_correct <= 1  AND  (quickly_corrected OR
                                                      NOT in_wrong_run)
  flag cognitive if  matches_top_error OR in_wrong_run OR
                    (spatial_dist_to_correct >= 2 AND NOT quickly_corrected)
  remainder: "ambiguous"

Per trajectory we return counts + rates; per task we aggregate across
subjects so you can ask "is this task dominated by cognitive failures?"
"""

from __future__ import annotations

import _paths  # noqa: F401  (sys.path bootstrap, safe under both invocations)

import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from human_style_features import (
    DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR, _parse_trajectory,
)
from human_targets import human_targets


BURST_CORRECTION_MS = 2000.0
BURST_CORRECTION_EDITS = 3
SPATIAL_MOTOR_MAX = 1  # Manhattan distance treated as a motor miss
RT_Z_COGNITIVE = 1.5   # RT z-score above this is "paused before" = deliberate


_FNAME_RE = re.compile(r"^subj_([^_]+)_trial_([0-9a-f]+)\.json\.csv$")


def _nearest_correct_distance(cx: int, cy: int, color: int,
                              correct_mask: np.ndarray) -> int:
    """
    Manhattan distance from (cx, cy) to the nearest cell whose Success
    grid color equals `color`. correct_mask[y, x] = bool (True where the
    Success grid has `color`).
    """
    if not correct_mask.any():
        return 999
    ys, xs = np.where(correct_mask)
    return int(np.min(np.abs(xs - cx) + np.abs(ys - cy)))


def classify_trajectory(task_id: str, trajectory_path: str) -> Dict:
    """Classify every edit in one trajectory, return per-trajectory stats."""
    tr = _parse_trajectory(trajectory_path)
    edits = tr["edits"]
    targets = human_targets(task_id)

    success = None
    top_errors = []
    for lbl, g in zip(targets["labels"], targets["grids"]):
        if lbl == "Success":
            success = g
        else:
            top_errors.append((lbl, g))
    if success is None:
        return {"task_id": task_id, "n_edits": len(edits),
                "n_motor": 0, "n_cognitive": 0, "n_ambiguous": 0,
                "n_correct": 0}

    Hs, Ws = success.shape
    # Precompute color masks for Success grid (per color -> bool mask)
    color_masks = {int(c): (success == c) for c in np.unique(success)}

    # Per-subject RT baseline
    rts = np.array([e["rt"] for e in edits if not np.isnan(e["rt"])])
    med_rt = float(np.median(rts)) if rts.size else 0.0
    mad_rt = float(np.median(np.abs(rts - med_rt))) if rts.size else 0.0
    # approximate std from MAD
    std_rt = 1.4826 * mad_rt if mad_rt > 0 else 1.0

    # Final edit-per-cell map (to detect "in_wrong_run": what did the
    # participant leave on the grid after the last edit at each cell?)
    # We still classify on a per-edit basis, but reuse a computed "final
    # cell state" to detect sustained wrong regions.
    final_state: Dict[Tuple[int, int], int] = {}
    for e in edits:
        final_state[(e["x"], e["y"])] = e["color"]

    # Build wrong-cell mask in final state to detect runs
    wrong_final = np.zeros_like(success, dtype=bool)
    for (x, y), c in final_state.items():
        if 0 <= y < Hs and 0 <= x < Ws and success[y, x] != c:
            wrong_final[y, x] = True

    # Connected components of wrong final cells (4-connectivity)
    from scipy.ndimage import label as cc_label
    comp_labels, n_comps = cc_label(wrong_final, structure=np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]]))

    # For per-edit lookup: was the *final* state of (x,y) part of a large
    # wrong component (size >= 3)?
    large_wrong_comp_size = np.zeros_like(success, dtype=int)
    for cid in range(1, n_comps + 1):
        size = int((comp_labels == cid).sum())
        large_wrong_comp_size[comp_labels == cid] = size

    n_motor = n_cognitive = n_ambig = n_correct = 0
    motor_rt = []
    cognitive_rt = []

    for i, e in enumerate(edits):
        x, y, c, rt = e["x"], e["y"], e["color"], e["rt"]
        if not (0 <= y < Hs and 0 <= x < Ws):
            n_ambig += 1
            continue
        target_color = int(success[y, x])
        if c == target_color:
            n_correct += 1
            continue

        # -- Wrong edit. Gather signals.
        dist = _nearest_correct_distance(
            x, y, c, color_masks.get(c, np.zeros_like(success, dtype=bool)))
        # quickly_corrected: does a later edit restore this cell within
        # BURST_CORRECTION_MS or BURST_CORRECTION_EDITS?
        quickly_corrected = False
        for j in range(i + 1, min(i + 1 + BURST_CORRECTION_EDITS * 2, len(edits))):
            nxt = edits[j]
            if nxt["x"] == x and nxt["y"] == y:
                dt = nxt["time"] - e["time"]
                within_edits = (j - i) <= BURST_CORRECTION_EDITS
                within_ms = dt <= BURST_CORRECTION_MS
                if (within_edits or within_ms) and nxt["color"] == target_color:
                    quickly_corrected = True
                break
        # rt z-score
        rt_z = 0.0
        if not np.isnan(rt):
            rt_z = (rt - med_rt) / std_rt
        # matches a Top Error?
        matches_te = False
        for _, te_grid in top_errors:
            if te_grid.shape == success.shape and int(te_grid[y, x]) == c:
                matches_te = True
                break
        # In a sustained wrong run at the *final* state?
        in_wrong_run = int(large_wrong_comp_size[y, x]) >= 3

        # Decide
        if dist <= SPATIAL_MOTOR_MAX and (quickly_corrected or (not in_wrong_run
                                                                and not matches_te
                                                                and rt_z < RT_Z_COGNITIVE)):
            n_motor += 1
            if not np.isnan(rt):
                motor_rt.append(rt)
        elif matches_te or in_wrong_run or (dist >= 2 and not quickly_corrected) or rt_z >= RT_Z_COGNITIVE:
            n_cognitive += 1
            if not np.isnan(rt):
                cognitive_rt.append(rt)
        else:
            n_ambig += 1

    total_wrong = n_motor + n_cognitive + n_ambig
    return {
        "task_id": task_id,
        "n_edits": len(edits),
        "n_correct": n_correct,
        "n_wrong": total_wrong,
        "n_motor": n_motor,
        "n_cognitive": n_cognitive,
        "n_ambiguous": n_ambig,
        "motor_rate": n_motor / max(total_wrong, 1),
        "cognitive_rate": n_cognitive / max(total_wrong, 1),
        "mean_rt_motor": float(np.mean(motor_rt)) if motor_rt else float("nan"),
        "mean_rt_cognitive": float(np.mean(cognitive_rt)) if cognitive_rt else float("nan"),
        "final_wrong_components": int(n_comps),
    }


def classify_all(data_root: str = DEFAULT_DATA_ROOT,
                 task_ids: List[str] = None) -> pd.DataFrame:
    root = os.path.join(data_root, _EXP2_EDIT_DIR)
    task_dirs = sorted(d for d in os.listdir(root) if d.endswith(".json"))
    if task_ids:
        wanted = {f"{tid}.json" for tid in task_ids}
        task_dirs = [d for d in task_dirs if d in wanted]

    rows = []
    for td in task_dirs:
        tid = td[:-5]
        task_dir = os.path.join(root, td)
        for fname in os.listdir(task_dir):
            m = _FNAME_RE.match(fname)
            if not m:
                continue
            subj_id = m.group(1)
            try:
                r = classify_trajectory(tid, os.path.join(task_dir, fname))
            except Exception as e:
                continue
            r["subject_id"] = subj_id
            rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_ids", nargs="+", default=None)
    ap.add_argument("--out_csv", default="prior_analysis/error_types.csv")
    args = ap.parse_args()

    df = classify_all(task_ids=args.task_ids)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}  ({len(df)} trajectories)")

    # Per-task summary
    per_task = df.groupby("task_id").agg(
        n_subjects=("subject_id", "nunique"),
        mean_motor_rate=("motor_rate", "mean"),
        mean_cognitive_rate=("cognitive_rate", "mean"),
        mean_n_wrong=("n_wrong", "mean"),
        mean_rt_motor=("mean_rt_motor", "mean"),
        mean_rt_cognitive=("mean_rt_cognitive", "mean"),
    ).round(3).sort_values("mean_cognitive_rate", ascending=False)
    print("\nTop 15 tasks by cognitive-error rate:")
    print(per_task.head(15).to_string())
    print("\nTop 15 tasks by motor-slip rate:")
    print(per_task.sort_values("mean_motor_rate", ascending=False).head(15).to_string())
    per_task.to_csv("prior_analysis/error_types_per_task.csv")
    print("\nAlso wrote prior_analysis/error_types_per_task.csv")

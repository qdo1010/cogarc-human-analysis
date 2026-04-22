"""
Do chunks on cognitive-error trajectories align with the TOP-ERROR grid
instead of Success?

Hypothesis:
  - If a trajectory is dominated by cognitive errors, the participant
    was executing a coherent-but-wrong plan. Their chunks should match
    one of the known Top-Error grids at IoU comparable to how correct
    chunks match Success.
  - If a trajectory is dominated by motor slips, errors are
    distributed randomly around Success cells; chunks should still
    align with Success, and Top-Error IoU should be low.

For each (subject, task) trajectory we:
  1. Segment into chunks (pause threshold, same as human_chunking.py).
  2. For each chunk compute:
       iou_success        = best IoU vs any Success 4-connected component
       iou_top_error      = best IoU vs any Top-Error component across
                            all Top-Error grids for that task
       best_top_error_lbl = which Top Error provided the best match
  3. Aggregate per trajectory: mean IoU vs Success, mean IoU vs Top
     Error, delta = iou_top_error - iou_success. Positive delta means
     the chunks line up better with an error grid than with Success.
  4. Merge with prior_analysis/error_types.csv (per-trajectory motor_rate
     and cognitive_rate) and split by bucket.

Outputs:
    prior_analysis/chunk_vs_error_per_trajectory.csv
    prior_analysis/chunk_vs_error_bucket_summary.csv
"""

from __future__ import annotations


import _paths  # noqa: F401

import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import label as cc_label

from human_chunking import (
    best_iou_with_success, identify_chunks,
)
from human_style_features import (
    DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR, _parse_trajectory,
)
from human_targets import (
    human_targets, available_task_ids, _EXP2_SUB_DIR, _grid_key, _load_grid,
)


_FNAME_RE = re.compile(r"^subj_([^_]+)_trial_([0-9a-f]+)\.json\.csv$")
_CC_STRUCT = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])


def _components(grid: np.ndarray) -> List[set]:
    """4-connected components of all non-background cells."""
    if grid.size == 0:
        return []
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    out: List[set] = []
    for color in np.unique(grid):
        if int(color) == bg:
            continue
        mask = (grid == color)
        labels, n = cc_label(mask, structure=_CC_STRUCT)
        for cid in range(1, n + 1):
            coords = np.argwhere(labels == cid)
            out.append(set(map(tuple, coords.tolist())))
    return out


def _diagnostic_cells(success: np.ndarray,
                      other: np.ndarray) -> set:
    """Cells where `other` disagrees with `success`. Restricts to cells
    whose value in `other` is non-background, so the chunk-matcher
    only rewards cells the participant actively painted differently.
    """
    if success.shape != other.shape:
        return set()
    bg_o = int(np.bincount(other.flatten()).argmax())
    diff = (success != other) & (other != bg_o)
    return set(map(tuple, np.argwhere(diff).tolist()))


def _targets_by_label(task_id: str) -> Dict[str, List[set]]:
    """Return {label -> list_of_4CC_masks} for Success + all Top Error
    grids available for this task."""
    t = human_targets(task_id)
    return {lbl: _components(g) for lbl, g in zip(t["labels"], t["grids"])}


def _best_iou_over(cells: set, target_comps: List[set]) -> float:
    best = 0.0
    for comp in target_comps:
        inter = len(cells & comp)
        if inter == 0:
            continue
        iou = inter / len(cells | comp)
        if iou > best:
            best = iou
    return float(best)


def _submission_labels_for_task(task_id: str) -> Dict[str, str]:
    """Return {subject_id -> label} for this task based on the raw
    submission1.json files. Label is 'Success', 'Top Error N', or 'Other'.
    """
    t = human_targets(task_id)
    key_to_label = {_grid_key(g): lbl
                    for g, lbl in zip(t["grids"], t["labels"])}
    out: Dict[str, str] = {}
    sub_dir = os.path.join(DEFAULT_DATA_ROOT, _EXP2_SUB_DIR,
                           f"{task_id}.json")
    if not os.path.isdir(sub_dir):
        return out
    suf = "_submission1.json"
    for fname in os.listdir(sub_dir):
        if not fname.endswith(suf):
            continue
        # expected: {task_id}_{subject_id}_submission1.json
        middle = fname[:-len(suf)]
        if not middle.startswith(task_id + "_"):
            continue
        subj = middle[len(task_id) + 1:]
        try:
            g = _load_grid(os.path.join(sub_dir, fname))
        except Exception:
            continue
        out[subj] = key_to_label.get(_grid_key(g), "Other")
    return out


def chunks_vs_errors_for_task(task_id: str) -> List[Dict]:
    tgts = _targets_by_label(task_id)
    labels = list(tgts.keys())              # ['Success', 'Top Error 1', ...]
    error_labels = [l for l in labels if l.startswith("Top Error")]
    subj_to_final_label = _submission_labels_for_task(task_id)

    # Diagnostic cell-sets per label: for each (y,x), identify the set
    # (plus the total diagnostic pool size, logged per row for later
    # per-task filtering)
    _placeholder = None  # noqa
    # of labels that have the SAME (color) at that cell. A label is the
    # unique identifier at (y,x) iff its color there differs from every
    # other label's color there. Two versions:
    #   diag_cells[label]         = set of (y,x) where label is unique
    #   diag_colored[label]       = set of (y,x,color) triples — cell +
    #                               the label's color at that cell. A
    #                               chunk matches a triple only if the
    #                               participant painted that exact color
    #                               at that cell. This is the color-
    #                               aware version of the vote.
    t = human_targets(task_id)
    label_grids = {lbl: g for lbl, g in zip(t["labels"], t["grids"])}
    diag_cells: Dict[str, set] = {lbl: set() for lbl in label_grids}
    diag_colored: Dict[str, set] = {lbl: set() for lbl in label_grids}
    if label_grids:
        # All grids share the same shape.
        shapes = {g.shape for g in label_grids.values()}
        if len(shapes) == 1:
            H, W = next(iter(shapes))
            for y in range(H):
                for x in range(W):
                    colors_here = {lbl: int(g[y, x])
                                   for lbl, g in label_grids.items()}
                    for lbl, c in colors_here.items():
                        if all(colors_here[o] != c
                               for o in colors_here if o != lbl):
                            diag_cells[lbl].add((y, x))
                            diag_colored[lbl].add((y, x, c))

    traj_dir = os.path.join(DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR,
                            f"{task_id}.json")
    rows: List[Dict] = []
    if not os.path.isdir(traj_dir):
        return rows

    for fname in os.listdir(traj_dir):
        m = _FNAME_RE.match(fname)
        if not m:
            continue
        subj = m.group(1)
        tr = _parse_trajectory(os.path.join(traj_dir, fname))
        if not tr["edits"]:
            continue
        chunks = identify_chunks(tr["edits"])
        if not chunks:
            continue
        chunk_index_counter = 0

        # Per-chunk IoU against every available label (Success + each
        # Top Error). Record which label gave the best IoU.
        s_ious, e_ious = [], []
        best_label_counts: Dict[str, int] = {l: 0 for l in labels}
        # Diagnostic vote: which label does this chunk's cells fall into
        # as a fraction of the label's diagnostic mask?
        diag_vote_counts: Dict[str, int] = {l: 0 for l in labels}
        # Color-aware diagnostic vote: chunk's (y,x,color) triples (the
        # color the participant LAST left at each cell within the chunk)
        # matched against each label's diagnostic_colored triples.
        diag_col_vote_counts: Dict[str, int] = {l: 0 for l in labels}
        # WEIGHTED color-aware vote: sum over chunks of
        # (# diagnostic triples matched). Higher weight == chunk hit
        # more diagnostic evidence.
        diag_col_weighted: Dict[str, int] = {l: 0 for l in labels}
        # Late-session sub-tally: only chunks in the LAST HALF of the
        # trajectory (by chunk index). The plan crystallises late, so
        # late chunks should carry more diagnostic signal.
        diag_col_weighted_late: Dict[str, int] = {l: 0 for l in labels}
        n_late_start = len(chunks) // 2
        # Early-session sub-tally: only chunks in the FIRST HALF. Tests
        # whether the wrong plan is present from the start.
        diag_col_weighted_early: Dict[str, int] = {l: 0 for l in labels}
        n_early_end = len(chunks) // 2
        diag_col_weighted_firstq: Dict[str, int] = {l: 0 for l in labels}
        n_firstq_end = max(1, len(chunks) // 4)
        # Per-chunk strength distribution: how many chunks hit k>=K
        # diagnostic triples for any label?
        chunk_strengths: List[int] = []
        for ch in chunks:
            cells = {(e["y"], e["x"]) for e in ch}
            iou_s = _best_iou_over(cells, tgts.get("Success", []))
            best_e, best_e_lbl = 0.0, ""
            for lbl in error_labels:
                iou_e = _best_iou_over(cells, tgts[lbl])
                if iou_e > best_e:
                    best_e = iou_e
                    best_e_lbl = lbl
            s_ious.append(iou_s); e_ious.append(best_e)

            # Overall best label (Success OR Top Error).
            if iou_s == 0 and best_e == 0:
                pass                         # no IoU match
            elif iou_s >= best_e:
                best_label_counts["Success"] += 1
            else:
                best_label_counts[best_e_lbl] += 1

            # Diagnostic vote: cells that hit a label's DIAGNOSTIC
            # mask (i.e. cells that distinguish that label from
            # Success) weighted by fraction of chunk overlapping it.
            if diag_cells:
                best_diag, best_diag_lbl = 0.0, ""
                for lbl, dcells in diag_cells.items():
                    if not dcells:
                        continue
                    inter = len(cells & dcells)
                    if inter == 0:
                        continue
                    # Fraction of chunk inside the diagnostic mask.
                    frac = inter / len(cells)
                    if frac > best_diag:
                        best_diag = frac
                        best_diag_lbl = lbl
                if best_diag_lbl:
                    diag_vote_counts[best_diag_lbl] += 1

            # Color-aware diagnostic vote: score a label by how many
            # (y, x, color) triples from the chunk match its diagnostic
            # triples. Only the LAST color the chunk placed at each
            # cell counts, matching the visible end-state of the chunk.
            if diag_colored:
                last_color: Dict[Tuple[int, int], int] = {}
                for e in ch:
                    last_color[(e["y"], e["x"])] = int(e["color"])
                chunk_triples = {(y, x, c)
                                 for (y, x), c in last_color.items()}
                per_label_n = {}
                for lbl, tri in diag_colored.items():
                    if not tri:
                        continue
                    n = len(chunk_triples & tri)
                    per_label_n[lbl] = n
                # Unweighted vote: which label got the most triples
                # on this chunk (winner-takes-chunk).
                best_n = max(per_label_n.values()) if per_label_n else 0
                best_n_lbl = max(per_label_n.items(),
                                 key=lambda kv: kv[1])[0] if per_label_n else ""
                if best_n > 0:
                    diag_col_vote_counts[best_n_lbl] += 1
                # Weighted vote across all labels, not just the winner.
                is_late = (chunk_index_counter >= n_late_start)
                is_early = (chunk_index_counter < n_early_end)
                is_firstq = (chunk_index_counter < n_firstq_end)
                for lbl, n in per_label_n.items():
                    diag_col_weighted[lbl] += n
                    if is_late:
                        diag_col_weighted_late[lbl] += n
                    if is_early:
                        diag_col_weighted_early[lbl] += n
                    if is_firstq:
                        diag_col_weighted_firstq[lbl] += n
                chunk_strengths.append(int(best_n))
            chunk_index_counter += 1

        s_arr = np.array(s_ious); e_arr = np.array(e_ious)
        # Dominant target label across chunks, and whether it's an error.
        dom = max(best_label_counts.items(), key=lambda kv: kv[1])
        dom_label, dom_count = dom
        n_matched = sum(best_label_counts.values())
        frac_top_error = sum(v for k, v in best_label_counts.items()
                             if k.startswith("Top Error")) / max(n_matched, 1)

        # Diagnostic-mask vote: which label's DIFFERENCE-from-Success
        # did this trajectory's chunks fill in?
        if sum(diag_vote_counts.values()) > 0:
            diag_dom_label, diag_dom_count = max(
                diag_vote_counts.items(), key=lambda kv: kv[1])
        else:
            diag_dom_label, diag_dom_count = "none", 0
        diag_total = sum(diag_vote_counts.values())

        # Color-aware diagnostic-mask vote (winner-per-chunk).
        if sum(diag_col_vote_counts.values()) > 0:
            diag_col_dom_label, diag_col_dom_count = max(
                diag_col_vote_counts.items(), key=lambda kv: kv[1])
        else:
            diag_col_dom_label, diag_col_dom_count = "none", 0
        diag_col_total = sum(diag_col_vote_counts.values())

        # Weighted color-aware (sum of triple matches across all chunks).
        w_total = sum(diag_col_weighted.values())
        if w_total > 0:
            diag_col_w_label, diag_col_w_score = max(
                diag_col_weighted.items(), key=lambda kv: kv[1])
        else:
            diag_col_w_label, diag_col_w_score = "none", 0

        # Late-half weighted.
        wl_total = sum(diag_col_weighted_late.values())
        if wl_total > 0:
            diag_col_wl_label, diag_col_wl_score = max(
                diag_col_weighted_late.items(), key=lambda kv: kv[1])
        else:
            diag_col_wl_label, diag_col_wl_score = "none", 0

        # Early-half weighted.
        we_total = sum(diag_col_weighted_early.values())
        if we_total > 0:
            diag_col_we_label, diag_col_we_score = max(
                diag_col_weighted_early.items(), key=lambda kv: kv[1])
        else:
            diag_col_we_label, diag_col_we_score = "none", 0
        # First-quarter weighted.
        wq_total = sum(diag_col_weighted_firstq.values())
        if wq_total > 0:
            diag_col_wq_label, diag_col_wq_score = max(
                diag_col_weighted_firstq.items(), key=lambda kv: kv[1])
        else:
            diag_col_wq_label, diag_col_wq_score = "none", 0

        final_label = subj_to_final_label.get(subj, "Unknown")
        # Per-task diagnosability: total number of mutually-exclusive
        # (y,x,color) triples across all error labels. Tasks where two
        # TEs differ by <3 cells have almost no signal to exploit.
        diag_pool_total = sum(len(v) for k, v in diag_colored.items()
                              if k.startswith("Top Error"))
        success_pool = len(diag_colored.get("Success", set()))
        rows.append({
            "task_id": task_id,
            "subject_id": subj,
            "n_chunks": len(chunks),
            "n_chunks_matched": int(n_matched),
            "mean_iou_success": float(s_arr.mean()),
            "mean_iou_top_error": float(e_arr.mean()),
            "max_iou_success": float(s_arr.max()),
            "max_iou_top_error": float(e_arr.max()),
            "delta_top_error_minus_success": float(
                e_arr.mean() - s_arr.mean()),
            "frac_chunks_error_wins": float((e_arr > s_arr).mean()),
            "frac_chunks_best_is_top_error": float(frac_top_error),
            "dominant_target_label": dom_label,
            "dominant_target_count": int(dom_count),
            "n_error_grids_available": int(len(error_labels)),
            "final_submission_label": final_label,
            "chunk_vote_matches_final": int(dom_label == final_label),
            "diag_vote_label": diag_dom_label,
            "diag_vote_count": int(diag_dom_count),
            "diag_vote_total": int(diag_total),
            "diag_vote_matches_final": int(diag_dom_label == final_label),
            "diag_col_vote_label": diag_col_dom_label,
            "diag_col_vote_count": int(diag_col_dom_count),
            "diag_col_vote_total": int(diag_col_total),
            "diag_col_vote_matches_final": int(
                diag_col_dom_label == final_label),
            "diag_col_w_label": diag_col_w_label,
            "diag_col_w_score": int(diag_col_w_score),
            "diag_col_w_total": int(w_total),
            "diag_col_w_matches_final": int(diag_col_w_label == final_label),
            "diag_col_wl_label": diag_col_wl_label,
            "diag_col_wl_score": int(diag_col_wl_score),
            "diag_col_wl_total": int(wl_total),
            "diag_col_wl_matches_final": int(diag_col_wl_label == final_label),
            "diag_col_we_label": diag_col_we_label,
            "diag_col_we_score": int(diag_col_we_score),
            "diag_col_we_total": int(we_total),
            "diag_col_we_matches_final": int(diag_col_we_label == final_label),
            "diag_col_wq_label": diag_col_wq_label,
            "diag_col_wq_score": int(diag_col_wq_score),
            "diag_col_wq_total": int(wq_total),
            "diag_col_wq_matches_final": int(diag_col_wq_label == final_label),
            "max_chunk_strength": int(max(chunk_strengths) if chunk_strengths
                                      else 0),
            "diag_pool_te_total": int(diag_pool_total),
            "diag_pool_success": int(success_pool),
        })
    return rows


def main():
    os.makedirs("prior_analysis", exist_ok=True)
    ids = available_task_ids()
    all_rows: List[Dict] = []
    for i, tid in enumerate(ids, 1):
        all_rows.extend(chunks_vs_errors_for_task(tid))
        if i % 10 == 0:
            print(f"  processed {i}/{len(ids)} tasks")
    df = pd.DataFrame(all_rows)
    df.to_csv("prior_analysis/chunk_vs_error_per_trajectory.csv",
              index=False)
    print(f"wrote chunk_vs_error_per_trajectory.csv  "
          f"({len(df)} trajectories)")

    # Merge with per-trajectory error-type classification.
    err = pd.read_csv("prior_analysis/error_types.csv")
    merged = df.merge(err[["task_id", "subject_id",
                           "n_motor", "n_cognitive", "n_ambiguous",
                           "motor_rate", "cognitive_rate",
                           "n_edits", "n_correct"]],
                      on=["task_id", "subject_id"], how="left")

    # Bucket each trajectory. Cutoffs match motor_vs_cognitive.py style:
    #   motor-dominant:     motor_rate >= 0.20 and cognitive_rate <= 0.20
    #   cognitive-dominant: cognitive_rate >= 0.50 and motor_rate < 0.10
    #   correct-dominant:   n_wrong == 0
    #   mixed: everything else
    def _bucket(row):
        if row["n_motor"] + row["n_cognitive"] + row["n_ambiguous"] == 0:
            return "correct"
        if row["motor_rate"] >= 0.20 and row["cognitive_rate"] <= 0.20:
            return "motor"
        if row["cognitive_rate"] >= 0.50 and row["motor_rate"] < 0.10:
            return "cognitive"
        return "mixed"
    merged["bucket"] = merged.apply(_bucket, axis=1)

    # Save per-trajectory
    merged.to_csv("prior_analysis/chunk_vs_error_per_trajectory.csv",
                  index=False)

    summary = (merged.groupby("bucket")
               .agg(n_trajectories=("subject_id", "count"),
                    mean_iou_success=("mean_iou_success", "mean"),
                    mean_iou_top_error=("mean_iou_top_error", "mean"),
                    mean_delta=("delta_top_error_minus_success", "mean"),
                    mean_frac_error_wins=("frac_chunks_error_wins", "mean"),
                    mean_n_chunks=("n_chunks", "mean"))
               .round(3).reset_index())
    summary.to_csv("prior_analysis/chunk_vs_error_bucket_summary.csv",
                   index=False)
    print("\nchunk alignment by trajectory bucket:")
    print(summary.to_string(index=False))

    # ---------------------------------------------------------------
    # DIAGNOSTIC TEST: can chunks predict which cognitive-error grid
    # the participant ended up submitting?
    # ---------------------------------------------------------------
    diag = merged[merged["final_submission_label"].isin(
        ["Success"] + [f"Top Error {i}" for i in range(1, 10)])].copy()
    if not len(diag):
        return

    print(f"\nDIAGNOSTIC: chunk vote vs actual submission label  "
          f"(n = {len(diag)} trajectories with a known label)")
    overall_acc = diag["chunk_vote_matches_final"].mean()
    chance = 1.0 / (1 + diag["n_error_grids_available"].mean())
    print(f"  overall chunk-vote accuracy: {overall_acc:.3f}   "
          f"(random-baseline ~ {chance:.3f})")

    # Per actual-final-label accuracy
    print("\nPer-final-label chunk-vote accuracy "
          "(does the chunk distribution agree with what they actually drew?):")
    pf = (diag.groupby("final_submission_label")
          .agg(n=("subject_id", "count"),
               chunk_vote_accuracy=("chunk_vote_matches_final", "mean"),
               mean_chunks=("n_chunks", "mean"))
          .round(3).reset_index()
          .sort_values("n", ascending=False))
    print(pf.to_string(index=False))

    # Confusion matrix: rows = actual final label, cols = chunk-vote.
    # (only labels with >=20 trajectories to keep it readable)
    keep = pf[pf["n"] >= 20]["final_submission_label"].tolist()
    if keep:
        cm = pd.crosstab(
            diag[diag["final_submission_label"].isin(keep)]
                ["final_submission_label"],
            diag[diag["final_submission_label"].isin(keep)]
                ["dominant_target_label"],
            normalize="index",
        ).round(3)
        print("\nRow-normalised confusion: actual final label (rows) vs "
              "chunk-vote label (cols):")
        print(cm.to_string())
        cm.to_csv("prior_analysis/chunk_vote_confusion.csv")

    # Restrict to a strict cognitive bucket for the cleanest test:
    # participant made many cognitive errors AND ended up on a known
    # Top-Error label (not Success, not Other).
    strict = diag[(diag["n_cognitive"] >= 10)
                  & (diag["cognitive_rate"] >= 0.7)
                  & (diag["final_submission_label"].str.startswith("Top Error"))]
    if len(strict):
        acc = strict["chunk_vote_matches_final"].mean()
        acc_diag = strict["diag_vote_matches_final"].mean()
        acc_diag_col = strict["diag_col_vote_matches_final"].mean()
        print(f"\nStrict cognitive set (n_cog>=10, cog_rate>=0.7, final is "
              f"Top Error): n={len(strict)}")
        print(f"  chunk-IoU-vote accuracy:            {acc:.3f}")
        print(f"  DIAGNOSTIC-cells-vote accuracy:     {acc_diag:.3f}  "
              "(cell match only)")
        print(f"  COLOR-AWARE diag-vote accuracy:     {acc_diag_col:.3f}  "
              "((y,x,color) triple match)")
        # Per Top-Error label
        per = (strict.groupby("final_submission_label")
               .agg(n=("subject_id", "count"),
                    chunk_vote_accuracy=("chunk_vote_matches_final", "mean"),
                    diag_vote_accuracy=("diag_vote_matches_final", "mean"),
                    diag_col_accuracy=("diag_col_vote_matches_final", "mean"))
               .round(3).reset_index()
               .sort_values("n", ascending=False))
        print(per.head(15).to_string(index=False))

    # DIAGNOSTIC-VOTE overall confusion matrix (using the strict set).
    if len(strict) and keep:
        cm_d = pd.crosstab(
            strict["final_submission_label"],
            strict["diag_vote_label"],
            normalize="index",
        ).round(3)
        print("\nDiagnostic-vote (cell) confusion (strict cognitive set):")
        print(cm_d.to_string())
        cm_d.to_csv("prior_analysis/chunk_vote_diag_confusion.csv")

        cm_dc = pd.crosstab(
            strict["final_submission_label"],
            strict["diag_col_vote_label"],
            normalize="index",
        ).round(3)
        print("\nColor-aware diagnostic-vote confusion (strict cognitive set):")
        print(cm_dc.to_string())
        cm_dc.to_csv("prior_analysis/chunk_vote_diag_color_confusion.csv")

        # Pairwise discrimination between Top Errors — four voting
        # schemes side by side.
        from itertools import combinations
        err_all = merged[merged["final_submission_label"].isin(
            ["Top Error 1", "Top Error 2", "Top Error 3"])].copy()

        print("\n" + "=" * 78)
        print("PAIRWISE DISCRIMINATION among error-ending trajectories")
        print("(four voting schemes; restricted to vote ∈ {TE_a, TE_b})")
        print("=" * 78)
        pair_rows = []
        for a, b in combinations(["Top Error 1", "Top Error 2", "Top Error 3"],
                                 2):
            for scheme, col in [
                    ("cell-only",            "diag_vote_label"),
                    ("color-aware (winner)", "diag_col_vote_label"),
                    ("color-aware (weight)", "diag_col_w_label"),
                    ("color-aware (late)",   "diag_col_wl_label"),
            ]:
                sub = err_all[err_all["final_submission_label"].isin([a, b])
                              & err_all[col].isin([a, b])]
                if len(sub) < 20:
                    continue
                acc_p = (sub[col] == sub["final_submission_label"]).mean()
                base = max((sub["final_submission_label"] == a).mean(),
                           (sub["final_submission_label"] == b).mean())
                pair_rows.append({
                    "pair": f"{a} vs {b}", "scheme": scheme, "n": len(sub),
                    "acc": float(acc_p), "majority_baseline": float(base),
                    "above_baseline_pp": float(100 * (acc_p - base)),
                })
        pd.DataFrame(pair_rows).to_csv(
            "prior_analysis/chunk_error_pairwise.csv", index=False)
        print(pd.DataFrame(pair_rows).round(3).to_string(index=False))

        # With-abstention: use the weighted color-aware score. Only
        # classify trajectories where the winning score is >= K.
        print("\n" + "=" * 78)
        print("WEIGHTED color-aware vote with ABSTENTION")
        print("(only classify trajectories whose winning label crossed K total "
              "triple-matches)")
        print("=" * 78)
        err_all_nz = err_all[err_all["diag_col_w_total"] > 0].copy()
        abst_rows = []
        for K in [1, 2, 3, 5, 8, 12]:
            kept = err_all_nz[err_all_nz["diag_col_w_score"] >= K].copy()
            if not len(kept):
                continue
            acc = (kept["diag_col_w_label"]
                   == kept["final_submission_label"]).mean()
            coverage = len(kept) / len(err_all)
            # majority baseline within the kept subset
            mj = kept["final_submission_label"].value_counts(
                normalize=True).iloc[0] if len(kept) else 0.0
            abst_rows.append({
                "min_score_K": K,
                "n_kept": len(kept),
                "coverage_of_errors": float(coverage),
                "accuracy_on_kept": float(acc),
                "majority_baseline_on_kept": float(mj),
                "above_baseline_pp": float(100 * (acc - mj)),
            })
        ab_df = pd.DataFrame(abst_rows)
        ab_df.to_csv("prior_analysis/chunk_error_abstention.csv",
                     index=False)
        print(ab_df.round(3).to_string(index=False))

        # Condition on task diagnosability — only tasks whose TE-pool
        # is big enough that chunks have something to detect.
        print("\n" + "=" * 78)
        print("ACCURACY BY TASK DIAGNOSABILITY")
        print("(diag_pool_te_total = # mutually-exclusive (y,x,color) triples "
              "across TEs on that task)")
        print("=" * 78)
        bins = [(0, 5), (5, 15), (15, 40), (40, 500)]
        diag_rows = []
        for lo, hi in bins:
            sub = err_all[(err_all["diag_pool_te_total"] >= lo)
                          & (err_all["diag_pool_te_total"] < hi)]
            if len(sub) < 20:
                continue
            acc = (sub["diag_col_w_label"]
                   == sub["final_submission_label"]).mean()
            mj = sub["final_submission_label"].value_counts(
                normalize=True).iloc[0]
            diag_rows.append({
                "pool_range": f"[{lo},{hi})", "n": len(sub),
                "acc_weighted_color": float(acc),
                "majority_baseline": float(mj),
                "above_baseline_pp": float(100 * (acc - mj)),
            })
        db_df = pd.DataFrame(diag_rows)
        db_df.to_csv("prior_analysis/chunk_error_by_diagnosability.csv",
                     index=False)
        print(db_df.round(3).to_string(index=False))

        # Early vs late: can the wrong plan be predicted before the
        # trajectory ends?
        print("\n" + "=" * 78)
        print("EARLY vs LATE chunks: can we predict TE label BEFORE "
              "the trajectory ends?")
        print("=" * 78)
        err_full = merged[merged["final_submission_label"].isin(
            ["Top Error 1", "Top Error 2", "Top Error 3"])]
        # For a fair comparison pick the balanced subset: TE2 vs TE3,
        # high diagnosability (pool >= 40).
        bal = err_full[err_full["diag_pool_te_total"] >= 40].copy()
        bal = bal[bal["final_submission_label"].isin(
            ["Top Error 2", "Top Error 3"])]
        print(f"(restricted to balanced subset: TE2 vs TE3 with pool>=40, "
              f"n={len(bal)})")
        for name, col in [
                ("first-quarter chunks", "diag_col_wq_label"),
                ("first-half chunks",    "diag_col_we_label"),
                ("last-half chunks",     "diag_col_wl_label"),
                ("all chunks",           "diag_col_w_label"),
        ]:
            sub = bal[bal[col].isin(["Top Error 2", "Top Error 3"])]
            if len(sub) < 20:
                print(f"  {name}: n={len(sub)}  (too small)")
                continue
            acc = (sub[col] == sub["final_submission_label"]).mean()
            base = max((sub["final_submission_label"] == "Top Error 2").mean(),
                       (sub["final_submission_label"] == "Top Error 3").mean())
            print(f"  {name:<22}  n={len(sub):3d}  acc={acc:.3f}  "
                  f"base={base:.3f}  above={100*(acc-base):+.1f}pp")

        # Subject-level consistency
        print("\n" + "=" * 78)
        print("SUBJECT-LEVEL ERROR TENDENCIES")
        print("(does a single subject have a personal TE-bias across trials?)")
        print("=" * 78)
        known = merged[merged["final_submission_label"].isin(
            ["Success", "Top Error 1", "Top Error 2", "Top Error 3"])].copy()
        per_subj = (known.groupby("subject_id")
                    .agg(n_trials=("task_id", "count"),
                         error_rate=("final_submission_label",
                                     lambda s: (s != "Success").mean()),
                         te1_rate=("final_submission_label",
                                   lambda s: (s == "Top Error 1").mean()),
                         te2_rate=("final_submission_label",
                                   lambda s: (s == "Top Error 2").mean()),
                         te3_rate=("final_submission_label",
                                   lambda s: (s == "Top Error 3").mean()),
                         )
                    .reset_index())
        per_subj = per_subj[per_subj["n_trials"] >= 5]
        per_subj.to_csv("prior_analysis/chunk_error_per_subject.csv",
                        index=False)
        print(f"n subjects with >=5 trials: {len(per_subj)}")
        print("\nerror-rate distribution across subjects:")
        print(per_subj[["n_trials", "error_rate", "te1_rate",
                        "te2_rate", "te3_rate"]]
              .describe().round(3).to_string())

        # Split-half reliability of subject error rate. If errors are a
        # stable personal trait, a subject's error rate on a random half
        # of their trials should correlate strongly with their rate on
        # the other half.
        from scipy.stats import spearmanr
        rng = np.random.default_rng(0)
        sh_err, sh_te1, sh_te2, sh_te3 = [], [], [], []
        for _ in range(50):
            splits = {"A": [], "B": []}
            for subj, sub in known.groupby("subject_id"):
                if len(sub) < 6:
                    continue
                idx = rng.permutation(len(sub))
                half = len(idx) // 2
                a = sub.iloc[idx[:half]]; b = sub.iloc[idx[half:2*half]]
                splits["A"].append({
                    "subject_id": subj,
                    "error_rate": (a.final_submission_label != "Success").mean(),
                    "te1_rate": (a.final_submission_label == "Top Error 1").mean(),
                    "te2_rate": (a.final_submission_label == "Top Error 2").mean(),
                    "te3_rate": (a.final_submission_label == "Top Error 3").mean(),
                })
                splits["B"].append({
                    "subject_id": subj,
                    "error_rate": (b.final_submission_label != "Success").mean(),
                    "te1_rate": (b.final_submission_label == "Top Error 1").mean(),
                    "te2_rate": (b.final_submission_label == "Top Error 2").mean(),
                    "te3_rate": (b.final_submission_label == "Top Error 3").mean(),
                })
            A = pd.DataFrame(splits["A"])
            B = pd.DataFrame(splits["B"])
            m = A.merge(B, on="subject_id", suffixes=("_a", "_b"))
            if len(m) < 20:
                continue
            sh_err.append(spearmanr(m.error_rate_a, m.error_rate_b)[0])
            sh_te1.append(spearmanr(m.te1_rate_a, m.te1_rate_b)[0])
            sh_te2.append(spearmanr(m.te2_rate_a, m.te2_rate_b)[0])
            sh_te3.append(spearmanr(m.te3_rate_a, m.te3_rate_b)[0])
        print("\nSplit-half Spearman (subject-level, 50 random splits):")
        print(f"  error_rate: mean rho = {np.mean(sh_err):+.3f}  "
              f"(higher = error-prone is stable personal trait)")
        print(f"  te1_rate  : mean rho = {np.mean(sh_te1):+.3f}")
        print(f"  te2_rate  : mean rho = {np.mean(sh_te2):+.3f}")
        print(f"  te3_rate  : mean rho = {np.mean(sh_te3):+.3f}")

        # Variance decomposition: how much of "trial ends on TE1" is
        # explained by task identity vs subject identity? One-way
        # ANOVA-style: fraction of variance captured by each grouping.
        def _eta_squared(dfv, grouping_col, outcome_col):
            grand = dfv[outcome_col].mean()
            ss_total = ((dfv[outcome_col] - grand) ** 2).sum()
            ss_between = 0.0
            for _, g in dfv.groupby(grouping_col):
                ss_between += len(g) * (g[outcome_col].mean() - grand) ** 2
            return float(ss_between / max(ss_total, 1e-9))

        print("\nVariance decomposition: what explains 'trial ends on TE_k'?")
        print(f"{'outcome':<22}{'by task':>10}{'by subject':>14}")
        for tag in ["Top Error 1", "Top Error 2", "Top Error 3"]:
            col = f"is_{tag.replace(' ', '_').lower()}"
            known[col] = (known.final_submission_label == tag).astype(int)
            eta_task = _eta_squared(known, "task_id", col)
            eta_subj = _eta_squared(known, "subject_id", col)
            print(f"  {tag:<22}{eta_task:>10.3f}{eta_subj:>14.3f}")
        known["is_error"] = (known.final_submission_label != "Success").astype(int)
        eta_task = _eta_squared(known, "task_id", "is_error")
        eta_subj = _eta_squared(known, "subject_id", "is_error")
        print(f"  {'any error':<22}{eta_task:>10.3f}{eta_subj:>14.3f}")


if __name__ == "__main__":
    main()

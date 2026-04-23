"""
Are pause-segmented chunks real cognitive units, or analytical artifacts?

Three tests against the same 75 tasks × Experiment-2 subjects:

    (A) Inter-edit RT distribution — if pauses are real cognitive transitions,
        the distribution should be bimodal (fast intra-chunk edits vs slow
        between-chunk planning pauses). A unimodal long-tailed distribution
        would mean pause-thresholds are just cutting noise.

    (B) Structural coherence vs null models. For each trajectory we
        re-chunk two ways:
            Null-RT:   shuffle inter-edit RTs within the trajectory, then
                       re-segment with the same pause rule. Preserves RT
                       distribution but breaks the pause-ordering.
            Null-Cut:  pick cut points uniformly at random while keeping
                       the same number of chunks per trajectory.
        We measure three chunk properties that should be HIGH if chunks are
        cognitive units:
            color_homogeneity — fraction of edits using the chunk's dominant color
            is_connected      — whether the cells form one 4-connected component
            success_iou_best  — IoU with the best-matching Success component
        Real chunks beating BOTH nulls on all three is decisive evidence.

    (C) (optional) Cross-subject agreement on chunking. For each task,
        compute a cell-pair co-chunk probability (how often cells u, v end
        up in the same chunk across subjects). Low variance ≈ task-level
        structure. We report the mean pairwise co-chunk entropy per task.

Outputs:
    prior_analysis/chunks_cognitive_units_summary.csv   (per-task table)
    prior_analysis/chunks_cognitive_units_by_chunk.csv  (per-chunk per-cond)
    prior_analysis/chunks_cognitive_units_figure.png
"""

from __future__ import annotations

import _paths  # noqa: F401
import argparse
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from human_chunking import (
    identify_chunks, chunk_features, best_iou_with_success, _success_components,
)
from human_style_features import _parse_trajectory, DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR
from human_targets import available_task_ids


# ---------------------------------------------------------------------------
# Null models
# ---------------------------------------------------------------------------

def null_shuffled_rt(edits: List[Dict], rng: random.Random) -> List[Dict]:
    """Shuffle RTs within the trajectory; re-segment with same threshold rule."""
    if not edits:
        return []
    rts = [e.get("rt", float("nan")) for e in edits]
    order = list(range(len(edits)))
    rng.shuffle(order)
    out = []
    for i, src in enumerate(order):
        e = dict(edits[i])
        e["rt"] = rts[src]
        out.append(e)
    return identify_chunks(out)


def null_random_cuts(edits: List[Dict], n_chunks: int,
                     rng: random.Random) -> List[List[Dict]]:
    """Preserve chunk count but pick cut points uniformly at random."""
    n = len(edits)
    if n == 0 or n_chunks <= 1:
        return [edits] if edits else []
    k = min(n_chunks - 1, n - 1)
    cuts = sorted(rng.sample(range(1, n), k)) if k > 0 else []
    out, prev = [], 0
    for c in cuts + [n]:
        if c > prev:
            out.append(edits[prev:c])
        prev = c
    return out


# ---------------------------------------------------------------------------
# Per-trajectory processing
# ---------------------------------------------------------------------------

def _collect_chunk_metrics(chunks, comp_masks):
    rows = []
    for ch in chunks:
        if not ch:
            continue
        feats = chunk_features(ch)
        iou, _ = best_iou_with_success(ch, comp_masks)
        rows.append({
            "size": feats["size"],
            "n_cells": feats["n_cells"],
            "color_homogeneity": feats["color_homogeneity"],
            "is_connected": bool(feats["is_connected"]),
            "success_iou_best": iou,
        })
    return rows


def _chunk_threshold(edits: List[Dict],
                     pause_factor: float = 2.0,
                     min_pause_ms: float = 500.0) -> float:
    rts = np.array([e["rt"] for e in edits
                    if not np.isnan(e.get("rt", float("nan")))])
    median_rt = float(np.median(rts)) if rts.size else 0.0
    return max(pause_factor * median_rt, min_pause_ms)


def _rts_tagged(edits: List[Dict], threshold: float) -> List[Tuple[float, bool]]:
    """Return list of (rt, is_inter) for each inter-edit gap."""
    out = []
    for e in edits[1:]:
        rt = e.get("rt", float("nan"))
        if np.isnan(rt):
            continue
        out.append((float(rt), bool(rt > threshold)))
    return out


def _cells_to_chunk_partition(chunks: List[List[Dict]]) -> Dict[Tuple[int, int], int]:
    """Map (y, x) -> chunk_index for a subject. Last write wins (within a chunk)."""
    m: Dict[Tuple[int, int], int] = {}
    for i, ch in enumerate(chunks):
        for e in ch:
            m[(int(e["y"]), int(e["x"]))] = i
    return m


def process_trajectory(task_id: str, subj: str, edits: List[Dict],
                       comp_masks, rng: random.Random):
    """Return lists of per-chunk metrics under three conditions + the RTs."""
    threshold = _chunk_threshold(edits)
    rts_tagged = _rts_tagged(edits, threshold)
    real = identify_chunks(edits)
    real_metrics = _collect_chunk_metrics(real, comp_masks)
    if not real_metrics:
        return None
    null_rt = null_shuffled_rt(edits, rng)
    null_rt_metrics = _collect_chunk_metrics(null_rt, comp_masks)
    null_cut = null_random_cuts(edits, n_chunks=len(real), rng=rng)
    null_cut_metrics = _collect_chunk_metrics(null_cut, comp_masks)

    return {
        "rts_tagged": rts_tagged,
        "real": real_metrics,
        "null_rt": null_rt_metrics,
        "null_cut": null_cut_metrics,
        "partition_real": _cells_to_chunk_partition(real),
        "partition_null_cut": _cells_to_chunk_partition(null_cut),
        "n_chunks_real": len(real),
        "n_chunks_null_rt": len(null_rt),
    }


# ---------------------------------------------------------------------------
# Cross-subject ARI
# ---------------------------------------------------------------------------

def _ari_pair(p1: Dict[Tuple[int, int], int],
              p2: Dict[Tuple[int, int], int]) -> float:
    common = sorted(set(p1) & set(p2))
    if len(common) < 2:
        return float("nan")
    labels1 = [p1[c] for c in common]
    labels2 = [p2[c] for c in common]
    return float(adjusted_rand_score(labels1, labels2))


def _task_ari(partitions: List[Dict[Tuple[int, int], int]],
              rng: random.Random, n_pairs: int = 50) -> List[float]:
    if len(partitions) < 2:
        return []
    pairs = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 5:
        i, j = rng.sample(range(len(partitions)), 2)
        ari = _ari_pair(partitions[i], partitions[j])
        if not np.isnan(ari):
            pairs.append(ari)
        attempts += 1
    return pairs


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="prior_analysis")
    ap.add_argument("--max_tasks", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    task_ids = available_task_ids()
    if args.max_tasks:
        task_ids = task_ids[:args.max_tasks]

    rng = random.Random(args.seed)

    all_rts_tagged: List[Tuple[float, bool]] = []
    chunk_rows: List[Dict] = []
    ari_rows: List[Dict] = []

    for i, tid in enumerate(task_ids, 1):
        _, comp_masks, _ = _success_components(tid)
        traj_dir = os.path.join(DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR, f"{tid}.json")
        if not os.path.isdir(traj_dir):
            continue

        task_partitions_real = []
        task_partitions_null = []
        for fname in sorted(os.listdir(traj_dir)):
            if not fname.startswith("subj_"):
                continue
            tr = _parse_trajectory(os.path.join(traj_dir, fname))
            edits = tr.get("edits") or []
            if not edits:
                continue
            subj = fname.split("_")[1]

            result = process_trajectory(tid, subj, edits, comp_masks, rng)
            if result is None:
                continue

            all_rts_tagged.extend(result["rts_tagged"])
            for cond in ("real", "null_rt", "null_cut"):
                for r in result[cond]:
                    r2 = dict(r)
                    r2.update({"task_id": tid, "subject_id": subj, "condition": cond})
                    chunk_rows.append(r2)
            task_partitions_real.append(result["partition_real"])
            task_partitions_null.append(result["partition_null_cut"])

        # Cross-subject ARI for this task
        ari_real = _task_ari(task_partitions_real, rng, n_pairs=50)
        ari_null = _task_ari(task_partitions_null, rng, n_pairs=50)
        for a in ari_real:
            ari_rows.append({"task_id": tid, "condition": "real", "ari": a})
        for a in ari_null:
            ari_rows.append({"task_id": tid, "condition": "null_cut", "ari": a})

        if i % 10 == 0 or i == len(task_ids):
            print(f"  [{i:3d}/{len(task_ids)}] tasks processed, "
                  f"chunks so far: {len(chunk_rows)}, "
                  f"ari pairs: {len(ari_rows)}")

    df = pd.DataFrame(chunk_rows)
    df.to_csv(os.path.join(args.out_dir,
                           "chunks_cognitive_units_by_chunk.csv"), index=False)
    print(f"[wrote] chunks_cognitive_units_by_chunk.csv "
          f"({len(df)} rows)")

    # Per-condition summary
    summary = (df.groupby("condition")
                 .agg(n_chunks=("size", "size"),
                      color_homogeneity_mean=("color_homogeneity", "mean"),
                      frac_connected=("is_connected", "mean"),
                      success_iou_mean=("success_iou_best", "mean"),
                      size_mean=("size", "mean"),
                      n_cells_mean=("n_cells", "mean"))
                 .reset_index())
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(args.out_dir,
                                "chunks_cognitive_units_summary.csv"),
                   index=False)

    rts_arr = np.array(all_rts_tagged,
                       dtype=[("rt", "f8"), ("is_inter", "?")])
    np.save(os.path.join(args.out_dir, "chunks_cognitive_units_rts_tagged.npy"),
            rts_arr)
    print(f"[wrote] tagged inter-edit RTs: n={len(rts_arr)} "
          f"(inter fraction = {rts_arr['is_inter'].mean():.3f})")

    ari_df = pd.DataFrame(ari_rows)
    ari_df.to_csv(os.path.join(args.out_dir,
                               "chunks_cognitive_units_ari.csv"), index=False)
    ari_summary = ari_df.groupby("condition")["ari"].agg(
        ["count", "mean", "std"])
    print(ari_summary)


if __name__ == "__main__":
    main()

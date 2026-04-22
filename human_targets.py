"""
Human target loader for CogARC Experiment 2.

For each of the 75 CogARC tasks this module exposes the set of common
first-attempt human responses on the task's test input, plus their
empirical probabilities computed from Experiment 2 raw submissions
(only _submission1.json files, per the CogARC README).

Exposed API:
    human_targets(task_id, data_root=DEFAULT_DATA_ROOT) -> dict
        {
            "task_id":  str,
            "labels":   ["Success", "Top Error 1", ...],        # ordered
            "grids":    [np.ndarray, ...],                      # parallel to labels
            "counts":   {label: int, "Other": int},
            "probs":    {label: float},     # fraction of submission1 files
            "n_submissions": int,
        }

    available_task_ids(data_root=DEFAULT_DATA_ROOT) -> List[str]
"""

from __future__ import annotations

import _paths  # noqa: F401  (sys.path bootstrap, safe under both invocations)

import json
import os
from collections import Counter
from functools import lru_cache
from typing import Dict, List

import numpy as np


# Data lives at <repo_root>/data/CogARC-dataRepository (gitignored).
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_ROOT = os.path.join(_HERE, "data", "CogARC-dataRepository")

_COMMON_DIR = "Common solutions"
_EXP2_SUB_DIR = os.path.join("Behavioral data", "Experiment 2", "Submissions")


def _load_grid(path: str) -> np.ndarray:
    with open(path, "r") as f:
        return np.asarray(json.load(f)["grid_data"], dtype=np.int64)


def _grid_key(grid: np.ndarray) -> tuple:
    return tuple(tuple(int(v) for v in row) for row in grid)


def _label_from_filename(fname: str, task_id: str) -> str:
    # e.g. "00d62c1b_Top Error 1.json" -> "Top Error 1"
    stem = fname[len(task_id) + 1:] if fname.startswith(task_id + "_") else fname
    return stem.replace(".json", "").strip()


def _sort_labels(labels: List[str]) -> List[str]:
    # Success first, then Top Error 1..N by numeric suffix
    def key(lbl: str):
        if lbl == "Success":
            return (0, 0)
        if lbl.startswith("Top Error"):
            try:
                return (1, int(lbl.split()[-1]))
            except ValueError:
                return (1, 999)
        return (2, lbl)
    return sorted(labels, key=key)


def available_task_ids(data_root: str = DEFAULT_DATA_ROOT) -> List[str]:
    root = os.path.join(data_root, _COMMON_DIR)
    return sorted(d.replace(".json", "") for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)))


@lru_cache(maxsize=None)
def human_targets(task_id: str, data_root: str = DEFAULT_DATA_ROOT) -> Dict:
    common_dir = os.path.join(data_root, _COMMON_DIR, f"{task_id}.json")
    if not os.path.isdir(common_dir):
        raise FileNotFoundError(f"No common-solutions folder for task {task_id}")

    # Load all common-solution grids
    fname_to_grid: Dict[str, np.ndarray] = {}
    for fname in sorted(os.listdir(common_dir)):
        if not fname.endswith(".json"):
            continue
        fname_to_grid[fname] = _load_grid(os.path.join(common_dir, fname))

    labels = [_label_from_filename(f, task_id) for f in fname_to_grid]
    labels = _sort_labels(labels)
    grids = [fname_to_grid[f"{task_id}_{lbl}.json"] for lbl in labels]

    # Build key -> label lookup for matching raw submissions
    key_to_label = {_grid_key(g): lbl for g, lbl in zip(grids, labels)}

    # Walk Exp 2 submission1 files and tally
    sub_dir = os.path.join(data_root, _EXP2_SUB_DIR, f"{task_id}.json")
    counts: Counter = Counter()
    n_total = 0
    if os.path.isdir(sub_dir):
        for fname in os.listdir(sub_dir):
            if not fname.endswith("_submission1.json"):
                continue
            g = _load_grid(os.path.join(sub_dir, fname))
            lbl = key_to_label.get(_grid_key(g), "Other")
            counts[lbl] += 1
            n_total += 1

    probs = {lbl: (counts[lbl] / n_total if n_total else 0.0) for lbl in labels}
    probs["Other"] = (counts.get("Other", 0) / n_total) if n_total else 0.0

    return {
        "task_id": task_id,
        "labels": labels,
        "grids": grids,
        "counts": dict(counts),
        "probs": probs,
        "n_submissions": n_total,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", default=None,
                    help="If omitted, prints a summary across all tasks.")
    args = ap.parse_args()

    if args.task_id:
        t = human_targets(args.task_id)
        print(f"task_id: {t['task_id']}")
        print(f"n_submissions (Exp 2, submission1): {t['n_submissions']}")
        for lbl in t["labels"]:
            p = t["probs"][lbl]
            c = t["counts"].get(lbl, 0)
            print(f"  {lbl:<14}  n={c:>4}  p={p:.1%}  shape={t['grids'][t['labels'].index(lbl)].shape}")
        print(f"  {'Other':<14}  n={t['counts'].get('Other',0):>4}  p={t['probs']['Other']:.1%}")
    else:
        ids = available_task_ids()
        print(f"{len(ids)} tasks. Top-1 human response summary:")
        rows = []
        for tid in ids:
            t = human_targets(tid)
            if not t["n_submissions"]:
                continue
            counts_over_known = {lbl: t["counts"].get(lbl, 0) for lbl in t["labels"]}
            top_lbl, top_n = max(counts_over_known.items(), key=lambda kv: kv[1])
            rows.append((tid, t["n_submissions"], top_lbl, t["probs"][top_lbl]))
        for tid, n, lbl, p in rows:
            print(f"  {tid}  n={n:>3}  top={lbl:<12}  p={p:.1%}")

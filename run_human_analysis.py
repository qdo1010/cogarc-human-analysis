"""
Driver: run the human-matching GNN sweep over CogARC tasks and save
analyzable artifacts per task + an aggregate summary.

Artifact layout:

    <output_dir>/
        summary.json
        <task_id>/
            targets.json                    # human response distribution (Exp 2)
            runs.jsonl                      # one row per (prior, edge_rule, arch)
            weighted_best.json              # pointer to argmax-of-weighted_acc config
            winners/
                <label_slug>__config.json   # winning config + scores
                <label_slug>__pred.json     # predicted grid (2D int list)
                <label_slug>__graph.gpickle # NetworkX graph on the test input
                <label_slug>__model.pt      # trained model state dict

Usage:
    python run_human_analysis.py --task_ids 00d62c1b 12eac192
    python run_human_analysis.py --all
    python run_human_analysis.py --all --priors hierarchical color_adjacency \
        --archs medium --edge_rules none color_based --epochs 400
"""

from __future__ import annotations

import _paths  # noqa: F401  (sys.path bootstrap, safe under both invocations)

import argparse
import json
import os
import pickle
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from human_targets import available_task_ids, human_targets
from human_solver import (
    DEFAULT_ARCH_CFGS,
    DEFAULT_EDGE_RULES,
    RunResult,
    sweep_task,
)
from enhanced_graph_generator import ENHANCED_ABSTRACTIONS
from solver import load_task


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _slug(label: str) -> str:
    return label.lower().replace(" ", "_")


def _grid_to_list(g: Optional[np.ndarray]):
    if g is None:
        return None
    return g.astype(int).tolist()


def _arch_name(a: Dict) -> str:
    return a.get("name", f"h{a['hidden_nf']}_l{a['n_layers']}")


def _config_summary(r: RunResult) -> Dict:
    return {
        "abstraction": r.abstraction,
        "edge_rule": r.edge_rule,
        "edge_params": r.edge_params,
        "arch_cfg": r.arch_cfg,
        "train_loss": r.train_loss,
        "epochs_ran": r.epochs_ran,
        "elapsed_s": r.elapsed_s,
        "argmax_label": r.scores.get("argmax_label"),
        "pred_prob": r.scores.get("pred_prob"),
        "weighted_acc": r.scores.get("weighted_acc"),
        "per_label_acc": r.scores.get("per_label_acc"),
        "per_label_exact": r.scores.get("per_label_exact"),
        "error": r.error,
        "config_id": r.config_id,
    }


def _write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _write_run_artifacts(run_dir: str, label: str, r: RunResult) -> Dict:
    os.makedirs(run_dir, exist_ok=True)
    slug = _slug(label)
    cfg = _config_summary(r) | {"label": label}
    _write_json(os.path.join(run_dir, f"{slug}__config.json"), cfg)
    _write_json(os.path.join(run_dir, f"{slug}__pred.json"),
                {"label": label, "grid": _grid_to_list(r.pred_grid)})
    if r.train_graph_snapshot is not None:
        with open(os.path.join(run_dir, f"{slug}__graph.gpickle"), "wb") as f:
            pickle.dump(r.train_graph_snapshot, f)
    if r.model_state is not None:
        torch.save(r.model_state, os.path.join(run_dir, f"{slug}__model.pt"))
    return cfg


def _write_targets(path: str, targets: Dict) -> None:
    payload = {
        "task_id": targets["task_id"],
        "n_submissions": targets["n_submissions"],
        "labels": targets["labels"],
        "counts": targets["counts"],
        "probs": targets["probs"],
        "grids": {lbl: _grid_to_list(g) for lbl, g in zip(targets["labels"], targets["grids"])},
    }
    _write_json(path, payload)


# ---------------------------------------------------------------------------
# Per-task run
# ---------------------------------------------------------------------------

def run_one_task(
    task_id: str,
    task: Dict,
    output_dir: str,
    priors: List[str],
    arch_cfgs: List[Dict],
    edge_rules: List[Tuple[Optional[str], Dict]],
    device: str,
    epochs: int,
    patience: int,
    verbose: bool,
) -> Dict:
    targets = human_targets(task_id)
    if not targets["n_submissions"]:
        print(f"  [{task_id}] skip: no Exp 2 submissions")
        return {"task_id": task_id, "skipped": "no_exp2"}

    task_dir = os.path.join(output_dir, task_id)
    os.makedirs(task_dir, exist_ok=True)
    _write_targets(os.path.join(task_dir, "targets.json"), targets)

    result = sweep_task(
        task_id=task_id,
        task=task,
        targets=targets,
        priors=priors,
        arch_cfgs=arch_cfgs,
        edge_rules=edge_rules,
        device=device,
        epochs=epochs,
        patience=patience,
        keep_model_state=True,
        keep_graph_snapshot=True,
        verbose=verbose,
    )

    # runs.jsonl: one lightweight row per config (no grids, no model state)
    runs_path = os.path.join(task_dir, "runs.jsonl")
    with open(runs_path, "w") as f:
        for r in result["runs"]:
            f.write(json.dumps(_config_summary(r), default=str) + "\n")

    # Per-label winners: save config + pred + graph + model
    win_dir = os.path.join(task_dir, "winners")
    winner_summaries = {}
    for lbl, r in result["winners"].items():
        winner_summaries[lbl] = _write_run_artifacts(win_dir, lbl, r)

    # Also save the weighted-best config (may duplicate a per-label winner)
    wb = result["weighted_best"]
    if wb is not None:
        weighted_best_path = os.path.join(task_dir, "weighted_best.json")
        _write_json(weighted_best_path, _config_summary(wb))

    summary_row = {
        "task_id": task_id,
        "n_submissions": targets["n_submissions"],
        "probs": targets["probs"],
        "winners": {
            lbl: {
                "config_id": s["config_id"],
                "weighted_acc": s["weighted_acc"],
                "pixel_acc_for_label": s["per_label_acc"].get(lbl),
                "exact_match_for_label": s["per_label_exact"].get(lbl),
                "argmax_label": s["argmax_label"],
            }
            for lbl, s in winner_summaries.items()
        },
        "weighted_best": (
            {
                "config_id": _config_summary(wb)["config_id"],
                "weighted_acc": _config_summary(wb)["weighted_acc"],
                "argmax_label": _config_summary(wb)["argmax_label"],
            } if wb is not None else None
        ),
        "n_configs_tried": len(result["runs"]),
    }
    return summary_row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_edge_rules(names: List[str]) -> List[Tuple[Optional[str], Dict]]:
    preset = {name or "none": (name, params) for name, params in DEFAULT_EDGE_RULES}
    out = []
    for n in names:
        if n not in preset:
            raise SystemExit(f"Unknown edge rule: {n} (choices: {list(preset)})")
        out.append(preset[n])
    return out


def _parse_archs(names: List[str]) -> List[Dict]:
    preset = {a["name"]: a for a in DEFAULT_ARCH_CFGS}
    out = []
    for n in names:
        if n not in preset:
            raise SystemExit(f"Unknown arch preset: {n} (choices: {list(preset)})")
        out.append(preset[n])
    return out


def _auto_device(flag: str) -> str:
    if flag != "auto":
        return flag
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task_dir", default="training/",
                    help="Directory with ARC task JSONs (CogARC subset).")
    ap.add_argument("--output_dir", default="prior_analysis/human_optim/",
                    help="Where to write per-task artifacts and summary.json.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--task_ids", nargs="+", help="Run just these tasks.")
    g.add_argument("--all", action="store_true", help="Run all 75 CogARC tasks.")
    ap.add_argument("--priors", nargs="+", default=list(ENHANCED_ABSTRACTIONS.keys()),
                    help="Graph priors to sweep.")
    ap.add_argument("--archs", nargs="+", default=["small", "medium"],
                    help=f"Architecture presets. Choices: {[a['name'] for a in DEFAULT_ARCH_CFGS]}")
    ap.add_argument("--edge_rules", nargs="+", default=["none", "color_based", "percentile"],
                    help=f"Edge rules. Choices: {[(r or 'none') for r,_ in DEFAULT_EDGE_RULES]}")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--patience", type=int, default=120)
    ap.add_argument("--device", default="auto", help="cpu/cuda/mps/auto")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--use_recommender", action="store_true",
                    help="Restrict priors per task to the top-k from "
                         "human_style_features.recommend_priors. Overrides --priors per task.")
    ap.add_argument("--rec_top_k", type=int, default=4,
                    help="Top-k priors to keep when --use_recommender is set.")
    args = ap.parse_args()

    device = _auto_device(args.device)
    arch_cfgs = _parse_archs(args.archs)
    edge_rules = _parse_edge_rules(args.edge_rules)

    task_ids = available_task_ids() if args.all else args.task_ids
    os.makedirs(args.output_dir, exist_ok=True)

    # If using the behavioral recommender, pre-compute the 75-task population
    # once so per-task z-scores are consistent.
    recommender_pop = None
    if args.use_recommender:
        from human_style_features import all_style_features
        recommender_pop = all_style_features()

    print(f"Device: {device}")
    if args.use_recommender:
        print(f"Recommender ON, top-{args.rec_top_k} priors per task from edit-sequence features")
    print(f"Tasks: {len(task_ids)}  |  Edges: {len(edge_rules)}  |  Archs: {len(arch_cfgs)}  "
          f"|  Configs/task: up to {args.rec_top_k if args.use_recommender else len(args.priors)}*{len(edge_rules)}*{len(arch_cfgs)}")

    rows = []
    t0 = time.time()
    for i, tid in enumerate(task_ids, 1):
        task_path = os.path.join(args.task_dir, f"{tid}.json")
        if not os.path.exists(task_path):
            print(f"  [{tid}] missing task file, skipping")
            continue
        task = load_task(task_path)
        print(f"\n[{i}/{len(task_ids)}] {tid}")

        if args.use_recommender and recommender_pop is not None:
            from human_style_features import recommend_priors
            trow = recommender_pop[recommender_pop.task_id == tid]
            if len(trow) == 0:
                print(f"  [{tid}] no features -> falling back to --priors")
                priors_for_task = args.priors
            else:
                priors_for_task = recommend_priors(
                    trow.iloc[0].to_dict(), population=recommender_pop,
                    top_k=args.rec_top_k,
                )
                print(f"  recommended priors: {priors_for_task}")
        else:
            priors_for_task = args.priors

        row = run_one_task(
            task_id=tid, task=task,
            output_dir=args.output_dir,
            priors=priors_for_task, arch_cfgs=arch_cfgs, edge_rules=edge_rules,
            device=device, epochs=args.epochs, patience=args.patience,
            verbose=args.verbose,
        )
        rows.append(row)
        _write_json(os.path.join(args.output_dir, "summary.json"),
                    {"rows": rows, "elapsed_s": time.time() - t0,
                     "priors": args.priors, "archs": [a["name"] for a in arch_cfgs],
                     "edge_rules": args.edge_rules,
                     "epochs": args.epochs})

    print(f"\nDone in {time.time()-t0:.1f}s. "
          f"Summary: {os.path.join(args.output_dir,'summary.json')}")


if __name__ == "__main__":
    main()

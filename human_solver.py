"""
Human-matching solver for CogARC tasks.

The GNN is trained on each task's ARC train input->output pairs (that's how
it learns a transformation). Scoring is then performed on the task's test
input against the distribution of common Experiment 2 human responses
(Success + Top Error 1..K). The winner for each human label is kept so
that priors + architectures reproducing a specific human error (not just
the correct answer) can be analyzed.

Public entry points:
    score_against_humans(pred_grid, targets)
    train_and_score(task, abstraction, arch_cfg, targets, ...)
    sweep_task(task_id, task, targets, priors, arch_cfgs, ...)

Artifacts are intentionally returned in-memory; persistence lives in
run_human_analysis.py so this module stays single-responsibility.
"""

from __future__ import annotations

import _paths  # noqa: F401  (sys.path bootstrap, safe under both invocations)

import copy
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from enhanced_egnn import ARCSolverModel
from enhanced_graph_generator import (
    ENHANCED_ABSTRACTIONS,
    graph_to_tensors,
    _get_background_color,
    _grid_to_array,
)
from solver import generate_graph, ORIGINAL_ABSTRACTIONS


# ---------------------------------------------------------------------------
# Architecture presets (the "architecture" axis we analyze)
# ---------------------------------------------------------------------------

DEFAULT_ARCH_CFGS: List[Dict] = [
    {"name": "small",  "hidden_nf": 64,  "n_layers": 4, "num_heads": 4, "dropout": 0.1},
    {"name": "medium", "hidden_nf": 128, "n_layers": 6, "num_heads": 4, "dropout": 0.1},
]

DEFAULT_EDGE_RULES: List[Tuple[Optional[str], Dict]] = [
    (None, {}),
    ("color_based", {"color_threshold": 0.3}),
    ("percentile",  {"percentile": 0.5}),
]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    if pred.shape != target.shape:
        return 0.0
    return float((pred == target).sum()) / target.size


def exact_match(pred: np.ndarray, target: np.ndarray) -> bool:
    return pred.shape == target.shape and bool(np.array_equal(pred, target))


def score_against_humans(pred_grid: np.ndarray, targets: Dict) -> Dict:
    """
    Compare `pred_grid` against every human-response grid for the task.

    Returns:
        {
            "per_label_acc":   {label: float},   # pixel match
            "per_label_exact": {label: bool},
            "weighted_acc":    float,  # sum_k p_k * acc_k over known human labels
            "argmax_label":    Optional[str],  # exact-match label (None if no match)
            "pred_prob":       float,  # probability of this response under Exp 2
        }
    """
    per_acc, per_exact = {}, {}
    weighted = 0.0
    argmax_label: Optional[str] = None

    for lbl, grid in zip(targets["labels"], targets["grids"]):
        acc = pixel_accuracy(pred_grid, grid)
        ex = exact_match(pred_grid, grid)
        per_acc[lbl] = acc
        per_exact[lbl] = ex
        weighted += targets["probs"].get(lbl, 0.0) * acc
        if ex and argmax_label is None:
            argmax_label = lbl

    if argmax_label is not None:
        pred_prob = targets["probs"].get(argmax_label, 0.0)
    else:
        pred_prob = 0.0  # "Other"

    return {
        "per_label_acc": per_acc,
        "per_label_exact": per_exact,
        "weighted_acc": weighted,
        "argmax_label": argmax_label,
        "pred_prob": pred_prob,
    }


# ---------------------------------------------------------------------------
# Training (kept local so we can own the training curve + final state cleanly)
# ---------------------------------------------------------------------------

def _tensors_to_device(tensors: Dict, device: str) -> Dict:
    return {
        "node_features": tensors["node_features"].to(device),
        "coordinates":   tensors["coordinates"].to(device),
        "edge_index":    tensors["edge_index"].to(device),
        "edge_features": tensors["edge_features"].to(device),
        "node_to_pixels": tensors["node_to_pixels"],
    }


def _compute_loss(model, graph_data, target_grid, grid_h, grid_w):
    _, grid_logits, node_logits = model(graph_data, grid_h, grid_w)
    target = torch.tensor(target_grid, dtype=torch.long, device=grid_logits.device)
    grid_loss = nn.functional.cross_entropy(
        grid_logits.unsqueeze(0), target.unsqueeze(0), reduction="mean"
    )
    node_loss = torch.tensor(0.0, device=grid_logits.device)
    n_used = 0
    for i, pixels in enumerate(graph_data["node_to_pixels"]):
        if not pixels:
            continue
        colors = [int(target_grid[r, c]) for r, c in pixels
                  if 0 <= r < grid_h and 0 <= c < grid_w]
        if colors:
            majority = Counter(colors).most_common(1)[0][0]
            t = torch.tensor([majority], dtype=torch.long, device=node_logits.device)
            node_loss = node_loss + nn.functional.cross_entropy(node_logits[i:i+1], t)
            n_used += 1
    if n_used > 0:
        node_loss = node_loss / n_used
    return 0.7 * grid_loss + 0.3 * node_loss


def _build_train_pairs(task: Dict, abstraction: str,
                       edge_rule: Optional[str], edge_params: Dict):
    pairs = []
    for ex in task["train"]:
        inp = np.array(ex["input"])
        out = np.array(ex["output"])
        bg = _get_background_color(inp)
        G = generate_graph(inp, abstraction, bg, edge_rule, edge_params)
        if G is None or G.number_of_nodes() == 0:
            continue
        tensors = graph_to_tensors(G)
        if tensors is None:
            continue
        pairs.append((tensors, out, out.shape[0], out.shape[1]))
    return pairs


def _predict_test(model, task: Dict, abstraction: str, device: str,
                  edge_rule: Optional[str], edge_params: Dict) -> Optional[np.ndarray]:
    ex = task["test"][0]
    inp = np.array(ex["input"])
    bg = _get_background_color(inp)
    G = generate_graph(inp, abstraction, bg, edge_rule, edge_params)
    if G is None or G.number_of_nodes() == 0:
        return None
    tensors = graph_to_tensors(G)
    if tensors is None:
        return None
    gd = _tensors_to_device(tensors, device)

    # Infer output shape from the training pairs (same size vs ratio)
    train_in = np.array(task["train"][0]["input"])
    train_out = np.array(task["train"][0]["output"])
    if train_in.shape == train_out.shape:
        out_h, out_w = inp.shape
    else:
        h_ratio = train_out.shape[0] / train_in.shape[0]
        w_ratio = train_out.shape[1] / train_in.shape[1]
        out_h = max(1, int(round(inp.shape[0] * h_ratio)))
        out_w = max(1, int(round(inp.shape[1] * w_ratio)))

    with torch.no_grad():
        grid_pred, _, _ = model(gd, out_h, out_w)
    return grid_pred.detach().cpu().numpy().astype(np.int64)


@dataclass
class RunResult:
    abstraction: str
    edge_rule: Optional[str]
    edge_params: Dict
    arch_cfg: Dict
    train_loss: float                      # final (best) training loss
    pred_grid: Optional[np.ndarray]        # prediction on test input
    scores: Dict                           # score_against_humans(...)
    epochs_ran: int
    elapsed_s: float
    model_state: Optional[Dict] = None     # CPU state dict (optional, can be dropped)
    train_graph_snapshot: Optional[object] = None  # NetworkX graph on test input
    error: Optional[str] = None
    # populated by sweep_task for convenience:
    config_id: Optional[str] = None


def train_and_score(
    task: Dict,
    targets: Dict,
    abstraction: str,
    arch_cfg: Dict,
    edge_rule: Optional[str] = None,
    edge_params: Optional[Dict] = None,
    device: str = "cpu",
    epochs: int = 600,
    patience: int = 120,
    lr: float = 1e-3,
    keep_model_state: bool = True,
    keep_graph_snapshot: bool = True,
    seed: int = 0,
) -> RunResult:
    edge_params = edge_params or {}

    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    try:
        pairs = _build_train_pairs(task, abstraction, edge_rule, edge_params)
    except Exception as e:
        return RunResult(abstraction, edge_rule, edge_params, arch_cfg,
                         float("inf"), None, _empty_scores(targets), 0,
                         time.time() - t0, error=f"prior_build: {e}")

    if not pairs:
        return RunResult(abstraction, edge_rule, edge_params, arch_cfg,
                         float("inf"), None, _empty_scores(targets), 0,
                         time.time() - t0, error="no_pairs")

    sample = pairs[0][0]
    in_node_nf = sample["node_features"].shape[1]
    # edge_features tensor always has a stable second dim (num_edge_types+1),
    # even when empty — so read the width directly to avoid train/test mismatch.
    in_edge_nf = sample["edge_features"].shape[1]

    model = ARCSolverModel(
        in_node_nf=in_node_nf,
        hidden_nf=arch_cfg["hidden_nf"],
        num_colors=11,
        in_edge_nf=in_edge_nf,
        device=device,
        num_heads=arch_cfg["num_heads"],
        n_layers=arch_cfg["n_layers"],
        dropout=arch_cfg.get("dropout", 0.1),
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    best_state = None
    no_improve = 0
    epoch_ran = 0

    try:
        for epoch in range(epochs):
            epoch_ran = epoch + 1
            model.train()
            total = 0.0
            for tensors, tgrid, gh, gw in pairs:
                gd = _tensors_to_device(tensors, device)
                optimizer.zero_grad()
                loss = _compute_loss(model, gd, tgrid, gh, gw)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total += loss.item()
            scheduler.step()
            avg = total / len(pairs)

            if avg < best_loss - 1e-4:
                best_loss = avg
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= patience:
                break
    except Exception as e:
        return RunResult(abstraction, edge_rule, edge_params, arch_cfg,
                         best_loss, None, _empty_scores(targets), epoch_ran,
                         time.time() - t0, model_state=best_state,
                         error=f"train: {e}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    try:
        pred = _predict_test(model, task, abstraction, device, edge_rule, edge_params)
    except Exception as e:
        return RunResult(abstraction, edge_rule, edge_params, arch_cfg,
                         best_loss, None, _empty_scores(targets), epoch_ran,
                         time.time() - t0,
                         model_state=best_state if keep_model_state else None,
                         error=f"predict: {e}")

    scores = score_against_humans(pred, targets) if pred is not None else _empty_scores(targets)

    snapshot = None
    if keep_graph_snapshot and pred is not None:
        try:
            inp = np.array(task["test"][0]["input"])
            bg = _get_background_color(inp)
            snapshot = generate_graph(inp, abstraction, bg, edge_rule, edge_params)
        except Exception:
            snapshot = None

    return RunResult(
        abstraction=abstraction,
        edge_rule=edge_rule,
        edge_params=edge_params,
        arch_cfg=arch_cfg,
        train_loss=best_loss,
        pred_grid=pred,
        scores=scores,
        epochs_ran=epoch_ran,
        elapsed_s=time.time() - t0,
        model_state=best_state if keep_model_state else None,
        train_graph_snapshot=snapshot,
    )


def _empty_scores(targets: Dict) -> Dict:
    return {
        "per_label_acc":   {lbl: 0.0 for lbl in targets["labels"]},
        "per_label_exact": {lbl: False for lbl in targets["labels"]},
        "weighted_acc":    0.0,
        "argmax_label":    None,
        "pred_prob":       0.0,
    }


# ---------------------------------------------------------------------------
# Sweep: iterate over priors + edge rules + architectures, pick winners
# ---------------------------------------------------------------------------

def _make_config_id(r: RunResult) -> str:
    er = "none" if not r.edge_rule else r.edge_rule
    return f"{r.abstraction}|{er}|{r.arch_cfg['name']}"


def sweep_task(
    task_id: str,
    task: Dict,
    targets: Dict,
    priors: Optional[List[str]] = None,
    arch_cfgs: Optional[List[Dict]] = None,
    edge_rules: Optional[List[Tuple[Optional[str], Dict]]] = None,
    device: str = "cpu",
    epochs: int = 600,
    patience: int = 120,
    keep_model_state: bool = True,
    keep_graph_snapshot: bool = True,
    verbose: bool = False,
) -> Dict:
    """
    Run the full (prior x edge_rule x arch) sweep for one task.

    Returns:
        {
            "task_id":        task_id,
            "targets":        targets,
            "runs":           List[RunResult],   # every config tried
            "winners":        Dict[label, RunResult],  # best exact/pixel match per label
            "weighted_best":  RunResult,         # config with highest weighted_acc
        }
    """
    priors = priors or list(ENHANCED_ABSTRACTIONS.keys())
    arch_cfgs = arch_cfgs or DEFAULT_ARCH_CFGS
    edge_rules = edge_rules or DEFAULT_EDGE_RULES

    runs: List[RunResult] = []
    for prior in priors:
        for (rule, params) in edge_rules:
            for arch in arch_cfgs:
                r = train_and_score(
                    task=task, targets=targets,
                    abstraction=prior, arch_cfg=arch,
                    edge_rule=rule, edge_params=params,
                    device=device, epochs=epochs, patience=patience,
                    keep_model_state=keep_model_state,
                    keep_graph_snapshot=keep_graph_snapshot,
                )
                r.config_id = _make_config_id(r)
                runs.append(r)
                if verbose:
                    tag = r.scores.get("argmax_label") or "—"
                    print(
                        f"  [{task_id}] {r.config_id:<45}  "
                        f"loss={r.train_loss:.4f}  "
                        f"wacc={r.scores['weighted_acc']:.3f}  "
                        f"match={tag:<12}  ({r.elapsed_s:.1f}s"
                        f"{'  ERR '+r.error if r.error else ''})"
                    )

    winners: Dict[str, RunResult] = {}
    for lbl in targets["labels"]:
        best = None
        best_score = -1.0
        for r in runs:
            if r.pred_grid is None:
                continue
            acc = r.scores["per_label_acc"].get(lbl, 0.0)
            # prefer exact match, then higher pixel acc, then lower train loss
            score = (2.0 if r.scores["per_label_exact"].get(lbl, False) else 0.0) + acc
            if score > best_score + 1e-9 or (
                abs(score - best_score) < 1e-9 and best is not None and r.train_loss < best.train_loss
            ):
                best_score = score
                best = r
        if best is not None:
            winners[lbl] = best

    weighted_best = None
    for r in runs:
        if r.pred_grid is None:
            continue
        if weighted_best is None or r.scores["weighted_acc"] > weighted_best.scores["weighted_acc"] + 1e-9:
            weighted_best = r

    return {
        "task_id": task_id,
        "targets": targets,
        "runs": runs,
        "winners": winners,
        "weighted_best": weighted_best,
    }

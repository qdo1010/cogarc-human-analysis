"""
Figure: human chunking compared against six canonical strategies.

    A  Radar chart of pooled chunk-feature fingerprints (humans + all
       strategies). Shows at a glance whether humans match any single
       canonical strategy or fall in between.

    B  Per-task winner distribution: for each of the 75 tasks, which
       strategy's feature vector is closest to humans?

    C  Two concrete examples — one task where object-first wins, one
       where row-first wins — with Success grids and the strategies'
       chunks side by side.

Output: prior_analysis/strategies_vs_human_figure.png
"""

from __future__ import annotations


import _paths  # noqa: F401

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from human_vs_strategies import (
    FEATURES_OF_INTEREST, STRATEGIES, _success_grid,
    strategy_color_first, strategy_nn_color_first, strategy_object_first,
    strategy_row_first, strategy_col_first, strategy_random_k3,
)

ARC_COLORS_HEX = {
    0: "#2B2B2B", 1: "#248ADA", 2: "#C71010", 3: "#1FC719",
    4: "#F7DE28", 5: "#878494", 6: "#F954F2", 7: "#EE6000",
    8: "#6B23A9", 9: "#8B5A28",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 13,
    "figure.titlesize": 20,
})


# Consistent colors per strategy.
STRAT_COLORS = {
    "human":          "#222222",
    "object_first":   "#1f77b4",
    "color_first":    "#2ca02c",
    "nn_color_first": "#9467bd",
    "row_first":      "#ff7f0e",
    "col_first":      "#8c564b",
    "random_k3":      "#aaaaaa",
}


def _draw_grid(ax, grid, title=""):
    H, W = grid.shape
    for y in range(H):
        for x in range(W):
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1,
                                   facecolor=ARC_COLORS_HEX.get(int(grid[y, x]), "#000"),
                                   edgecolor="#e8e8e8", linewidth=0.3))
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=12, pad=4)


def _radar(ax, features, values_by_label):
    """Classic radar plot. values_by_label = {label: [v1, v2, ...]}."""
    N = len(features)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += [angles[0]]
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1"], fontsize=9, color="#888")
    ax.grid(True, linestyle="--", alpha=0.4)

    # Sort labels so 'human' is drawn last (on top).
    order = sorted(values_by_label.keys(),
                   key=lambda k: 0 if k != "human" else 1)
    for label in order:
        vals = values_by_label[label] + [values_by_label[label][0]]
        lw = 2.8 if label == "human" else 1.4
        alpha = 1.0 if label == "human" else 0.7
        ax.plot(angles, vals, color=STRAT_COLORS[label],
                lw=lw, label=label, alpha=alpha)
        if label == "human":
            ax.fill(angles, vals, color=STRAT_COLORS[label], alpha=0.12)


def build():
    all_df = pd.read_csv("prior_analysis/strategy_chunks_per_task.csv")
    dist_df = pd.read_csv("prior_analysis/strategy_vs_human_distance.csv")

    # Features to show on the radar. Normalise each to [0, 1] using
    # the pooled humans + strategies range so axes are comparable.
    radar_feats = [
        "frac_single_color",
        "frac_connected",
        "frac_same_row",
        "frac_same_col",
        "frac_multi_cc_same_color",
        "frac_one_cc",
        "mean_iou_cc",
        "mean_iou_color_class",
    ]
    # Task-pooled mean for each strategy.
    pooled = all_df.groupby("strategy")[radar_feats].mean()
    # Normalise column-wise to [0, 1] within the pooled table.
    col_min = pooled.min(axis=0)
    col_max = pooled.max(axis=0)
    col_range = (col_max - col_min).replace(0, 1)
    normed = ((pooled - col_min) / col_range).clip(0, 1)
    values_by_label = {
        label: normed.loc[label].tolist()
        for label in ["human"] + list(STRATEGIES.keys())
    }

    # ---- Figure layout: 2 rows. Top = A + B. Bottom = C examples. ----
    fig = plt.figure(figsize=(22, 15))
    gs = fig.add_gridspec(
        2, 3, width_ratios=[1.2, 1.0, 1.0], height_ratios=[1.0, 0.9],
        hspace=0.35, wspace=0.35,
    )

    # --- A Radar ---
    axA = fig.add_subplot(gs[0, 0], projection="polar")
    _radar(axA, radar_feats, values_by_label)
    axA.legend(loc="upper left", bbox_to_anchor=(-0.25, 1.15),
               frameon=False, fontsize=12)
    axA.set_title(
        "A.  Pooled chunk-feature fingerprint\n"
        "(humans in bold; all axes normalised to [0, 1])",
        loc="left", pad=30,
    )

    # --- B Per-task winner bars ---
    axB = fig.add_subplot(gs[0, 1:])
    winners = (dist_df.loc[dist_df.groupby("task_id")["distance"].idxmin()]
               ["strategy"].value_counts())
    order = ["object_first", "random_k3", "row_first", "col_first",
             "color_first", "nn_color_first"]
    vals = [winners.get(s, 0) for s in order]
    bars = axB.bar(np.arange(len(order)), vals,
                   color=[STRAT_COLORS[s] for s in order],
                   edgecolor="#222", linewidth=0.8)
    for i, v in enumerate(vals):
        axB.text(i, v + 0.6, str(v), ha="center", fontsize=13, fontweight="bold")
    axB.set_xticks(np.arange(len(order)))
    axB.set_xticklabels(order, rotation=15)
    axB.set_ylabel("tasks where strategy is closest to humans")
    axB.set_ylim(0, max(vals) * 1.18 + 3)
    axB.set_title(
        "B.  Per-task winner: which strategy best matches humans?  "
        f"(n = 75 tasks)",
        loc="left", pad=8,
    )
    axB.spines["top"].set_visible(False); axB.spines["right"].set_visible(False)

    # --- C Two concrete examples ---
    # Find one task where object_first wins and another where row_first wins.
    task_best = dist_df.loc[dist_df.groupby("task_id")["distance"].idxmin()]
    pick = {}
    for s in ("object_first", "row_first"):
        cand = task_best[task_best.strategy == s].sort_values("distance").head(5)
        if len(cand):
            pick[s] = cand.iloc[0]["task_id"]

    # ...for each, draw Success + object/row/col chunk overlays in small panels
    examples = [("object_first", pick.get("object_first")),
                ("row_first", pick.get("row_first"))]

    # Dedicate the bottom row to 2 examples × (Success + 3 strategies)
    outer = gs[1, :]
    ex_gs = outer.subgridspec(1, 2, wspace=0.1)
    for ex_idx, (winner, tid) in enumerate(examples):
        if tid is None:
            continue
        panel = ex_gs[0, ex_idx].subgridspec(1, 4, wspace=0.12)
        success = _success_grid(tid)
        # Grid success
        ax1 = fig.add_subplot(panel[0, 0])
        _draw_grid(ax1, success, title=f"Task {tid}\nSuccess grid")

        # Chunks visualisation for each of 3 strategies: object, row, col
        for i, (label, fn) in enumerate((
                ("object_first", strategy_object_first),
                ("row_first", strategy_row_first),
                ("col_first", strategy_col_first))):
            ax = fig.add_subplot(panel[0, i + 1])
            _draw_grid(ax, np.zeros_like(success))
            chunks = fn(success)
            cmap = plt.get_cmap("tab20")
            for k, ch in enumerate(chunks):
                face = cmap(k % 20)
                for e in ch:
                    ax.add_patch(Rectangle(
                        (e["x"] - 0.5, e["y"] - 0.5), 1, 1,
                        facecolor=ARC_COLORS_HEX.get(e["color"], "#fff"),
                        edgecolor=face, linewidth=2.2))
            title = f"{label}  ({len(chunks)} chunks)"
            if label == winner:
                title += "  ✓ matches humans"
            ax.set_title(title, fontsize=11, pad=4,
                         color=STRAT_COLORS[label] if label == winner else "#333",
                         fontweight="bold" if label == winner else "normal")
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values(): s.set_visible(False)

    fig.suptitle(
        "Is human chunking task-forced, or does it follow a strategy?\n"
        "Answer: humans are NOT any single canonical strategy — they match different strategies on different tasks.",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = "prior_analysis/strategies_vs_human_figure.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    build()

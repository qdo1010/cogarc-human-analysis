"""
Paper figure: drawing-strategy analysis.

Top row — Component priority
    A. Regression coefficients: which object features predict EARLIER touch?
    B. Aggregate scatter: component area vs mean touch priority, pooled
       across 75 tasks (each point = one Success-grid component).
    C. Exemplar task annotated — each Success component labeled with its
       rank in subjects' touch order.

Bottom row — Chunking
    D. Chunk size distribution pooled across 130 k chunks.
    E. Chunk ↔ Success-component IoU distribution — low values mean
       human chunks do NOT align with the connected-component
       segmentation that a standard ARC graph prior uses.
    F. Example trajectory colored by chunk, so you can see chunk
       boundaries against the grid.

Output:
    prior_analysis/drawing_strategy_figure.png
"""

from __future__ import annotations


import _paths  # noqa: F401
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from human_chunking import (
    _FNAME_RE, _success_components, chunk_features, chunks_for_task,
    identify_chunks,
)
from human_component_priority import _components, _first_touch_ranks
from human_style_features import DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR, _parse_trajectory
from human_targets import human_targets

ARC_COLORS_HEX = {
    0: "#2B2B2B", 1: "#248ADA", 2: "#C71010", 3: "#1FC719",
    4: "#F7DE28", 5: "#878494", 6: "#F954F2", 7: "#EE6000",
    8: "#6B23A9", 9: "#8B5A28",
}


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
    "figure.titlesize": 18,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


OUT = "prior_analysis/drawing_strategy_figure.png"


def _draw_arc_grid(ax, grid, title=""):
    H, W = grid.shape
    for y in range(H):
        for x in range(W):
            c = int(grid[y, x])
            ax.add_patch(Rectangle(
                (x - 0.5, y - 0.5), 1, 1,
                facecolor=ARC_COLORS_HEX.get(c, "#000"),
                edgecolor="#e8e8e8", linewidth=0.4))
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=13, pad=6)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def build_figure():
    reg = pd.read_csv("prior_analysis/component_priority_feature_regression.csv")
    comp = pd.read_csv("prior_analysis/component_priority_per_task.csv")
    chunks = pd.read_csv("prior_analysis/chunks_per_trajectory.csv")

    fig = plt.figure(figsize=(22, 13))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.1],
                          height_ratios=[1.0, 1.0],
                          hspace=0.38, wspace=0.32)

    # --- A: regression coefficients ---
    axA = fig.add_subplot(gs[0, 0])
    reg_sorted = reg.sort_values("beta_within_task")
    colors = ["#3a66a5" if b < 0 else "#b84a3d"
              for b in reg_sorted["beta_within_task"]]
    axA.barh(range(len(reg_sorted)), reg_sorted["beta_within_task"],
             color=colors, edgecolor="#222", linewidth=0.6)
    axA.set_yticks(range(len(reg_sorted)))
    axA.set_yticklabels(reg_sorted["feature"])
    axA.axvline(0, color="#333", lw=0.8)
    axA.set_xlabel("z-scored slope on touch priority\n(← earlier   |   later →)")
    axA.set_title("A.  Which object features predict EARLY touch?\n"
                  "(task-fixed-effect OLS, 1316 components × 75 tasks)",
                  loc="left", pad=8)

    # --- B: area vs touch priority ---
    axB = fig.add_subplot(gs[0, 1])
    sub = comp.dropna(subset=["mean_touch_priority", "area"])
    axB.scatter(sub["area"], sub["mean_touch_priority"],
                s=14, alpha=0.35, color="#3a66a5", edgecolors="none")
    # Trendline: LOESS-ish via rolling median on log-binned area
    bin_edges = np.logspace(np.log10(max(sub["area"].min(), 1)),
                            np.log10(sub["area"].max()), 12)
    mids, med = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        m = (sub["area"] >= lo) & (sub["area"] < hi)
        if m.sum() > 10:
            mids.append(np.sqrt(lo * hi))
            med.append(sub.loc[m, "mean_touch_priority"].median())
    axB.plot(mids, med, color="#b84a3d", lw=2.2, marker="o",
             markersize=7, label="binned median")
    axB.set_xscale("log")
    axB.set_xlabel("Component area (pixels, log)")
    axB.set_ylabel("Mean touch priority\n(0 = first, 1 = last)")
    axB.set_title("B.  Bigger components are touched earlier",
                  loc="left", pad=8)
    axB.legend(frameon=False)

    # --- C: exemplar task with component ranks ---
    axC = fig.add_subplot(gs[0, 2])
    # Pick a task with several components and a clean priority signal.
    subc = comp.groupby("task_id").agg(
        n=("component_id", "count"),
        spread=("mean_touch_priority", lambda s: s.max() - s.min()),
    )
    cand = subc[(subc["n"] >= 4) & (subc["n"] <= 10) & (subc["spread"] > 0.3)]
    exemplar = cand.sort_values("spread", ascending=False).index[0] \
        if not cand.empty else subc.index[0]
    t = human_targets(exemplar)
    success = next(g for lbl, g in zip(t["labels"], t["grids"]) if lbl == "Success")
    _draw_arc_grid(axC, success)
    task_comps = comp[comp.task_id == exemplar].sort_values(
        "mean_touch_priority").reset_index(drop=True)
    # Rank 1 .. K from earliest to latest
    for rank, row in task_comps.iterrows():
        ax_text = axC.text(row["centroid_c"], row["centroid_r"],
                           str(rank + 1), ha="center", va="center",
                           fontsize=17, color="white", fontweight="bold")
        ax_text.set_path_effects([])
    axC.set_title(f"C.  Exemplar task {exemplar}\n"
                  "Numbers = average human first-touch rank "
                  "(1 = drawn first)",
                  loc="left", pad=8)

    # --- D: chunk size distribution ---
    axD = fig.add_subplot(gs[1, 0])
    sizes = chunks["size"].values
    bins = np.logspace(0, np.log10(max(sizes.max(), 2)), 35)
    axD.hist(sizes, bins=bins, color="#3a66a5", edgecolor="#222",
             linewidth=0.3)
    axD.set_xscale("log")
    axD.axvline(np.median(sizes), color="#b84a3d", lw=2,
                label=f"median = {np.median(sizes):.0f} edits")
    axD.set_xlabel("Chunk size (edits, log scale)")
    axD.set_ylabel("Chunks")
    axD.set_title(f"D.  Chunk-size distribution\n"
                  f"(n = {len(sizes):,} chunks across 75 tasks × "
                  f"{chunks['subject_id'].nunique()} subjects)",
                  loc="left", pad=8)
    axD.legend(frameon=False)

    # --- E: Chunk ↔ Success segmentation (CC vs color class) ---
    axE = fig.add_subplot(gs[1, 1])
    n_cc = chunks["n_success_cc_spanned"].values
    # 3-way composition: 0-CC (wrong cells), 1-CC (object-wise), 2+-CC (color-over-CC)
    frac_out = float((n_cc == 0).mean())
    frac_one = float((n_cc == 1).mean())
    frac_multi = float((n_cc >= 2).mean())
    multi_mask = (chunks["n_success_cc_spanned"] >= 2)
    color_homo_in_multi = float(
        (chunks.loc[multi_mask, "color_homogeneity"] >= 0.95).mean()
    )
    cats = [
        ("outside Success", frac_out, "#9a9a9a"),
        ("inside 1 CC", frac_one, "#3a66a5"),
        ("span 2+ CCs,\nsame color", frac_multi * color_homo_in_multi, "#b84a3d"),
        ("span 2+ CCs,\nmixed color", frac_multi * (1 - color_homo_in_multi), "#e0a9a1"),
    ]
    labels, vals, colors = zip(*cats)
    axE.barh(np.arange(len(cats)), vals, color=colors,
             edgecolor="#222", linewidth=0.7)
    axE.set_yticks(np.arange(len(cats)))
    axE.set_yticklabels(labels, fontsize=12)
    axE.invert_yaxis()
    axE.set_xlim(0, max(vals) * 1.30)
    for i, v in enumerate(vals):
        axE.text(v + 0.008, i, f"{v:.0%}", va="center",
                 fontsize=12, color="#222", fontweight="bold")
    axE.set_xlabel("Fraction of all 130 k chunks")
    axE.set_title(
        "E.  Chunks vs connected-component segmentation",
        loc="left", pad=8,
    )

    # --- F: exemplar trajectory colored by chunk ---
    axF = fig.add_subplot(gs[1, 2])
    # Pick a trajectory with many chunks and a non-trivial task
    ex_task = chunks.groupby("task_id").size().sort_values().index[-10]
    subj_candidates = chunks[chunks.task_id == ex_task].groupby("subject_id").size()
    if len(subj_candidates):
        ex_subj = subj_candidates.sort_values().index[len(subj_candidates) // 2]
    else:
        ex_subj = None
    success, comp_masks, _ = _success_components(ex_task)
    # Load this subject's trajectory
    traj_dir = os.path.join(DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR, f"{ex_task}.json")
    traj = None
    for fname in os.listdir(traj_dir):
        m = _FNAME_RE.match(fname)
        if m and m.group(1) == ex_subj:
            traj = _parse_trajectory(os.path.join(traj_dir, fname))["edits"]
            break
    _draw_arc_grid(axF, np.zeros_like(success))
    if traj:
        chs = identify_chunks(traj)
        cmap = plt.get_cmap("tab20")
        for k, ch in enumerate(chs):
            col = cmap(k % 20)
            for e in ch:
                axF.add_patch(Rectangle(
                    (e["x"] - 0.5, e["y"] - 0.5), 1, 1,
                    facecolor=col, edgecolor="black", linewidth=0.8))
        axF.set_title(f"F.  Task {ex_task}, participant {ex_subj}: "
                      f"{len(chs)} chunks (distinct color per chunk)",
                      loc="left", pad=8)

    fig.suptitle(
        "How humans draw their answers: object priority and cognitive chunks",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    fig.savefig(OUT, dpi=260, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build_figure()

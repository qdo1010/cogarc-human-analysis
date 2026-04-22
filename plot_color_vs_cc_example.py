"""
Concrete example of the claim:

    "On tasks whose solution rule is color-indexed, the ordering
     reverses and color becomes dominant."

We contrast two tasks:

  HIGH color-over-CC task:   0d3d703e  (94% of chunks span 2+ CCs)
  LOW  color-over-CC task:   0d87d2a6  (0.2%)

For each: render the task (all train pairs + test input + Success) and
then one representative participant's chunk sequence, where same-
color-homogeneous chunks that span multiple Success connected
components are highlighted in red.

Output: prior_analysis/color_vs_cc_example.png
"""

from __future__ import annotations


import _paths  # noqa: F401

import json
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from human_chunking import _success_components, identify_chunks, chunk_features
from human_style_features import DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR, _parse_trajectory

ARC_COLORS_HEX = {
    0: "#2B2B2B", 1: "#248ADA", 2: "#C71010", 3: "#1FC719",
    4: "#F7DE28", 5: "#878494", 6: "#F954F2", 7: "#EE6000",
    8: "#6B23A9", 9: "#8B5A28",
}
_FNAME_RE = re.compile(r"^subj_([^_]+)_trial_([0-9a-f]+)\.json\.csv$")

HIGH_TASK = "0d3d703e"
LOW_TASK = "0d87d2a6"
OUT = "prior_analysis/color_vs_cc_example.png"


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14, "axes.titlesize": 15, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 12,
    "figure.titlesize": 19,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _draw_arc_grid(ax, grid, label=""):
    if grid.size == 0:
        ax.axis("off"); return
    H, W = grid.shape
    for y in range(H):
        for x in range(W):
            c = int(grid[y, x])
            ax.add_patch(Rectangle(
                (x - 0.5, y - 0.5), 1, 1,
                facecolor=ARC_COLORS_HEX.get(c, "#000"),
                edgecolor="#e8e8e8", linewidth=0.3))
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    if label: ax.set_title(label, fontsize=12, pad=4)


def _render_task(fig, row_gs, tid, max_train_pairs: int = 2):
    """Render up to ``max_train_pairs`` train pairs + the test pair.
    Uses wider spacing so in/out labels don't overlap on small grids.
    """
    task = json.load(open(f"{DEFAULT_DATA_ROOT}/Task JSONs/{tid}.json"))
    train = task["train"][:max_train_pairs]
    pairs = [("Train " + str(i + 1), np.array(p["input"]), np.array(p["output"]))
             for i, p in enumerate(train)]
    pairs.append(("Test", np.array(task["test"][0]["input"]),
                  np.array(task["test"][0]["output"])))
    ncols = len(pairs) * 2
    # Spacing: wider wspace so short labels don't crowd each other on
    # small grids. 0.5 is generous for 3x3 / 20x20 alike.
    sub = row_gs.subgridspec(1, ncols, wspace=0.55)
    for i, (label, inp, out) in enumerate(pairs):
        ax_in = fig.add_subplot(sub[0, 2 * i])
        ax_out = fig.add_subplot(sub[0, 2 * i + 1])
        _draw_arc_grid(ax_in, inp, label=f"{label}\ninput")
        _draw_arc_grid(
            ax_out, out,
            label=f"{label}\n" + ("output" if label != "Test" else "correct"),
        )


def _load_trajectory(tid, subj):
    d = os.path.join(DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR, f"{tid}.json")
    for fname in os.listdir(d):
        m = _FNAME_RE.match(fname)
        if m and m.group(1) == subj:
            return _parse_trajectory(os.path.join(d, fname))["edits"]
    return []


def _pick_illustrative_subject(tid: str, prefer_cross: bool):
    """Pick a subject whose chunks most clearly illustrate the point."""
    _, comp_masks, _ = _success_components(tid)
    d = os.path.join(DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR, f"{tid}.json")
    best = None; best_score = -1
    for fname in os.listdir(d):
        m = _FNAME_RE.match(fname)
        if not m: continue
        edits = _parse_trajectory(os.path.join(d, fname))["edits"]
        if len(edits) < 6 or len(edits) > 90:
            continue
        chs = identify_chunks(edits)
        multi_cc = sum(
            1 for c in chs
            if sum(1 for cm in comp_masks if {(e["y"], e["x"]) for e in c} & cm) >= 2
        )
        score = multi_cc if prefer_cross else (len(chs) - multi_cc)
        if score > best_score and 3 <= len(chs) <= 18:
            best, best_score = (m.group(1), edits, chs), score
    return best


def _draw_trajectory_by_chunk(ax, success, edits, chs, comp_masks, header=""):
    H, W = success.shape
    for y in range(H):
        for x in range(W):
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1,
                                   facecolor="#f6f6f6",
                                   edgecolor="#e8e8e8", linewidth=0.3))
    for i in range(W + 1):
        ax.plot([i - 0.5, i - 0.5], [-0.5, H - 0.5], color="#dddddd", lw=0.3)
    for j in range(H + 1):
        ax.plot([-0.5, W - 0.5], [j - 0.5, j - 0.5], color="#dddddd", lw=0.3)

    cmap = plt.get_cmap("tab20")
    for k, chunk in enumerate(chs):
        cells = {(e["y"], e["x"]) for e in chunk}
        n_cc = sum(1 for cm in comp_masks if cells & cm)
        # Highlight multi-CC single-color chunks with a thick red border
        multi_cross = (n_cc >= 2) and (
            len({e["color"] for e in chunk}) == 1
        )
        # chunk fill color
        chunk_face = cmap(k % 20)
        edge_color = "#b84a3d" if multi_cross else "black"
        edge_width = 2.4 if multi_cross else 0.7
        for e in chunk:
            ax.add_patch(Rectangle(
                (e["x"] - 0.5, e["y"] - 0.5), 1, 1,
                facecolor=ARC_COLORS_HEX.get(int(e["color"]), "#fff"),
                edgecolor=edge_color, linewidth=edge_width))
        # Label chunk number at centroid
        xs = np.array([e["x"] for e in chunk])
        ys = np.array([e["y"] for e in chunk])
        ax.text(xs.mean(), ys.mean(), str(k + 1),
                ha="center", va="center", fontsize=11,
                color="white", fontweight="bold")

    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    if header:
        ax.set_title(header, fontsize=13, pad=6)


def main():
    _, comp_hi, _ = _success_components(HIGH_TASK)
    success_hi = None
    task = json.load(open(f"{DEFAULT_DATA_ROOT}/Task JSONs/{HIGH_TASK}.json"))
    success_hi = np.array(task["test"][0]["output"])

    _, comp_lo, _ = _success_components(LOW_TASK)
    task_lo = json.load(open(f"{DEFAULT_DATA_ROOT}/Task JSONs/{LOW_TASK}.json"))
    success_lo = np.array(task_lo["test"][0]["output"])

    hi_pick = _pick_illustrative_subject(HIGH_TASK, prefer_cross=True)
    lo_pick = _pick_illustrative_subject(LOW_TASK, prefer_cross=False)

    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(
        nrows=4, ncols=1,
        height_ratios=[0.18, 1.0, 0.18, 1.0], hspace=0.45,
    )

    # --- High color-over-CC: 0d3d703e ---
    axA_hdr = fig.add_subplot(gs[0, 0]); axA_hdr.axis("off")
    axA_hdr.text(0, 0.5, f"COLOR-INDEXED task  {HIGH_TASK}  "
                 f"—  94% of chunks span 2+ connected components",
                 fontsize=17, fontweight="bold", color="#b84a3d",
                 va="center", ha="left")
    # split row 2 into left (task panels) and right (trajectory)
    sub_hi = gs[1, 0].subgridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.25)
    _render_task(fig, sub_hi[0, 0], HIGH_TASK)
    ax_traj_hi = fig.add_subplot(sub_hi[0, 1])
    if hi_pick:
        subj, edits, chs = hi_pick
        _draw_trajectory_by_chunk(
            ax_traj_hi, success_hi, edits, chs, comp_hi,
            header=f"Participant {subj} — {len(chs)} chunks  "
                   f"(red border = single color, spans multiple CCs)",
        )

    # --- Low color-over-CC: 0d87d2a6 ---
    axB_hdr = fig.add_subplot(gs[2, 0]); axB_hdr.axis("off")
    axB_hdr.text(0, 0.5, f"OBJECT-INDEXED task  {LOW_TASK}  "
                 f"—  <1% of chunks span 2+ connected components",
                 fontsize=17, fontweight="bold", color="#3a66a5",
                 va="center", ha="left")
    sub_lo = gs[3, 0].subgridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.25)
    _render_task(fig, sub_lo[0, 0], LOW_TASK)
    ax_traj_lo = fig.add_subplot(sub_lo[0, 1])
    if lo_pick:
        subj, edits, chs = lo_pick
        _draw_trajectory_by_chunk(
            ax_traj_lo, success_lo, edits, chs, comp_lo,
            header=f"Participant {subj} — {len(chs)} chunks  "
                   f"(every chunk stays inside one CC)",
        )

    fig.suptitle(
        "Concrete example: when solution rule is color-indexed, chunks jump across "
        "same-color regions; when rule is object-indexed, chunks hug connected components.",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    fig.savefig(OUT, dpi=260, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

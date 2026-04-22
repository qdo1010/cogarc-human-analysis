"""
Paper-quality figure for the sequence / attention analysis.

For four tasks spanning the order-consistency spectrum, render:

    Column 1 — the Success grid (what needs to be drawn)
    Column 2 — first-edit heatmap (where do participants start?)
    Column 3 — attention transition graph (arrows = where people move next)
    Column 4 — color priority (bar chart of mean first-use rank per color)

Output:
    prior_analysis/sequence_figure.png
"""

from __future__ import annotations

import _paths  # noqa: F401  (sys.path bootstrap, safe under both invocations)

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from human_sequence import (
    attention_graph, color_priority, first_edit_heatmap,
)
from human_targets import human_targets
from motor_vs_cognitive import ARC_COLORS_HEX, _draw_arc_grid, FIG_DPI


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 15,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 13,
    "figure.titlesize": 21,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# Which tasks to showcase: one per consistency tier.
def _pick_cases(df: pd.DataFrame) -> List[str]:
    df = df.sort_values("order_consistency", ascending=False).reset_index(drop=True)
    high   = df.iloc[0]["task_id"]                                    # ρ ≈ 1.0
    upper  = df.iloc[len(df) // 4]["task_id"]                         # upper quartile
    lower  = df.iloc[3 * len(df) // 4]["task_id"]                     # lower quartile
    low    = df.sort_values("order_consistency").iloc[0]["task_id"]   # ρ near 0
    return [high, upper, lower, low]


def _get_success(task_id):
    t = human_targets(task_id)
    for lbl, g in zip(t["labels"], t["grids"]):
        if lbl == "Success":
            return g
    return np.zeros((10, 10), dtype=np.int64)


def _draw_success(ax, tid):
    g = _get_success(tid)
    _draw_arc_grid(ax, g, label="Success (correct answer)", label_fontsize=14)


def _draw_heatmap(ax, heat, success_shape):
    H, W = success_shape
    cmap = LinearSegmentedColormap.from_list(
        "heat", ["#f7f7f7", "#fde7dc", "#f7a67d", "#dd4b3e", "#7a0d0d"]
    )
    im = ax.imshow(heat, cmap=cmap, vmin=0,
                   vmax=max(heat.max(), 0.05),
                   interpolation="nearest",
                   extent=(-0.5, W - 0.5, H - 0.5, -0.5))
    for i in range(W + 1):
        ax.plot([i - 0.5, i - 0.5], [-0.5, H - 0.5], color="#dddddd", lw=0.5)
    for j in range(H + 1):
        ax.plot([-0.5, W - 0.5], [j - 0.5, j - 0.5], color="#dddddd", lw=0.5)
    top_k = min(3, int((heat > 0).sum()))
    flat = np.argsort(heat.ravel())[::-1][:top_k]
    for f in flat:
        y, x = np.unravel_index(f, heat.shape)
        p = heat[y, x]
        if p >= 0.03:
            ax.text(x, y, f"{p:.0%}", ha="center", va="center",
                    fontsize=11, color="black", fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("First-edit heatmap\n(fraction of participants)",
                 fontsize=14, pad=6)
    return im


def _draw_transitions(ax, edges: np.ndarray, visits: np.ndarray,
                      keep_frac: float = 0.15):
    """Render top transition edges as arrows; background = per-cell visit count."""
    H, W = visits.shape
    v = np.log1p(visits.astype(float))
    ax.imshow(v, cmap="Greys", alpha=0.7,
              vmin=0, vmax=max(v.max(), 1e-6),
              interpolation="nearest",
              extent=(-0.5, W - 0.5, H - 0.5, -0.5))
    for i in range(W + 1):
        ax.plot([i - 0.5, i - 0.5], [-0.5, H - 0.5], color="#cccccc", lw=0.4)
    for j in range(H + 1):
        ax.plot([-0.5, W - 0.5], [j - 0.5, j - 0.5], color="#cccccc", lw=0.4)

    if edges.size:
        order = np.argsort(edges[:, 4])[::-1]
        k = max(1, int(len(order) * keep_frac))
        top = edges[order[:k]]
        max_w = top[:, 4].max() + 1e-9
        for fx, fy, tx, ty, w in top:
            alpha = 0.35 + 0.65 * (w / max_w)
            lw = 0.9 + 3.0 * (w / max_w)
            ax.annotate(
                "", xy=(tx, ty), xytext=(fx, fy),
                arrowprops=dict(arrowstyle="->", color="#1f3b73",
                                alpha=alpha, lw=lw,
                                shrinkA=2, shrinkB=2),
            )

    ax.set_aspect("equal")
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Attention transition graph\n(top 15% edit-to-edit edges)",
                 fontsize=14, pad=6)


def _draw_color_priority(ax, task_id):
    cp = color_priority(task_id)
    cols = cp["colors"]
    if not cols:
        ax.set_axis_off(); return
    # Only colors used by >=20% of subjects so the plot isn't cluttered.
    cols = [c for c in cols if c["frac_subjects_used"] >= 0.2]
    if not cols:
        cols = cp["colors"][:6]

    ranks = [c["mean_first_rank"] for c in cols]
    labels = [str(c["color"]) for c in cols]
    face = [ARC_COLORS_HEX.get(c["color"], "#000") for c in cols]
    fracs = [c["frac_subjects_used"] for c in cols]
    y = np.arange(len(cols))
    ax.barh(y, ranks, color=face, edgecolor="#333", linewidth=0.8)
    for i, (r, f) in enumerate(zip(ranks, fracs)):
        ax.text(r + 0.2, i, f"{r:.1f}  ({f:.0%})", va="center",
                fontsize=11, color="#333")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=14, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("Mean 1-based rank at first use")
    ax.set_title("Color priority\n(color · mean first-use rank · frac subjects)",
                 fontsize=14, pad=6)
    ax.set_xlim(left=0)
    # Hide the y-axis line and the top spine
    ax.spines["left"].set_visible(False)


def build_sequence_figure(out_path: str = "prior_analysis/sequence_figure.png"):
    df = pd.read_csv("prior_analysis/sequence_features.csv")
    cases = _pick_cases(df)

    # Per case: one narrow header row + one taller panel row. 2 rows * N cases.
    fig = plt.figure(figsize=(22, 6.0 * len(cases)))
    gs = fig.add_gridspec(
        nrows=len(cases) * 2, ncols=4,
        height_ratios=[0.15, 1.0] * len(cases),
        width_ratios=[1.0, 1.0, 1.0, 1.25],
        hspace=0.55, wspace=0.40,
    )

    for r, tid in enumerate(cases):
        row = df[df.task_id == tid].iloc[0]
        header = (
            f"Task {tid}     "
            f"order consistency ρ = {row.order_consistency:.2f}     "
            f"peak first-edit cell probability = {row.max_first_edit_prob:.0%}     "
            f"n_subjects = {int(row.n_subjects)}"
        )
        ax_hdr = fig.add_subplot(gs[2 * r, :])
        ax_hdr.set_axis_off()
        ax_hdr.text(0.0, 0.5, header, transform=ax_hdr.transAxes,
                    fontsize=19, fontweight="bold", ha="left", va="center",
                    color="#2c3e50")

        success = _get_success(tid)
        heat = first_edit_heatmap(tid)
        ag = attention_graph(tid)

        axA = fig.add_subplot(gs[2 * r + 1, 0])
        _draw_arc_grid(axA, success, label="Success grid", label_fontsize=14)
        axB = fig.add_subplot(gs[2 * r + 1, 1])
        _draw_heatmap(axB, heat, success.shape)
        axC = fig.add_subplot(gs[2 * r + 1, 2])
        _draw_transitions(axC, ag["edges"], ag["node_visits"])
        axD = fig.add_subplot(gs[2 * r + 1, 3])
        _draw_color_priority(axD, tid)

    fig.suptitle(
        "Sequence analysis across the order-consistency spectrum\n"
        "For each task: the correct answer, where participants start (first-edit heatmap), "
        "how edits flow between cells (attention transitions), and the order in which colors are first used.",
        y=0.995, fontsize=19,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    build_sequence_figure()

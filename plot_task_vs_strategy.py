"""
Figure: what about a TASK predicts which strategy humans use?

Panels
    A  Affordance histogram — how many of the 6 canonical strategies
       produce >=2 chunks on each task's Success grid. Answers H1.
    B  Distribution of per-task human convergence: entropy (bits) and
       fraction of subjects choosing the modal strategy.
    C  Heatmap: Spearman ρ of task features (rows) vs per-task SUBJECT
       USAGE share of each canonical strategy (columns).
    D  Three scatter plots highlighting the strongest predictors:
       n_success_cc → object_first, n_colors_output → color_first,
       mean_cc_area → random_k3 (smaller outputs look less organised).
    E  Two concrete task examples — one where subjects converge, one
       where they spread — each showing the Success grid plus the
       strategy-usage bar.

Output: prior_analysis/task_vs_strategy_figure.png
"""

from __future__ import annotations


import _paths  # noqa: F401

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

from human_vs_strategies import STRATEGIES, _success_grid


ARC_COLORS_HEX = {
    0: "#2B2B2B", 1: "#248ADA", 2: "#C71010", 3: "#1FC719",
    4: "#F7DE28", 5: "#878494", 6: "#F954F2", 7: "#EE6000",
    8: "#6B23A9", 9: "#8B5A28",
}

STRAT_COLORS = {
    "object_first":   "#1f77b4",
    "color_first":    "#2ca02c",
    "nn_color_first": "#9467bd",
    "row_first":      "#ff7f0e",
    "col_first":      "#8c564b",
    "random_k3":      "#aaaaaa",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 12,
    "figure.titlesize": 20,
})


def _draw_grid(ax, grid, title=""):
    H, W = grid.shape
    for y in range(H):
        for x in range(W):
            ax.add_patch(Rectangle(
                (x - 0.5, y - 0.5), 1, 1,
                facecolor=ARC_COLORS_HEX.get(int(grid[y, x]), "#000"),
                edgecolor="#e8e8e8", linewidth=0.3))
    ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, pad=4)


def build():
    tfeat = pd.read_csv("prior_analysis/task_features_for_strategy.csv")
    subj = pd.read_csv("prior_analysis/subject_best_strategy.csv")

    usage = (subj.groupby(["task_id", "best_strategy"]).size()
             .unstack(fill_value=0))
    usage = usage.div(usage.sum(axis=1), axis=0)
    for s in STRATEGIES.keys():
        if s not in usage.columns:
            usage[s] = 0.0
    usage = usage[list(STRATEGIES.keys())].reset_index()

    merged = tfeat.merge(
        usage.rename(columns={s: f"usage_{s}" for s in STRATEGIES.keys()}),
        on="task_id",
    )

    fig = plt.figure(figsize=(24, 18))
    gs = fig.add_gridspec(
        3, 3,
        height_ratios=[1.0, 1.0, 1.1],
        width_ratios=[1.0, 1.0, 1.2],
        hspace=0.55, wspace=0.40,
    )

    # ---- A Affordance histogram ----
    axA = fig.add_subplot(gs[0, 0])
    counts = merged["n_afforded"].value_counts().sort_index()
    bars = axA.bar(counts.index.astype(str), counts.values,
                   color="#4c72b0", edgecolor="#222", linewidth=0.8)
    for b, v in zip(bars, counts.values):
        axA.text(b.get_x() + b.get_width() / 2, v + 1, str(v),
                 ha="center", fontsize=13, fontweight="bold")
    axA.set_xlabel("# canonical strategies that produce ≥2 chunks")
    axA.set_ylabel("# tasks (of 75)")
    axA.set_title("A.  How many strategies does each task afford?",
                  loc="left", pad=8)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)
    axA.set_ylim(0, max(counts.values) * 1.15)
    axA.text(0.02, 0.90,
             "67 of 75 tasks afford all 6:\n"
             "affordance alone doesn't\nconstrain strategy choice.",
             transform=axA.transAxes, fontsize=11,
             va="top", ha="left",
             bbox=dict(facecolor="#f3f3f3", edgecolor="#aaa", boxstyle="round"))

    # ---- B Human convergence distribution ----
    axB = fig.add_subplot(gs[0, 1])
    axB.hist(merged["dominant_frac"], bins=18, color="#dd8452",
             edgecolor="#222", linewidth=0.6)
    axB.axvline(merged["dominant_frac"].mean(), color="#222", ls="--", lw=1.5,
                label=f"mean = {merged['dominant_frac'].mean():.2f}")
    axB.set_xlabel("dominant-strategy fraction per task")
    axB.set_ylabel("# tasks")
    axB.set_title("B.  Do subjects converge on one strategy?",
                  loc="left", pad=8)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)
    axB.legend(loc="upper left", frameon=False)

    # ---- C Heatmap: task features × strategy usage ----
    axC = fig.add_subplot(gs[0, 2])
    feat_rows = [
        "n_success_cc", "n_colors_output", "n_colors_input", "n_new_colors",
        "mean_cc_area", "max_cc_area", "n_cells_nonbg", "frac_nonbg",
        "grid_h", "grid_w", "symmetry_h", "symmetry_v",
        "n_reset_mean", "n_showex_mean", "used_copy_mean", "n_edits_mean",
    ]
    strat_cols = ["object_first", "color_first", "nn_color_first",
                  "row_first", "col_first", "random_k3"]
    M = np.zeros((len(feat_rows), len(strat_cols)))
    P = np.zeros_like(M)
    for i, f in enumerate(feat_rows):
        for j, s in enumerate(strat_cols):
            col = f"usage_{s}"
            sub = merged[[f, col]].dropna()
            if len(sub) < 10 or sub[f].nunique() < 2 or sub[col].nunique() < 2:
                M[i, j] = np.nan
                P[i, j] = 1.0
                continue
            rho, pval = spearmanr(sub[f], sub[col])
            M[i, j] = rho; P[i, j] = pval
    im = axC.imshow(M, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    axC.set_xticks(range(len(strat_cols)))
    axC.set_xticklabels([s.replace("_", "\n") for s in strat_cols],
                        fontsize=10)
    axC.set_yticks(range(len(feat_rows)))
    axC.set_yticklabels(feat_rows, fontsize=10)
    for i in range(len(feat_rows)):
        for j in range(len(strat_cols)):
            if np.isnan(M[i, j]):
                continue
            star = "*" if P[i, j] < 0.05 else ""
            axC.text(j, i, f"{M[i, j]:+.2f}{star}",
                     ha="center", va="center", fontsize=9,
                     color="white" if abs(M[i, j]) > 0.35 else "#222")
    axC.set_title(
        "C.  Spearman ρ: task features × subject-usage share of each strategy\n"
        "(* p<0.05 uncorrected)",
        loc="left", pad=8,
    )
    cbar = fig.colorbar(im, ax=axC, fraction=0.035, pad=0.02)
    cbar.set_label("Spearman ρ", fontsize=11)

    # ---- D Three scatter plots of strongest predictors ----
    scatters = [
        ("n_success_cc", "object_first",
         "More distinct objects → more subjects draw object-by-object"),
        ("n_colors_output", "color_first",
         "More output colors → more subjects draw color-by-color"),
        ("mean_cc_area", "random_k3",
         "Smaller objects → chunks look unstructured (random-like)"),
    ]
    for i, (x, s, subtitle) in enumerate(scatters):
        ax = fig.add_subplot(gs[1, i])
        xv = merged[x].values
        yv = merged[f"usage_{s}"].values
        ax.scatter(xv, yv, color=STRAT_COLORS[s], edgecolor="#222",
                   linewidth=0.4, alpha=0.75, s=55)
        # trend line (rank-based, so fit on ranks and invert)
        from scipy.stats import rankdata
        rx, ry = rankdata(xv), rankdata(yv)
        b, a = np.polyfit(rx, ry, 1)
        # Draw a straight line in rank space, then map back roughly by
        # picking 10 quantiles of x so the visual overlays the data.
        order = np.argsort(xv)
        qx = np.linspace(0, len(xv) - 1, 50)
        # use a simple linear regression on the raw values for the overlay
        rho, pval = spearmanr(xv, yv)
        bx = np.polyfit(xv, yv, 1)
        xs = np.linspace(xv.min(), xv.max(), 50)
        ax.plot(xs, np.polyval(bx, xs), color=STRAT_COLORS[s],
                lw=2.0, alpha=0.9)
        ax.set_xlabel(x)
        ax.set_ylabel(f"fraction of subjects\nclosest to {s}")
        ax.set_title(f"D{i+1}.  ρ = {rho:+.2f}  (p = {pval:.1e})",
                     loc="left", pad=6)
        ax.text(0.02, 0.97, subtitle,
                transform=ax.transAxes, fontsize=10,
                va="top", ha="left",
                bbox=dict(facecolor="white", edgecolor="#aaa",
                          boxstyle="round", alpha=0.85))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(-0.02, max(1.0, yv.max() * 1.1))

    # ---- E Two concrete task examples ----
    # Pick one task with low entropy (strong convergence) and one with
    # high entropy (subjects spread across strategies). Prefer examples
    # on non-tiny grids so the reader can actually see the structure.
    candidates_low = merged[merged["grid_h"] * merged["grid_w"] >= 25]
    low = candidates_low.nsmallest(1, "human_entropy_bits").iloc[0]
    high = merged.nlargest(1, "human_entropy_bits").iloc[0]

    outer = gs[2, :]
    ex_gs = outer.subgridspec(1, 2, wspace=0.22)

    for ex_idx, (label, row) in enumerate([
            ("Low entropy — subjects agree", low),
            ("High entropy — subjects disagree", high)]):

        inner = ex_gs[0, ex_idx].subgridspec(1, 2, wspace=0.35,
                                             width_ratios=[1, 1.3])
        tid = row["task_id"]

        # Header — two lines so long strings don't collide with the
        # neighbouring example.
        prefix = "E1." if ex_idx == 0 else "E2."
        ax_hdr = fig.add_subplot(ex_gs[0, ex_idx]); ax_hdr.set_axis_off()
        ax_hdr.text(
            0.0, 1.06,
            f"{prefix}  {label}",
            transform=ax_hdr.transAxes,
            fontsize=14, fontweight="bold",
            ha="left", va="bottom",
        )
        ax_hdr.text(
            0.0, 1.01,
            f"Task {tid}   ·   entropy {row['human_entropy_bits']:.2f} bits"
            f"   ·   dominant: {row['dominant_strategy']} "
            f"({row['dominant_frac']*100:.0f}%)",
            transform=ax_hdr.transAxes,
            fontsize=11,
            ha="left", va="bottom",
        )

        # Success grid
        ax1 = fig.add_subplot(inner[0, 0])
        _draw_grid(ax1, _success_grid(tid), title="Success")

        # Strategy usage bar
        ax2 = fig.add_subplot(inner[0, 1])
        strat_order = list(STRATEGIES.keys())
        vals = [row.get(f"usage_{s}", 0.0) for s in strat_order]
        bars = ax2.barh(range(len(strat_order)), vals,
                        color=[STRAT_COLORS[s] for s in strat_order],
                        edgecolor="#222", linewidth=0.6)
        ax2.set_yticks(range(len(strat_order)))
        ax2.set_yticklabels(strat_order, fontsize=11)
        ax2.invert_yaxis()
        ax2.set_xlim(0, 1)
        ax2.set_xlabel("fraction of subjects")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        for b, v in zip(bars, vals):
            ax2.text(v + 0.01, b.get_y() + b.get_height() / 2,
                     f"{v*100:.0f}%", va="center", fontsize=10)
        ax2.set_title("subject strategy usage", fontsize=12, pad=5)

    fig.suptitle(
        "What about a TASK predicts which strategy humans use?\n"
        "Affordance alone is uninformative (almost every task affords all 6), "
        "but task-intrinsic features (# objects, # colors, object size) "
        "predict per-strategy usage.",
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = "prior_analysis/task_vs_strategy_figure.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    build()

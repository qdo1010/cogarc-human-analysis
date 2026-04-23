"""
Figure: individual differences in chunking behavior.

Panel A: split-half reliability per chunk feature (sorted, with 95% CIs).
         Shows which dimensions are stable individual-difference traits.

Panel B: lumper vs splitter scatter — mean chunk size (per subject)
         against overall error rate. Subjects who consistently paint
         bigger chunks make fewer errors.

Panel C: style × performance correlation heatmap, restricted to the
         reliable style dimensions (ρ_reliability > 0.2) and the headline
         performance metrics.
"""

from __future__ import annotations

import _paths  # noqa: F401
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


RELIABLE_FEATURES = ["size", "n_cells", "n_chunks_total",
                     "is_connected", "fill_ratio"]
PERF_METRICS = ["overall_error_rate", "te1_rate", "te2_rate", "te3_rate",
                "cognitive_rate", "motor_rate"]
PERF_LABELS = {
    "overall_error_rate": "any-error\nrate",
    "te1_rate": "TE1\nrate",
    "te2_rate": "TE2\nrate",
    "te3_rate": "TE3\nrate",
    "cognitive_rate": "cognitive\nper edit",
    "motor_rate": "motor\nper edit",
}
FEAT_LABELS = {
    "size": "chunk size\n(edits)",
    "n_cells": "n_cells per\nchunk",
    "n_chunks_total": "n_chunks per\ntrajectory",
    "is_connected": "frac of chunks\n4-connected",
    "fill_ratio": "fill ratio\n(compactness)",
    "bbox_area": "bbox area",
    "nn_chain_rate": "draw-order\nadjacency",
    "color_homogeneity": "color\nhomogeneity",
    "success_iou_best": "success-IoU\nper chunk",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_dir", default="prior_analysis")
    ap.add_argument("--out",
                    default="prior_analysis/individual_differences_figure.png")
    args = ap.parse_args()

    rel = pd.read_csv(os.path.join(args.prior_dir,
                                   "individual_differences_reliability.csv"))
    corr = pd.read_csv(os.path.join(args.prior_dir,
                                    "individual_differences_correlations.csv"))
    merged = pd.read_csv(os.path.join(args.prior_dir,
                                      "individual_differences_merged.csv"))

    rel = rel.sort_values("mean_rho", ascending=True).reset_index(drop=True)

    fig = plt.figure(figsize=(18, 7.5))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.4, 1.0, 1.3],
                  hspace=0.3, wspace=0.35,
                  top=0.88, bottom=0.14, left=0.05, right=0.98)

    # ---- Panel A ----
    axA = fig.add_subplot(gs[0, 0])
    ys = np.arange(len(rel))
    colors = ["#4a9960" if r > 0.3 else ("#f1b04c" if r > 0 else "#a6a6a6")
              for r in rel["mean_rho"]]
    err_lo = rel["mean_rho"] - rel["ci95_lo"]
    err_hi = rel["ci95_hi"] - rel["mean_rho"]
    axA.barh(ys, rel["mean_rho"], xerr=[err_lo, err_hi],
             color=colors, edgecolor="black", alpha=0.9, capsize=3)
    axA.axvline(0, color="grey", lw=0.5)
    axA.axvline(0.3, color="#b84a3d", ls="--", lw=1,
                label="trait threshold (ρ = 0.3)")
    axA.set_yticks(ys)
    axA.set_yticklabels([FEAT_LABELS.get(f, f).replace("\n", " ")
                         for f in rel["feature"]], fontsize=9)
    axA.set_xlabel("split-half Spearman ρ across subjects")
    axA.set_title("A. Reliability of chunking dimensions as individual traits",
                  loc="left", fontsize=12)
    axA.legend(loc="lower right", fontsize=9)
    axA.grid(axis="x", alpha=0.25)

    # ---- Panel B: lumper vs splitter scatter ----
    axB = fig.add_subplot(gs[0, 1])
    from scipy.stats import spearmanr
    pair = merged[["size", "overall_error_rate"]].dropna()
    x = pair["size"].values
    y = pair["overall_error_rate"].values
    axB.scatter(x, y, s=30, alpha=0.7, color="#2d6cdf",
                edgecolors="black", linewidths=0.4)
    rho, p = spearmanr(x, y)
    coef = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    p_str = f"p = {p:.1e}" if p >= 1e-3 else f"p < 1e-3"
    axB.plot(xs, np.polyval(coef, xs), color="#b84a3d", lw=2,
             label=f"Spearman ρ = {rho:.2f}  ({p_str}, n = {len(x)})")
    axB.set_xlabel("mean chunk size (cells per chunk)")
    axB.set_ylabel("overall error rate")
    axB.set_title("B. Lumpers outperform splitters",
                  loc="left", fontsize=12)
    axB.legend(loc="upper right", fontsize=9)
    axB.grid(alpha=0.25)

    # ---- Panel C: heatmap style × performance (reliable dims only) ----
    axC = fig.add_subplot(gs[0, 2])
    sub = corr[corr["style"].isin(RELIABLE_FEATURES)]
    mat = sub.pivot(index="style", columns="performance",
                    values="rho").reindex(index=RELIABLE_FEATURES,
                                          columns=PERF_METRICS)
    im = axC.imshow(mat.values, cmap="RdBu_r", vmin=-0.25, vmax=0.25,
                    aspect="auto")
    axC.set_xticks(range(len(PERF_METRICS)))
    axC.set_xticklabels([PERF_LABELS[m] for m in PERF_METRICS], fontsize=8.5)
    axC.set_yticks(range(len(RELIABLE_FEATURES)))
    axC.set_yticklabels([FEAT_LABELS[f] for f in RELIABLE_FEATURES],
                        fontsize=8.5)
    for i in range(len(RELIABLE_FEATURES)):
        for j in range(len(PERF_METRICS)):
            v = mat.values[i, j]
            axC.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=8.5,
                     color="white" if abs(v) > 0.15 else "black")
    axC.set_title("C. Chunking style × performance (Spearman ρ)",
                  loc="left", fontsize=12)
    cbar = fig.colorbar(im, ax=axC, shrink=0.8, pad=0.02)
    cbar.ax.set_ylabel("Spearman ρ", fontsize=9)

    fig.suptitle(
        f"Individual differences in chunking behavior "
        f"(n = {len(merged)} subjects × 75 tasks, 130k chunks)",
        fontsize=13, y=0.96,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

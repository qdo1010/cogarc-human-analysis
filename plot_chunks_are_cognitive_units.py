"""
Figure: chunks are cognitive units, not analytical artifacts.

Panel A: inter-edit RT distribution on log scale with intra-chunk (fast,
         motor) and inter-chunk (slow, planning) distributions overlaid —
         makes the bimodality quantitative.

Panels B/C/D: real vs null-RT vs null-Cut on three structural coherence
              metrics (color_homogeneity, frac_connected, success_iou_best).

Panel E: cross-subject chunking agreement (Adjusted Rand Index across
         pairs of subjects on the same task). Real vs null-Cut.
"""

from __future__ import annotations

import _paths  # noqa: F401
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CONDS = ["real", "null_rt", "null_cut"]
COND_LABELS = {"real": "real\npause-segmented",
               "null_rt": "null-RT\n(shuffle RTs)",
               "null_cut": "null-Cut\n(random cuts)"}
COND_COLORS = {"real": "#b84a3d",
               "null_rt": "#8a7bc4",
               "null_cut": "#a6a6a6"}


def _cond_stats(df, metric, conds=None):
    conds = conds or CONDS
    out = {}
    for cond in conds:
        x = df.loc[df["condition"] == cond, metric].values
        if x.dtype == bool:
            x = x.astype(float)
        if x.dtype.kind == "f":
            x = x[~np.isnan(x)]
        if len(x) == 0:
            out[cond] = (np.nan, np.nan, np.nan)
            continue
        mean = float(np.mean(x))
        rng = np.random.default_rng(0)
        boots = [float(rng.choice(x, size=len(x), replace=True).mean())
                 for _ in range(200)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out[cond] = (mean, mean - lo, hi - mean)
    return out


def _bar_panel(ax, df, metric, title, ylabel, yrange=None, conds=None):
    conds = conds or CONDS
    stats = _cond_stats(df, metric, conds)
    xs = np.arange(len(conds))
    means = [stats[c][0] for c in conds]
    err_low = [stats[c][1] for c in conds]
    err_high = [stats[c][2] for c in conds]
    bars = ax.bar(
        xs, means, color=[COND_COLORS[c] for c in conds],
        edgecolor="black", yerr=[err_low, err_high],
        capsize=5, alpha=0.9,
    )
    if yrange and yrange[1] is not None:
        _span = yrange[1] - yrange[0]
    else:
        _span = max(means) if means else 1.0
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + 0.008 * _span,
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([COND_LABELS[c] for c in conds], fontsize=8.5)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11.5)
    if yrange:
        ax.set_ylim(*yrange)
    ax.grid(axis="y", alpha=0.25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--by_chunk_csv",
                    default="prior_analysis/chunks_cognitive_units_by_chunk.csv")
    ap.add_argument("--rts_npy",
                    default="prior_analysis/chunks_cognitive_units_rts_tagged.npy")
    ap.add_argument("--ari_csv",
                    default="prior_analysis/chunks_cognitive_units_ari.csv")
    ap.add_argument("--out",
                    default="prior_analysis/chunks_cognitive_units_figure.png")
    args = ap.parse_args()

    df = pd.read_csv(args.by_chunk_csv)
    rts = np.load(args.rts_npy)  # structured: rt, is_inter
    rts = rts[rts["rt"] > 0]
    intra_rts = rts["rt"][~rts["is_inter"]]
    inter_rts = rts["rt"][rts["is_inter"]]
    ari_df = pd.read_csv(args.ari_csv)

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1.0],
                          hspace=0.45, wspace=0.32,
                          top=0.90, bottom=0.09, left=0.05, right=0.98)

    # ----- Panel A: RT histogram with intra/inter overlay -----
    axA = fig.add_subplot(gs[0, :])
    lo = max(rts["rt"].min(), 1.0)
    hi = float(rts["rt"].max() + 1)
    bins = np.logspace(np.log10(lo), np.log10(hi), 80)
    axA.hist(intra_rts, bins=bins, color="#4a9960", alpha=0.75,
             label=f"intra-chunk (n = {len(intra_rts):,}, "
                   f"median = {np.median(intra_rts):.0f} ms)")
    axA.hist(inter_rts, bins=bins, color="#b84a3d", alpha=0.75,
             label=f"inter-chunk (n = {len(inter_rts):,}, "
                   f"median = {np.median(inter_rts):.0f} ms)")
    axA.set_xscale("log")
    axA.set_xlabel("inter-edit reaction time (ms, log scale)")
    axA.set_ylabel("count")
    axA.axvline(500, color="grey", ls=":", lw=1.2,
                label="500 ms (algorithm floor)")
    med_all = float(np.median(rts["rt"]))
    axA.axvline(med_all, color="#2d6cdf", ls="--", lw=1.3,
                label=f"global median = {med_all:.0f} ms")
    axA.set_title(
        f"A. Inter-edit RT distribution (n = {len(rts):,} gaps) — "
        "intra-chunk edits and inter-chunk pauses separate cleanly",
        loc="left", fontsize=12,
    )
    axA.legend(loc="upper right", fontsize=9)
    axA.grid(alpha=0.25)

    # ----- Panels B / C / D -----
    axB = fig.add_subplot(gs[1, 0])
    _bar_panel(axB, df, "color_homogeneity",
               "B. Color homogeneity within chunk",
               "mean fraction (dominant / total)",
               yrange=(0.88, 1.01))

    axC = fig.add_subplot(gs[1, 1])
    _bar_panel(axC, df, "is_connected",
               "C. Chunk is one 4-connected component",
               "fraction of chunks",
               yrange=(0.0, 1.0))

    axD = fig.add_subplot(gs[1, 2])
    _bar_panel(axD, df, "success_iou_best",
               "D. IoU with best-matching Success component",
               "mean IoU",
               yrange=(0.0, None))

    # ----- Panel E: ARI -----
    axE = fig.add_subplot(gs[1, 3])
    _bar_panel(axE, ari_df, "ari",
               "E. Cross-subject ARI (chunk partitions)",
               "mean Adjusted Rand Index",
               yrange=(0.0, None),
               conds=["real", "null_cut"])

    # Summary for suptitle
    summary = (df.groupby("condition").agg(
        homog=("color_homogeneity", "mean"),
        conn=("is_connected", "mean"),
        iou=("success_iou_best", "mean"),
    ))
    ari_means = ari_df.groupby("condition")["ari"].mean()
    gap_conn = summary.loc["real", "conn"] - max(
        summary.loc["null_rt", "conn"], summary.loc["null_cut", "conn"])
    gap_iou = summary.loc["real", "iou"] - max(
        summary.loc["null_rt", "iou"], summary.loc["null_cut", "iou"])
    gap_ari = ari_means.loc["real"] - ari_means.loc["null_cut"]

    fig.suptitle(
        "Pause-segmented chunks are cognitive units: evidence from 75 tasks × Experiment 2  "
        f"(Δconnectedness = +{gap_conn:.2f},  "
        f"ΔSuccess-IoU = +{gap_iou:.3f},  "
        f"ΔARI = +{gap_ari:.2f})",
        fontsize=13, y=0.975,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

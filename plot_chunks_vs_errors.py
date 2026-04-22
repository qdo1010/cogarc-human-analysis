"""
Figure: can chunks tell us which common error a person will make,
and is it a personal trait or a task-forced bias?

Three panels:
    A  Early-vs-late prediction accuracy on the balanced subset
       (Top Error 2 vs Top Error 3, diagnostic-pool >= 40). First-
       quarter chunks predict the final TE at 74% vs 56% baseline.
    B  Split-half Spearman reliability of subject-level error
       behaviour (226 subjects, 50 random splits). error_rate is a
       stable personal trait; which specific TE is not.
    C  Variance decomposition (eta^2) of trial outcomes by task
       identity vs subject identity. Task >> subject for every
       outcome, especially TE1.

Output: prior_analysis/chunks_vs_errors_figure.png
"""

from __future__ import annotations


import _paths  # noqa: F401

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 20,
})


C_PRED = "#2d6cdf"          # color-aware weighted prediction
C_BASE = "#bbbbbb"          # majority baseline
C_TASK = "#8B5A28"          # task-level eta2 (warm brown)
C_SUBJ = "#4a9960"          # subject-level eta2 (muted green)
C_ERR  = "#b84a3d"          # error stability bars
C_CORR = "#888888"          # non-error-rate bars


def _compute_early_late():
    df = pd.read_csv("prior_analysis/chunk_vs_error_per_trajectory.csv")
    bal = df[(df.diag_pool_te_total >= 40)
             & (df.final_submission_label.isin(
                 ["Top Error 2", "Top Error 3"]))].copy()
    rows = []
    for name, col in [
            ("first\nquarter",  "diag_col_wq_label"),
            ("first\nhalf",     "diag_col_we_label"),
            ("all\nchunks",     "diag_col_w_label"),
            ("last\nhalf",      "diag_col_wl_label"),
    ]:
        sub = bal[bal[col].isin(["Top Error 2", "Top Error 3"])]
        if len(sub) < 15:
            rows.append({"window": name, "acc": np.nan, "base": np.nan,
                         "n": len(sub)})
            continue
        acc = (sub[col] == sub.final_submission_label).mean()
        base = max((sub.final_submission_label == "Top Error 2").mean(),
                   (sub.final_submission_label == "Top Error 3").mean())
        rows.append({"window": name, "acc": float(acc), "base": float(base),
                     "n": int(len(sub))})
    return pd.DataFrame(rows)


def _compute_subject_reliability():
    df = pd.read_csv("prior_analysis/chunk_vs_error_per_trajectory.csv")
    known = df[df.final_submission_label.isin(
        ["Success", "Top Error 1", "Top Error 2", "Top Error 3"])].copy()
    rng = np.random.default_rng(0)
    out = {k: [] for k in ["error_rate", "te1_rate", "te2_rate", "te3_rate"]}
    for _ in range(50):
        a_rows, b_rows = [], []
        for subj, sub in known.groupby("subject_id"):
            if len(sub) < 6:
                continue
            idx = rng.permutation(len(sub))
            half = len(idx) // 2
            a = sub.iloc[idx[:half]]; b = sub.iloc[idx[half:2*half]]
            a_rows.append({
                "subject_id": subj,
                "error_rate": (a.final_submission_label != "Success").mean(),
                "te1_rate": (a.final_submission_label == "Top Error 1").mean(),
                "te2_rate": (a.final_submission_label == "Top Error 2").mean(),
                "te3_rate": (a.final_submission_label == "Top Error 3").mean(),
            })
            b_rows.append({
                "subject_id": subj,
                "error_rate": (b.final_submission_label != "Success").mean(),
                "te1_rate": (b.final_submission_label == "Top Error 1").mean(),
                "te2_rate": (b.final_submission_label == "Top Error 2").mean(),
                "te3_rate": (b.final_submission_label == "Top Error 3").mean(),
            })
        A = pd.DataFrame(a_rows); B = pd.DataFrame(b_rows)
        m = A.merge(B, on="subject_id", suffixes=("_a", "_b"))
        if len(m) < 20:
            continue
        for k in out:
            r = spearmanr(m[f"{k}_a"], m[f"{k}_b"])[0]
            if np.isfinite(r):
                out[k].append(r)
    # Also keep one "snapshot" scatter for the top panel inset (A/B axes).
    # Use the last split's A and B.
    return {k: float(np.mean(v)) for k, v in out.items()}, m


def _eta2_decomposition():
    df = pd.read_csv("prior_analysis/chunk_vs_error_per_trajectory.csv")
    known = df[df.final_submission_label.isin(
        ["Success", "Top Error 1", "Top Error 2", "Top Error 3"])].copy()

    def eta2(dfv, grouping, outcome):
        grand = dfv[outcome].mean()
        ss_total = ((dfv[outcome] - grand) ** 2).sum()
        ss_between = 0.0
        for _, g in dfv.groupby(grouping):
            ss_between += len(g) * (g[outcome].mean() - grand) ** 2
        return float(ss_between / max(ss_total, 1e-9))

    known["is_error"] = (known.final_submission_label != "Success").astype(int)
    known["is_te1"] = (known.final_submission_label == "Top Error 1").astype(int)
    known["is_te2"] = (known.final_submission_label == "Top Error 2").astype(int)
    known["is_te3"] = (known.final_submission_label == "Top Error 3").astype(int)
    rows = []
    for tag, col in [("any error", "is_error"),
                     ("TE1", "is_te1"),
                     ("TE2", "is_te2"),
                     ("TE3", "is_te3")]:
        rows.append({
            "outcome": tag,
            "by_task": eta2(known, "task_id", col),
            "by_subject": eta2(known, "subject_id", col),
        })
    return pd.DataFrame(rows)


def build():
    el = _compute_early_late()
    rel, m_snap = _compute_subject_reliability()
    eta = _eta2_decomposition()

    fig = plt.figure(figsize=(24, 15))
    gs = fig.add_gridspec(2, 3,
                          height_ratios=[1, 1.05],
                          width_ratios=[1.1, 1.0, 1.1],
                          hspace=0.55, wspace=0.32,
                          top=0.88, bottom=0.08,
                          left=0.05, right=0.98)

    # ---- A early vs late ----
    axA = fig.add_subplot(gs[0, 0])
    x = np.arange(len(el))
    w = 0.38
    bars_acc = axA.bar(x - w/2, el["acc"] * 100, w,
                       color=C_PRED, edgecolor="#222", label="chunk prediction")
    bars_base = axA.bar(x + w/2, el["base"] * 100, w,
                        color=C_BASE, edgecolor="#222", label="majority baseline")
    for i, (a, b, n) in enumerate(zip(el["acc"], el["base"], el["n"])):
        if np.isfinite(a):
            axA.text(i - w/2, a * 100 + 1.5, f"{a*100:.0f}%",
                     ha="center", fontsize=11, fontweight="bold",
                     color=C_PRED)
            axA.text(i + w/2, b * 100 + 1.5, f"{b*100:.0f}%",
                     ha="center", fontsize=11, color="#555")
            axA.text(i, -3, f"n={n}", ha="center", fontsize=10, color="#666")
    axA.set_xticks(x)
    axA.set_xticklabels(el["window"])
    axA.set_ylabel("accuracy (%)")
    axA.set_ylim(0, 100)
    axA.axhline(50, color="#cccccc", ls=":", lw=1)
    axA.set_title(
        "A.  Early chunks predict which TE — better than late chunks.\n"
        "TE2 vs TE3 only, diagnostic pool ≥ 40 (balanced subset).",
        loc="left", pad=8,
    )
    axA.legend(loc="upper right", frameon=False)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)
    # Callout on the first-quarter bar
    axA.annotate(
        "wrong plan is visible\nin the first 25% of chunks",
        xy=(0 - w/2, el["acc"].iloc[0] * 100),
        xytext=(0.45, 88),
        fontsize=11, ha="center",
        arrowprops=dict(arrowstyle="->", color="#333"),
        bbox=dict(facecolor="#fff8dc", edgecolor="#aaa",
                  boxstyle="round,pad=0.3"),
    )

    # ---- B subject-level split-half reliability ----
    axB = fig.add_subplot(gs[0, 1])
    order = ["error_rate", "te1_rate", "te2_rate", "te3_rate"]
    names = ["error\nrate", "TE1\nrate", "TE2\nrate", "TE3\nrate"]
    colors = [C_ERR, C_CORR, C_CORR, C_CORR]
    vals = [rel[k] for k in order]
    bars = axB.bar(range(len(order)), vals, color=colors,
                   edgecolor="#222", linewidth=0.8)
    for b, v in zip(bars, vals):
        axB.text(b.get_x() + b.get_width()/2, v + (0.02 if v >= 0 else -0.05),
                 f"{v:+.2f}", ha="center", fontsize=12, fontweight="bold")
    axB.set_xticks(range(len(order)))
    axB.set_xticklabels(names)
    axB.set_ylabel("split-half Spearman ρ\n(stability across a subject's trials)")
    axB.set_ylim(-0.05, 0.55)
    axB.axhline(0, color="#888", lw=0.8)
    axB.set_title(
        "B.  Error-PRONE is a personal trait.\n"
        "Which SPECIFIC TE is not.   n = 226 subjects, 50 splits.",
        loc="left", pad=8,
    )
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)
    # annotation captions placed well below the zero line so they
    # don't fight the bars
    axB.text(0, -0.02, "(stable)", ha="center", fontsize=10,
             color=C_ERR, fontweight="bold")
    for i in [1, 2, 3]:
        axB.text(i, -0.02, "(near zero)", ha="center", fontsize=9.5,
                 color="#555")
    axB.set_ylim(-0.06, 0.55)

    # ---- C variance decomposition ----
    axC = fig.add_subplot(gs[0, 2])
    x = np.arange(len(eta))
    w = 0.38
    bars_t = axC.bar(x - w/2, eta["by_task"] * 100, w,
                     color=C_TASK, edgecolor="#222", label="by task identity")
    bars_s = axC.bar(x + w/2, eta["by_subject"] * 100, w,
                     color=C_SUBJ, edgecolor="#222", label="by subject identity")
    for i, (t, s) in enumerate(zip(eta["by_task"], eta["by_subject"])):
        axC.text(i - w/2, t * 100 + 0.3, f"{t*100:.1f}%",
                 ha="center", fontsize=11, fontweight="bold", color=C_TASK)
        axC.text(i + w/2, s * 100 + 0.3, f"{s*100:.1f}%",
                 ha="center", fontsize=11, fontweight="bold", color=C_SUBJ)
    axC.set_xticks(x)
    axC.set_xticklabels(eta["outcome"])
    axC.set_ylabel("variance explained (η², %)")
    axC.set_ylim(0, max(eta["by_task"].max(), eta["by_subject"].max()) * 120)
    axC.set_title(
        "C.  Which trial outcome = which TASK, not which SUBJECT.\n"
        "Task explains 4–5× more variance than subject does.",
        loc="left", pad=8,
    )
    axC.legend(loc="upper right", frameon=False)
    axC.spines["top"].set_visible(False)
    axC.spines["right"].set_visible(False)

    # ---- D full discrimination picture (pairwise + pool) ----
    axD = fig.add_subplot(gs[1, :2])
    pw = pd.read_csv("prior_analysis/chunk_error_pairwise.csv")
    pw_ca = pw[pw.scheme == "color-aware (winner)"].copy()
    pw_ca["above_pp"] = 100 * (pw_ca.acc - pw_ca.majority_baseline)
    pairs = pw_ca["pair"].tolist()
    x = np.arange(len(pairs)); w = 0.4
    axD.bar(x - w/2, pw_ca["acc"] * 100, w, color=C_PRED,
            edgecolor="#222", label="chunk prediction")
    axD.bar(x + w/2, pw_ca["majority_baseline"] * 100, w, color=C_BASE,
            edgecolor="#222", label="majority baseline")
    for i, (a, b, n) in enumerate(zip(
            pw_ca["acc"], pw_ca["majority_baseline"], pw_ca["n"])):
        axD.text(i - w/2, a * 100 + 1.5, f"{a*100:.0f}%",
                 ha="center", fontsize=11, fontweight="bold", color=C_PRED)
        axD.text(i + w/2, b * 100 + 1.5, f"{b*100:.0f}%",
                 ha="center", fontsize=11, color="#555")
        axD.text(i, -3, f"n={n}", ha="center", fontsize=10, color="#666")
    axD.set_xticks(x); axD.set_xticklabels(pairs)
    axD.set_ylabel("pairwise accuracy (%)")
    axD.set_ylim(0, 100)
    axD.set_title(
        "D.  Pairwise discrimination among error-ending trajectories.\n"
        "Only on the BALANCED pair (TE2 vs TE3, no majority dominance) "
        "does chunk prediction beat baseline (+7.4pp).",
        loc="left", pad=8,
    )
    axD.legend(loc="upper right", frameon=False)
    axD.spines["top"].set_visible(False)
    axD.spines["right"].set_visible(False)

    # ---- E accuracy × task diagnosability ----
    axE = fig.add_subplot(gs[1, 2])
    db = pd.read_csv("prior_analysis/chunk_error_by_diagnosability.csv")
    x = np.arange(len(db)); w = 0.4
    axE.bar(x - w/2, db["acc_weighted_color"] * 100, w, color=C_PRED,
            edgecolor="#222", label="chunk prediction")
    axE.bar(x + w/2, db["majority_baseline"] * 100, w, color=C_BASE,
            edgecolor="#222", label="majority baseline")
    for i, (a, b, n) in enumerate(zip(
            db["acc_weighted_color"], db["majority_baseline"], db["n"])):
        axE.text(i - w/2, a * 100 + 1.5, f"{a*100:.0f}%",
                 ha="center", fontsize=10, fontweight="bold", color=C_PRED)
        axE.text(i + w/2, b * 100 + 1.5, f"{b*100:.0f}%",
                 ha="center", fontsize=10, color="#555")
        axE.text(i, -3, f"n={n}", ha="center", fontsize=9, color="#666")
    axE.set_xticks(x)
    axE.set_xticklabels(db["pool_range"])
    axE.set_xlabel("# mutually-exclusive (y,x,color) triples\nacross TEs on the task")
    axE.set_ylabel("accuracy (%)")
    axE.set_ylim(0, 100)
    axE.set_title(
        "E.  Tasks with more diagnostic triples\n"
        "give better chunk-based predictions.",
        loc="left", pad=8,
    )
    axE.legend(loc="upper right", frameon=False, fontsize=10)
    axE.spines["top"].set_visible(False)
    axE.spines["right"].set_visible(False)

    fig.suptitle(
        "Can chunks predict which common error a person will make?\n"
        "Yes — but only when class imbalance doesn't swamp the signal.  "
        "Early chunks beat late chunks.  The bias is task-level, not personal.",
        y=0.97, fontsize=18,
    )
    out = "prior_analysis/chunks_vs_errors_figure.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    build()

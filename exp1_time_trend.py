"""
Within-subject time-trend analysis for motor vs cognitive errors.

We re-apply the motor/cognitive classifier from human_error_type.py to
Experiment 1 trajectories, because Exp 1 presented all 75 tasks in a
FIXED order (column 'problem order (experiment 1)' in Task tags.csv)
whereas Exp 2 randomized per subject without exposing the order.

For each Exp 1 trajectory we get:
    (subject_id, task_id, trial_index 1..75, motor_rate, cognitive_rate, n_wrong)

Then we ask:
    Does motor_rate increase over trial_index?   (fatigue/sloppiness?)
    Does cognitive_rate decrease over trial_index? (learning/practice?)
    Or both? Or neither?

Outputs:
    prior_analysis/exp1_trial_trends.csv
    prior_analysis/exp1_time_trend_figure.png
"""

from __future__ import annotations

import _paths  # noqa: F401  (sys.path bootstrap, safe under both invocations)

import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from human_targets import DEFAULT_DATA_ROOT, human_targets
from human_error_type import classify_trajectory


EXP1_EDIT_DIR = os.path.join("Behavioral data", "Experiment 1", "Edit sequences")
_FNAME_RE = re.compile(r"^subj_([^_]+)_trial_([0-9a-f]+)\.json\.csv$")


def _collect_exp1_classifications() -> pd.DataFrame:
    root = os.path.join(DEFAULT_DATA_ROOT, EXP1_EDIT_DIR)
    tags = pd.read_csv(os.path.join(DEFAULT_DATA_ROOT, "Task tags.csv"))
    tags["task_id"] = tags["Task Name"].str.replace(".json", "", regex=False)
    trial_order = dict(zip(tags["task_id"],
                           tags["problem order (experiment 1)"]))

    rows = []
    for td in sorted(os.listdir(root)):
        if not td.endswith(".json"):
            continue
        tid = td[:-5]
        order = trial_order.get(tid)
        if order is None or pd.isna(order):
            continue
        task_dir = os.path.join(root, td)
        for fname in os.listdir(task_dir):
            m = _FNAME_RE.match(fname)
            if not m:
                continue
            subj = m.group(1)
            try:
                r = classify_trajectory(tid, os.path.join(task_dir, fname))
            except Exception:
                continue
            r["subject_id"] = subj
            r["trial_index"] = int(order)
            r["experiment"] = 1
            rows.append(r)
    return pd.DataFrame(rows)


def build_figure(df: pd.DataFrame,
                 out_path: str = "prior_analysis/exp1_time_trend_figure.png"):
    df = df.copy()
    df["has_wrong"] = df["n_wrong"] > 0

    # -- overall binned trend --
    # Bin trial_index into deciles (8 bins × ~9 tasks)
    n_bins = 10
    df["bin"] = pd.cut(df["trial_index"], n_bins, labels=False)
    overall = df.groupby("bin").agg(
        n=("trial_index", "size"),
        mid_trial=("trial_index", "mean"),
        mean_motor=("motor_rate", "mean"),
        sem_motor=("motor_rate", lambda x: x.std() / np.sqrt(len(x))),
        mean_cog=("cognitive_rate", "mean"),
        sem_cog=("cognitive_rate", lambda x: x.std() / np.sqrt(len(x))),
    ).reset_index()

    # -- per-subject regression slopes --
    per_subj_slopes = []
    for subj, sub in df.groupby("subject_id"):
        if len(sub) < 10:
            continue
        # Regress motor_rate & cog_rate on trial_index
        t = sub["trial_index"].values.astype(float)
        for feat in ("motor_rate", "cognitive_rate"):
            y = sub[feat].values
            if np.isnan(y).all() or np.std(y) == 0:
                continue
            b = np.polyfit(t, y, 1)[0]
            per_subj_slopes.append({
                "subject_id": subj, "feature": feat, "slope_per_trial": b,
                "n_trials": len(sub),
            })
    slopes = pd.DataFrame(per_subj_slopes)

    # -- overall Spearman rho (non-parametric trend) --
    rho_motor, p_motor = spearmanr(df["trial_index"], df["motor_rate"])
    rho_cog, p_cog = spearmanr(df["trial_index"], df["cognitive_rate"])

    # -- Figure --
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 15,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0],
                          hspace=0.45, wspace=0.35)

    motor_col = "#3a66a5"
    cog_col = "#b84a3d"

    # (A) Binned trend: motor vs cognitive over trial order
    axA = fig.add_subplot(gs[0, :2])
    axA.errorbar(overall["mid_trial"], overall["mean_motor"],
                 yerr=overall["sem_motor"], color=motor_col, lw=2.5,
                 marker="o", markersize=9, capsize=4,
                 label=f"motor-slip rate  (ρ={rho_motor:+.2f}, p={p_motor:.1e})")
    axA.errorbar(overall["mid_trial"], overall["mean_cog"],
                 yerr=overall["sem_cog"], color=cog_col, lw=2.5,
                 marker="s", markersize=9, capsize=4,
                 label=f"cognitive-error rate  (ρ={rho_cog:+.2f}, p={p_cog:.1e})")
    axA.set_xlabel("Trial index (Exp 1 fixed order, 1–75)")
    axA.set_ylabel("Mean rate (fraction of wrong edits)")
    axA.set_title(
        "A.  Session-progression trend: do motor and cognitive rates change with trial order?",
        loc="left", pad=8,
    )
    axA.legend(loc="center right", frameon=False)
    axA.set_xlim(0, 76)
    axA.set_ylim(0, max(1.0, overall[["mean_motor", "mean_cog"]].max().max() + 0.1))

    # (B) Per-subject slope distribution
    axB = fig.add_subplot(gs[0, 2])
    m_slopes = slopes[slopes.feature == "motor_rate"]["slope_per_trial"]
    c_slopes = slopes[slopes.feature == "cognitive_rate"]["slope_per_trial"]
    bins = np.linspace(
        min(m_slopes.min(), c_slopes.min()) - 0.001,
        max(m_slopes.max(), c_slopes.max()) + 0.001,
        22,
    )
    axB.hist(m_slopes, bins=bins, color=motor_col, alpha=0.6,
             label=f"motor  (median={m_slopes.median():+.4f}/trial)")
    axB.hist(c_slopes, bins=bins, color=cog_col, alpha=0.6,
             label=f"cognitive  (median={c_slopes.median():+.4f}/trial)")
    axB.axvline(0, color="#333", linestyle="--", lw=0.8)
    axB.set_xlabel("Within-subject slope of rate vs trial index")
    axB.set_ylabel("subjects")
    axB.set_title("B.  Per-subject regression slopes", loc="left", pad=8)
    axB.legend(loc="upper right", frameon=False)

    # (C) Subject-by-subject spaghetti (motor)
    axC = fig.add_subplot(gs[1, 0])
    for subj, sub in df.groupby("subject_id"):
        sub = sub.sort_values("trial_index")
        if len(sub) < 10:
            continue
        axC.plot(sub["trial_index"], sub["motor_rate"],
                 color=motor_col, alpha=0.12, lw=0.8)
    # overlay mean
    axC.plot(overall["mid_trial"], overall["mean_motor"],
             color=motor_col, lw=3.0, marker="o", markersize=6)
    axC.set_xlabel("Trial index")
    axC.set_ylabel("Motor-slip rate")
    axC.set_title("C.  Each line = one participant  (motor rate)",
                  loc="left", pad=8)
    axC.set_xlim(0, 76); axC.set_ylim(0, 1)

    # (D) Subject-by-subject spaghetti (cognitive)
    axD = fig.add_subplot(gs[1, 1])
    for subj, sub in df.groupby("subject_id"):
        sub = sub.sort_values("trial_index")
        if len(sub) < 10:
            continue
        axD.plot(sub["trial_index"], sub["cognitive_rate"],
                 color=cog_col, alpha=0.12, lw=0.8)
    axD.plot(overall["mid_trial"], overall["mean_cog"],
             color=cog_col, lw=3.0, marker="s", markersize=6)
    axD.set_xlabel("Trial index")
    axD.set_ylabel("Cognitive-error rate")
    axD.set_title("D.  Each line = one participant  (cognitive rate)",
                  loc="left", pad=8)
    axD.set_xlim(0, 76); axD.set_ylim(0, 1)

    # (E) n_wrong edits per trial over time (are people investing less effort?)
    axE = fig.add_subplot(gs[1, 2])
    eff = df.groupby("bin").agg(
        mid_trial=("trial_index", "mean"),
        n_edits=("n_edits", "mean"),
        n_wrong=("n_wrong", "mean"),
    ).reset_index()
    rho_ed, p_ed = spearmanr(df["trial_index"], df["n_edits"])
    axE.plot(eff["mid_trial"], eff["n_edits"], color="#2c3e50",
             lw=2.5, marker="o", markersize=8,
             label=f"edits per trial  (ρ={rho_ed:+.2f})")
    axE.set_xlabel("Trial index")
    axE.set_ylabel("Mean edits per trial")
    axE.set_title("E.  Effort over the session", loc="left", pad=8)
    axE.legend(loc="upper right", frameon=False)
    axE.set_xlim(0, 76)

    fig.suptitle(
        "Within-subject progression across the Experiment 1 fixed task order (49 subjects × up to 75 trials each)\n"
        "Do participants slip more (motor) or reason better (cognitive) as the session advances?",
        y=0.995, fontsize=18,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out_path, dpi=260, bbox_inches="tight")
    print(f"wrote {out_path}")
    return {
        "rho_motor": float(rho_motor), "p_motor": float(p_motor),
        "rho_cog": float(rho_cog), "p_cog": float(p_cog),
        "n_subjects": int(df["subject_id"].nunique()),
        "n_trials": int(len(df)),
    }


def main():
    df = _collect_exp1_classifications()
    df.to_csv("prior_analysis/exp1_trial_trends.csv", index=False)
    print(f"classified {len(df)} Exp 1 trajectories "
          f"({df['subject_id'].nunique()} subjects, "
          f"{df['task_id'].nunique()} tasks).")
    summary = build_figure(df)
    print("\nKey statistics:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

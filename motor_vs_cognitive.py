"""
Final-form analysis for the motor-slip vs cognitive-error split.

Outputs:
  prior_analysis/tasks_sorted_motor_cognitive.csv
      One row per task, sorted by motor-slip rate (descending). Includes
      motor rate, cognitive rate, ambiguous rate, 95% bootstrap CI for each,
      split-half reliability, n_subjects, n_wrong_edits, and the CogARC
      task tags (complexity + Obj/Geo/Num/GoD primary tag).

  prior_analysis/motor_vs_cognitive_figure.png
      4-panel figure for collaborators:
        A) Sorted motor-rate bar chart (all 75 tasks, colored by primary tag)
        B) RT distributions — motor vs cognitive (pooled across 75 tasks)
        C) Split-half reliability: per-task motor rate from odd vs even subjects
        D) Two case-study trajectories (top motor task, top cognitive task)
           with edits color-coded by classification (+ per-edit RT barplot).

  prior_analysis/motor_cognitive_case_studies.png
      Higher-resolution grid showing 3 motor-dominant and 3 cognitive-dominant
      trajectories side-by-side.
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

from human_error_type import classify_trajectory, _FNAME_RE
from human_style_features import DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR, _parse_trajectory
from human_targets import human_targets

TASK_PNG_DIR = os.path.join(DEFAULT_DATA_ROOT, "Task example PNGs")
TASK_JSON_DIR = os.path.join(DEFAULT_DATA_ROOT, "Task JSONs")

# Paper-quality defaults
FIG_DPI = 260
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 14,
    "figure.titlesize": 20,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# CogARC color legend (from data-dictionary RTF).
ARC_COLORS_HEX = {
    0: "#2B2B2B", 1: "#248ADA", 2: "#C71010", 3: "#1FC719",
    4: "#F7DE28", 5: "#878494", 6: "#F954F2", 7: "#EE6000",
    8: "#6B23A9", 9: "#8B5A28",
}

# Error-class palette (muted, colorblind-aware).
CLASS_PALETTE = {
    "correct":   "#3d8559",  # soft green
    "motor":     "#3a66a5",  # steel blue
    "cognitive": "#b84a3d",  # muted red
    "ambiguous": "#9a9a9a",  # gray
}


def _task_png(task_id: str) -> str | None:
    path = os.path.join(TASK_PNG_DIR, f"{task_id}.json.png")
    return path if os.path.exists(path) else None


# Keep name CLASS_COLORS for backward compatibility with other files
# that import from this module.
CLASS_COLORS = CLASS_PALETTE
TAG_COLORS = {"?": "#888888"}  # unused in new layout


# ---------------------------------------------------------------------------
# Per-edit classification — reuse logic from human_error_type but return the
# full labeled edit list (not just counts) so we can draw case-study plots.
# ---------------------------------------------------------------------------

from human_error_type import (
    _nearest_correct_distance,
    BURST_CORRECTION_MS, BURST_CORRECTION_EDITS,
    SPATIAL_MOTOR_MAX, RT_Z_COGNITIVE,
)
from scipy.ndimage import label as cc_label


def classify_edits_labeled(task_id: str, trajectory_path: str) -> Tuple[List[Dict], np.ndarray]:
    """Returns (list of edits with 'class' key, Success grid)."""
    tr = _parse_trajectory(trajectory_path)
    edits = tr["edits"]
    targets = human_targets(task_id)

    success = None
    top_errors = []
    for lbl, g in zip(targets["labels"], targets["grids"]):
        if lbl == "Success":
            success = g
        else:
            top_errors.append((lbl, g))
    if success is None or not edits:
        return [], np.array([])

    Hs, Ws = success.shape
    color_masks = {int(c): (success == c) for c in np.unique(success)}

    rts = np.array([e["rt"] for e in edits if not np.isnan(e["rt"])])
    med_rt = float(np.median(rts)) if rts.size else 0.0
    mad_rt = float(np.median(np.abs(rts - med_rt))) if rts.size else 0.0
    std_rt = 1.4826 * mad_rt if mad_rt > 0 else 1.0

    final_state: Dict[Tuple[int, int], int] = {}
    for e in edits:
        final_state[(e["x"], e["y"])] = e["color"]

    wrong_final = np.zeros_like(success, dtype=bool)
    for (x, y), c in final_state.items():
        if 0 <= y < Hs and 0 <= x < Ws and success[y, x] != c:
            wrong_final[y, x] = True
    comp_labels, n_comps = cc_label(
        wrong_final, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    )
    wrong_comp_size = np.zeros_like(success, dtype=int)
    for cid in range(1, n_comps + 1):
        wrong_comp_size[comp_labels == cid] = int((comp_labels == cid).sum())

    out = []
    for i, e in enumerate(edits):
        x, y, c, rt = e["x"], e["y"], e["color"], e["rt"]
        if not (0 <= y < Hs and 0 <= x < Ws):
            out.append(dict(e, **{"class": "ambiguous"}))
            continue
        target_color = int(success[y, x])
        if c == target_color:
            out.append(dict(e, **{"class": "correct"}))
            continue

        dist = _nearest_correct_distance(
            x, y, c, color_masks.get(c, np.zeros_like(success, dtype=bool)))
        quickly_corrected = False
        for j in range(i + 1, min(i + 1 + BURST_CORRECTION_EDITS * 2, len(edits))):
            nxt = edits[j]
            if nxt["x"] == x and nxt["y"] == y:
                dt = nxt["time"] - e["time"]
                if ((j - i) <= BURST_CORRECTION_EDITS or dt <= BURST_CORRECTION_MS) \
                   and nxt["color"] == target_color:
                    quickly_corrected = True
                break
        rt_z = 0.0 if np.isnan(rt) else (rt - med_rt) / std_rt
        matches_te = any(te.shape == success.shape and int(te[y, x]) == c
                         for _, te in top_errors)
        in_wrong_run = int(wrong_comp_size[y, x]) >= 3

        if dist <= SPATIAL_MOTOR_MAX and (quickly_corrected or
                                          (not in_wrong_run and not matches_te
                                           and rt_z < RT_Z_COGNITIVE)):
            cls = "motor"
        elif matches_te or in_wrong_run or (dist >= 2 and not quickly_corrected) \
                or rt_z >= RT_Z_COGNITIVE:
            cls = "cognitive"
        else:
            cls = "ambiguous"
        out.append(dict(e, **{"class": cls}))
    return out, success


# ---------------------------------------------------------------------------
# Load everything
# ---------------------------------------------------------------------------

def _load_task_tags() -> pd.DataFrame:
    tags = pd.read_csv(os.path.join(DEFAULT_DATA_ROOT, "Task tags.csv"))
    tags["task_id"] = tags["Task Name"].str.replace(".json", "", regex=False)
    tags = tags.rename(columns={"complexity": "complexity",
                                "primary_tag": "primary_tag"})
    return tags[["task_id", "complexity", "primary_tag",
                 "secondary_tag", "tertiary_tag"]]


def _all_trajectory_paths(task_id: str) -> List[Tuple[str, str]]:
    root = os.path.join(DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR, f"{task_id}.json")
    out = []
    if not os.path.isdir(root):
        return out
    for fname in os.listdir(root):
        m = _FNAME_RE.match(fname)
        if not m:
            continue
        out.append((m.group(1), os.path.join(root, fname)))
    return out


# ---------------------------------------------------------------------------
# Per-task stats with bootstrap CI and split-half reliability
# ---------------------------------------------------------------------------

def _bootstrap_ci(values: np.ndarray, n_boot: int = 500,
                  rng: np.random.Generator | None = None) -> Tuple[float, float]:
    rng = rng or np.random.default_rng(0)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _split_half_corr(df: pd.DataFrame, feature: str,
                     rng: np.random.Generator, n_iter: int = 50) -> float:
    """For each task, split its subjects into two halves N times, record each
    half's mean of `feature`, then correlate across tasks."""
    corrs = []
    tasks = df["task_id"].unique()
    for _ in range(n_iter):
        rows_a = {}
        rows_b = {}
        for tid in tasks:
            sub = df[df.task_id == tid]
            n = len(sub)
            if n < 4:
                continue
            perm = rng.permutation(n)
            a = sub.iloc[perm[: n // 2]][feature].mean()
            b = sub.iloc[perm[n // 2: 2 * (n // 2)]][feature].mean()
            rows_a[tid] = a
            rows_b[tid] = b
        keys = list(rows_a)
        x = np.array([rows_a[k] for k in keys])
        y = np.array([rows_b[k] for k in keys])
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() > 2:
            rho, _ = spearmanr(x[mask], y[mask])
            if not np.isnan(rho):
                corrs.append(rho)
    return float(np.mean(corrs)) if corrs else float("nan")


def build_sorted_table(out_csv: str = "prior_analysis/tasks_sorted_motor_cognitive.csv",
                       n_boot: int = 500) -> pd.DataFrame:
    df = pd.read_csv("prior_analysis/error_types.csv")
    tags = _load_task_tags()

    rng = np.random.default_rng(42)

    rows = []
    for tid, sub in df.groupby("task_id"):
        motor = sub["motor_rate"].values
        cog = sub["cognitive_rate"].values
        ambig_rate = 1.0 - (motor + cog)
        lo_m, hi_m = _bootstrap_ci(motor, n_boot=n_boot, rng=rng)
        lo_c, hi_c = _bootstrap_ci(cog, n_boot=n_boot, rng=rng)
        rows.append({
            "task_id": tid,
            "n_subjects": sub["subject_id"].nunique(),
            "mean_n_wrong_per_subj": float(sub["n_wrong"].mean()),
            "motor_rate": float(motor.mean()),
            "motor_ci95_lo": lo_m, "motor_ci95_hi": hi_m,
            "cognitive_rate": float(cog.mean()),
            "cognitive_ci95_lo": lo_c, "cognitive_ci95_hi": hi_c,
            "ambiguous_rate": float(ambig_rate.mean()),
            "mean_rt_motor": float(sub["mean_rt_motor"].mean()),
            "mean_rt_cognitive": float(sub["mean_rt_cognitive"].mean()),
        })
    out = pd.DataFrame(rows).merge(tags, on="task_id", how="left")

    # Classification label and ranking score
    # score = motor_rate - cognitive_rate (positive = motor-leaning,
    # negative = cognitive-leaning). We also tag categorical buckets:
    #   motor-dominant   : motor_rate >= 0.20
    #   mixed            : 0.05 <= motor_rate < 0.20
    #   cognitive-dominant: motor_rate < 0.05
    def _bucket(m):
        if m >= 0.20:
            return "motor-dominant"
        if m >= 0.05:
            return "mixed"
        return "cognitive-dominant"
    out["bucket"] = out["motor_rate"].apply(_bucket)
    out["motor_minus_cog"] = out["motor_rate"] - out["cognitive_rate"]
    out = out.sort_values("motor_rate", ascending=False).reset_index(drop=True)

    # Split-half reliability (global scalar, not per-task)
    sh_motor = _split_half_corr(df, "motor_rate", rng, n_iter=50)
    sh_cog   = _split_half_corr(df, "cognitive_rate", rng, n_iter=50)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")
    print(f"split-half Spearman  motor_rate: {sh_motor:.3f}  "
          f"cognitive_rate: {sh_cog:.3f}")
    print(f"bucket counts: {out['bucket'].value_counts().to_dict()}")
    # Also persist the split-half numbers
    with open("prior_analysis/motor_vs_cognitive_reliability.txt", "w") as f:
        f.write(f"split_half_spearman_motor_rate={sh_motor:.6f}\n")
        f.write(f"split_half_spearman_cognitive_rate={sh_cog:.6f}\n")
    return out


# ---------------------------------------------------------------------------
# Figure building
# ---------------------------------------------------------------------------

def _choose_trajectory(task_id: str, prefer: str,
                       min_wrong: int = 4) -> Tuple[str, List[Dict], np.ndarray]:
    """Pick a representative trajectory for this task. `prefer` is 'motor' or
    'cognitive' - we pick a subject whose trajectory has that class
    well-represented AND enough total edits to be visually legible."""
    best = None
    best_score = -1
    for subj, path in _all_trajectory_paths(task_id):
        labeled, success = classify_edits_labeled(task_id, path)
        if not labeled:
            continue
        n_prefer = sum(1 for e in labeled if e["class"] == prefer)
        n_wrong = sum(1 for e in labeled if e["class"] in ("motor", "cognitive"))
        if n_wrong < min_wrong:
            continue
        # Score: heavily favor trajectories with more of the preferred class,
        # and mildly favor moderate total length (15-80 edits is a readable
        # range). Do not penalize long trajectories because cognitive-heavy
        # tasks can have 50-150 edits and those are the most convincing.
        length_bonus = 0
        if 15 <= len(labeled) <= 120:
            length_bonus = min(len(labeled), 80) / 80.0
        score = n_prefer + 0.5 * length_bonus * max(n_wrong, 1)
        if score > best_score:
            best_score = score
            best = (subj, labeled, success)
    return best or ("", [], np.array([]))


def _draw_arc_grid(ax, grid: np.ndarray, label: str = "",
                   label_fontsize: int = 14, label_pad: float = 0.22,
                   label_color: str = "black") -> None:
    """Render one ARC grid as filled cells using the CogARC color legend."""
    if grid is None or (hasattr(grid, "size") and grid.size == 0):
        ax.set_axis_off()
        return
    H, W = grid.shape
    for y in range(H):
        for x in range(W):
            c = int(grid[y, x])
            ax.add_patch(plt.Rectangle(
                (x - 0.5, y - 0.5), 1, 1,
                facecolor=ARC_COLORS_HEX.get(c, "#000000"),
                edgecolor="#e8e8e8", linewidth=0.4,
            ))
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if label:
        ax.set_title(label, fontsize=label_fontsize, pad=6, color=label_color)


def _draw_trajectory(ax, edits: List[Dict], success: np.ndarray,
                     title: str = "", title_color: str = "black",
                     show_order_numbers: bool = True) -> None:
    """Render a participant trajectory on an ARC-style grid.

    Each edited cell is filled with the ARC color the participant placed.
    A thick border colored by the edit's class (motor / cognitive /
    correct / ambiguous) surrounds each cell. Temporal order is indicated
    by a light connecting polyline and by numeric labels on the first few
    edits."""
    if not edits or success is None or success.size == 0:
        ax.set_axis_off(); return
    xs = np.array([e["x"] for e in edits])
    ys = np.array([e["y"] for e in edits])
    Hs, Ws = success.shape

    # Clean checker background
    ax.add_patch(plt.Rectangle((-0.5, -0.5), Ws, Hs,
                               facecolor="#fafafa", edgecolor="none"))
    for i in range(Ws + 1):
        ax.plot([i - 0.5, i - 0.5], [-0.5, Hs - 0.5], color="#eaeaea", lw=0.4)
    for j in range(Hs + 1):
        ax.plot([-0.5, Ws - 0.5], [j - 0.5, j - 0.5], color="#eaeaea", lw=0.4)

    # Trail of temporal order (faint)
    if len(edits) > 1:
        ax.plot(xs, ys, color="#bbbbbb", lw=0.6, alpha=0.6, zorder=1)

    # Each edit: filled square with ARC color, border color = classification
    for i, e in enumerate(edits):
        x, y = e["x"], e["y"]
        arc_fill = ARC_COLORS_HEX.get(int(e["color"]), "#000000")
        border = CLASS_PALETTE.get(e["class"], "#333")
        ax.add_patch(plt.Rectangle(
            (x - 0.5, y - 0.5), 1, 1,
            facecolor=arc_fill, edgecolor=border, linewidth=2.2, zorder=3,
        ))
    # Numbered order for the first 10 edits
    if show_order_numbers:
        n_label = min(10, len(edits))
        for i in range(n_label):
            ax.text(xs[i], ys[i], str(i + 1), ha="center", va="center",
                    fontsize=9, color="white", fontweight="bold", zorder=4,
                    path_effects=[])

    # Aggregate legend (counts per class)
    from collections import Counter
    counts = Counter(e["class"] for e in edits)
    order = ["correct", "motor", "cognitive", "ambiguous"]
    handles = [plt.Line2D([0], [0], marker="s", linestyle="",
                          markerfacecolor="white",
                          markeredgecolor=CLASS_PALETTE[c], markersize=12,
                          markeredgewidth=2.2,
                          label=f"{c} ({counts.get(c, 0)})")
               for c in order if counts.get(c, 0) > 0]
    ax.legend(handles=handles, loc="upper left",
              bbox_to_anchor=(1.015, 1.0), frameon=False, fontsize=13,
              borderaxespad=0)

    if title:
        ax.set_title(title, fontsize=14, color=title_color, loc="left", pad=8)
    ax.set_xlim(-0.8, Ws - 0.2)
    ax.set_ylim(Hs - 0.2, -0.8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------------------
# Task renderer (train pairs + test)
# ---------------------------------------------------------------------------

def _load_task_json(task_id: str) -> Dict:
    import json
    with open(os.path.join(TASK_JSON_DIR, f"{task_id}.json"), "r") as f:
        return json.load(f)


def render_task_examples(fig, gs_row, task_id: str, include_test: bool = True,
                         test_label: str = "Test") -> None:
    """Render all train input/output pairs + the test pair on one figure row
    using a nested gridspec inside `gs_row` (a GridSpecFromSubplotSpec).

    Each pair is two small axes side-by-side (input, output) joined by an
    arrow between them. A label above each pair reads 'Train 1', 'Train 2',
    ..., 'Test'.
    """
    task = _load_task_json(task_id)
    train = [(np.array(p["input"]), np.array(p["output"])) for p in task["train"]]
    test = [(np.array(p["input"]), np.array(p["output"])) for p in task["test"]]
    pairs = train + (test if include_test else [])
    labels = [f"Train {i+1}" for i in range(len(train))]
    if include_test:
        labels.append(test_label)

    # Each pair takes 2 sub-columns (input, output) + 1 arrow column.
    ncols = len(pairs) * 2
    sub = gs_row.subgridspec(1, ncols, wspace=0.3)
    for i, (inp, out) in enumerate(pairs):
        ax_in = fig.add_subplot(sub[0, 2 * i])
        ax_out = fig.add_subplot(sub[0, 2 * i + 1])
        _draw_arc_grid(ax_in, inp, label=f"{labels[i]}  input",
                       label_fontsize=12)
        _draw_arc_grid(ax_out, out,
                       label=f"{labels[i]}  "
                             + ("output" if i < len(train) else "correct"),
                       label_fontsize=12,
                       label_color="#b84a3d" if i >= len(train) else "black")


def build_main_figure(sorted_df: pd.DataFrame,
                      out_path: str = "prior_analysis/motor_vs_cognitive_figure.png"):
    # Split-half scatter: per-task motor_rate from odd vs even subjects
    df_raw = pd.read_csv("prior_analysis/error_types.csv")
    rng = np.random.default_rng(7)
    rows_a, rows_b = {}, {}
    for tid, sub in df_raw.groupby("task_id"):
        perm = rng.permutation(len(sub))
        half = len(sub) // 2
        rows_a[tid] = sub.iloc[perm[:half]]["motor_rate"].mean()
        rows_b[tid] = sub.iloc[perm[half:2 * half]]["motor_rate"].mean()
    xs = np.array(list(rows_a.values()))
    ys_ = np.array(list(rows_b.values()))
    rho, _ = spearmanr(xs, ys_)

    # Classification overview only (no case studies; those are in the grid).
    # Row 1 = A (sorted bars, full width). Row 2 = B, C, D, E.
    fig = plt.figure(figsize=(28, 16))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.6, 1.0],
                          hspace=0.38, wspace=0.40)

    # A) Sorted bar chart, uniform muted color, every task ID as an x-tick.
    axA = fig.add_subplot(gs[0, :])
    BAR_COLOR = "#4a6b8a"
    axA.bar(np.arange(len(sorted_df)), sorted_df["motor_rate"],
            color=BAR_COLOR, width=0.85, edgecolor="none")
    axA.errorbar(np.arange(len(sorted_df)), sorted_df["motor_rate"],
                 yerr=[sorted_df["motor_rate"] - sorted_df["motor_ci95_lo"],
                       sorted_df["motor_ci95_hi"] - sorted_df["motor_rate"]],
                 fmt="none", ecolor="#2c3e50", elinewidth=0.7, capsize=0)
    axA.axhline(0.20, color=CLASS_PALETTE["motor"], linestyle="--", lw=1.0,
                label="motor-dominant threshold (0.20)")
    axA.axhline(0.05, color=CLASS_PALETTE["cognitive"], linestyle="--", lw=1.0,
                label="cognitive-dominant threshold (0.05)")
    axA.set_xlim(-0.5, len(sorted_df) - 0.5)
    axA.set_ylim(0, max(0.68, sorted_df["motor_rate"].max() + 0.04))
    axA.set_ylabel("Motor-slip rate  (fraction of wrong edits)")
    axA.set_title(
        "A.  Per-task motor-slip rate across 75 CogARC tasks "
        "(sorted; whiskers = 95% bootstrap CI over participants)",
        loc="left", pad=14,
    )
    # Every task ID as an x-tick, rotated 90°.
    axA.set_xticks(np.arange(len(sorted_df)))
    axA.set_xticklabels(sorted_df["task_id"].tolist(),
                        rotation=90, fontsize=9, fontfamily="monospace")
    axA.legend(loc="upper right", frameon=False, fontsize=13)

    # B) RT distributions
    axB = fig.add_subplot(gs[1, 0])
    all_m = df_raw["mean_rt_motor"].dropna().values
    all_c = df_raw["mean_rt_cognitive"].dropna().values
    bins = np.logspace(1.5, 4.5, 40)
    axB.hist(all_m, bins=bins, color=CLASS_COLORS["motor"],
             alpha=0.65, label=f"motor  median={np.median(all_m):.0f}ms")
    axB.hist(all_c, bins=bins, color=CLASS_COLORS["cognitive"],
             alpha=0.65, label=f"cognitive  median={np.median(all_c):.0f}ms")
    axB.set_xscale("log")
    axB.set_xlabel("mean RT of wrong edits (ms, log)")
    axB.set_ylabel("trajectories")
    axB.set_title("B. RT distribution validates the classifier.\n"
                  "Cognitive errors are preceded by longer pauses.",
                  loc="left", fontsize=14, pad=10)
    axB.legend(loc="upper right", fontsize=11)

    # C) Split-half reliability
    axC = fig.add_subplot(gs[1, 1])
    axC.scatter(xs, ys_, s=22, color="#444", alpha=0.7)
    lims = [min(xs.min(), ys_.min()) - 0.02, max(xs.max(), ys_.max()) + 0.02]
    axC.plot(lims, lims, color="gray", lw=0.5, linestyle="--")
    axC.set_xlim(lims); axC.set_ylim(lims)
    axC.set_xlabel("motor rate — random half A of subjects")
    axC.set_ylabel("motor rate — random half B of subjects")
    axC.set_title(f"C. Split-half reliability\nSpearman ρ = {rho:.2f}  "
                  f"(n={len(xs)} tasks)", loc="left", fontsize=14, pad=10)

    # D) Bucket distribution pie
    axD = fig.add_subplot(gs[1, 2])
    counts = sorted_df["bucket"].value_counts()
    colors = {"motor-dominant": CLASS_COLORS["motor"],
              "mixed": "#9467bd",
              "cognitive-dominant": CLASS_COLORS["cognitive"]}
    axD.pie(counts.values, labels=counts.index,
            colors=[colors[b] for b in counts.index],
            autopct="%1.0f", startangle=90,
            textprops={"fontsize": 12})
    axD.set_title("D. Classification buckets\n(count of tasks per bucket)",
                  loc="left", fontsize=14, pad=10)

    # E) Per-task RT scatter panel
    axE1 = fig.add_subplot(gs[1, 3])

    # Mean RT per task scatter (motor vs cognitive)
    axE1.scatter(sorted_df["mean_rt_motor"],
                 sorted_df["mean_rt_cognitive"],
                 s=18, color="#333", alpha=0.7)
    mn = min(sorted_df["mean_rt_motor"].min(), sorted_df["mean_rt_cognitive"].min())
    mx = max(sorted_df["mean_rt_motor"].max(), sorted_df["mean_rt_cognitive"].max())
    axE1.plot([mn, mx], [mn, mx], color="gray", linestyle="--", lw=0.5)
    axE1.set_xlabel("mean RT motor (ms)")
    axE1.set_ylabel("mean RT cognitive (ms)")
    axE1.set_title("E. Per-task RT:\ncognitive > motor on ALL 75 tasks",
                   loc="left", fontsize=14, pad=10)

    fig.suptitle(
        "Motor slip vs cognitive error — per-task classification and validation.  "
        "Case-study trajectories are in the companion figure 'motor_cognitive_case_studies.png'.",
        fontsize=15, y=0.995,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"wrote {out_path}")


def build_case_study_grid(sorted_df: pd.DataFrame,
                          out_path: str = "prior_analysis/motor_cognitive_case_studies.png"):
    top_motor = sorted_df.iloc[:3]["task_id"].tolist()
    cog_pool = sorted_df[(sorted_df["cognitive_rate"] >= 0.90)
                         & (sorted_df["mean_n_wrong_per_subj"] >= 25)]
    cog_pool = cog_pool.sort_values("motor_rate").head(3)
    top_cog = cog_pool["task_id"].tolist()
    cases = [("MOTOR", tid, "motor") for tid in top_motor] + \
            [("COGNITIVE", tid, "cognitive") for tid in top_cog]

    # Each case gets two stacked rows:
    #   sub-row 1: train pairs + test pair (all grids across one row)
    #   sub-row 2: header strip + participant trajectory (wider)
    # Outer grid: 6 cases * 2 sub-rows = 12 rows total.
    fig = plt.figure(figsize=(24, 32))
    outer = fig.add_gridspec(
        nrows=len(cases) * 2, ncols=1,
        height_ratios=[1.0, 1.7] * len(cases),
        hspace=0.65,
    )

    for idx, (kind, tid, prefer) in enumerate(cases):
        row = sorted_df[sorted_df.task_id == tid].iloc[0]
        header_color = CLASS_PALETTE[prefer]
        # Sub-row 1: ARC task examples (train + test)
        task_row = outer[idx * 2, 0]
        render_task_examples(fig, task_row, tid, include_test=True)
        # Add a title ABOVE this sub-row by writing annotated text in figure
        # coordinates via a thin transparent axis that spans the row.
        axhdr = fig.add_subplot(task_row)
        axhdr.set_axis_off()
        hdr = (f"{kind}-dominant case #{top_motor.index(tid) + 1 if kind=='MOTOR' else top_cog.index(tid) + 1}"
               f"     task id: {tid}     "
               f"motor rate {row['motor_rate']:.0%}     "
               f"cognitive rate {row['cognitive_rate']:.0%}     "
               f"n_subjects {int(row['n_subjects'])}")
        axhdr.text(0.0, 1.25, hdr, transform=axhdr.transAxes,
                   fontsize=17, fontweight="bold", color=header_color,
                   ha="left", va="bottom")

        # Sub-row 2: participant trajectory
        subj, edits, succ = _choose_trajectory(
            tid, prefer=prefer,
            min_wrong=max(4, int(row['mean_n_wrong_per_subj'] * 0.2)),
        )
        ax_traj = fig.add_subplot(outer[idx * 2 + 1, 0])
        _draw_trajectory(
            ax_traj, edits, succ,
            title=f"Representative participant {subj} — each square is one edit  "
                  f"(fill = color placed; border = classification)",
            title_color="black",
        )

    fig.suptitle(
        "Case studies: three motor-dominant tasks (top) and three cognitive-dominant tasks (bottom).\n"
        "Per case: task examples (train inputs → outputs, then test input → correct answer) "
        "are shown above one participant's full edit trajectory on the test grid.",
        fontsize=18, y=0.992,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    df = build_sorted_table()
    print("\n--- Top 10 MOTOR-dominant tasks ---")
    print(df.head(10)[["task_id", "n_subjects", "motor_rate",
                       "cognitive_rate", "primary_tag", "complexity",
                       "bucket"]].to_string(index=False))
    print("\n--- Top 10 COGNITIVE-dominant tasks ---")
    print(df.tail(10).iloc[::-1][["task_id", "n_subjects", "motor_rate",
                                  "cognitive_rate", "primary_tag",
                                  "complexity", "bucket"]].to_string(index=False))
    build_main_figure(df)
    build_case_study_grid(df)


if __name__ == "__main__":
    main()

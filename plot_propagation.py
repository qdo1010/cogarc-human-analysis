"""
Visualize edit-propagation features:
  1) Distributions across the 75-task population with the 3 majority-error
     tasks marked, so you can see where they sit relative to others.
  2) One real participant trajectory per highlighted task, rendered as the
     grid of edited cells with edit order as arrows and each edit marked
     as EXTENSION (adjacent to a prior same-color edit) or SEED (new
     disjoint region).

Outputs:
  prior_analysis/edit_propagation_distributions.png
  prior_analysis/edit_propagation_trajectories.png
"""

from __future__ import annotations

import _paths  # noqa: F401  (sys.path bootstrap, safe under both invocations)

import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from human_style_features import (
    DEFAULT_DATA_ROOT, _EXP2_EDIT_DIR, _parse_trajectory,
    all_style_features,
)


HIGHLIGHT = {
    "0d87d2a6": "#d62728",  # red
    "1f0c79e5": "#2ca02c",  # green
    "834ec97d": "#1f77b4",  # blue
}

OUT_DIR = "prior_analysis"


# ---------------------------------------------------------------------------
# Figure 1: distributions
# ---------------------------------------------------------------------------

def plot_distributions(df: pd.DataFrame, out_path: str) -> None:
    specs = [
        ("extension_rate_mean",
         "Extension rate", "fraction of edits extending\nan existing same-color region", "linear"),
        ("new_seed_rate_mean",
         "New-seed rate", "fraction starting a new disjoint region", "linear"),
        ("stamp_burst_rate_mean",
         "Stamp-burst rate", "fraction of edits in rapid\nsame-color adjacent bursts", "linear"),
        ("same_color_component_count_mean",
         "Same-color component count",
         "# disjoint painted regions\n(per subject, averaged)", "linear"),
        ("same_color_component_max_mean",
         "Largest same-color region",
         "max # cells in any single\npainted component (log)", "log"),
        ("n_edits_mean",
         "Edits per subject", "total grid edits per subject (log)", "log"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (col, title, subtitle, xscale) in zip(axes.flat, specs):
        vals = df[col].values
        vv = vals[vals > 0] if xscale == "log" else vals
        if xscale == "log":
            bins = np.logspace(np.log10(max(vv.min(), 1e-3)),
                               np.log10(vv.max()), 25)
        else:
            bins = 25
        ax.hist(vv, bins=bins, color="#cccccc", edgecolor="#555", alpha=0.9)
        if xscale == "log":
            ax.set_xscale("log")
        # Overlay the 3 highlighted tasks
        for tid, color in HIGHLIGHT.items():
            v = df.loc[df.task_id == tid, col].values
            if len(v):
                ax.axvline(v[0], color=color, lw=2, label=tid)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(subtitle, fontsize=9)
        ax.set_ylabel("tasks")
    # Shared legend at top-right
    axes.flat[0].legend(loc="upper right", fontsize=9, title="majority-error tasks")
    fig.suptitle(
        "Edit-propagation features across 75 CogARC tasks (Experiment 2)\n"
        "Each histogram = 75 tasks. Vertical lines = the 3 tasks where the "
        "most common human response is an error.",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Figure 2: trajectories
# ---------------------------------------------------------------------------

def _choose_trajectory_file(task_id: str, data_root: str) -> str:
    """Pick a median-length trajectory for this task (so it's representative)."""
    root = os.path.join(data_root, _EXP2_EDIT_DIR, f"{task_id}.json")
    files = [os.path.join(root, f) for f in os.listdir(root) if f.endswith(".csv")]
    lens = []
    for p in files:
        tr = _parse_trajectory(p)
        lens.append((len(tr["edits"]), p))
    lens.sort()
    # Median
    return lens[len(lens) // 2][1]


ARC_COLORS = {
    0: "#2B2B2B", 1: "#248ADA", 2: "#C71010", 3: "#1FC719",
    4: "#F7DE28", 5: "#878494", 6: "#F954F2", 7: "#EE6000",
    8: "#6B23A9", 9: "#8B5A28",
}


def _classify_edits(edits: List[Dict]) -> List[str]:
    """For each edit, 'extension' if 4-adjacent to a prior same-color edit,
    else 'seed'. 'recolor' if same cell seen before."""
    seen_by_color: Dict[int, set] = {}
    out = []
    for e in edits:
        c, x, y = e["color"], e["x"], e["y"]
        cells = seen_by_color.setdefault(c, set())
        if (x, y) in cells:
            out.append("recolor")
        elif not cells:
            out.append("seed")
        elif ((x - 1, y) in cells or (x + 1, y) in cells
              or (x, y - 1) in cells or (x, y + 1) in cells):
            out.append("extension")
        else:
            out.append("seed")
        cells.add((x, y))
    return out


def plot_trajectories(out_path: str,
                      data_root: str = DEFAULT_DATA_ROOT) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for ax, (tid, border_color) in zip(axes, HIGHLIGHT.items()):
        path = _choose_trajectory_file(tid, data_root)
        tr = _parse_trajectory(path)
        edits = tr["edits"]
        classes = _classify_edits(edits)

        xs = np.array([e["x"] for e in edits])
        ys = np.array([e["y"] for e in edits])
        cs = [ARC_COLORS.get(e["color"], "#ffffff") for e in edits]

        gx, gy = xs.max() + 2, ys.max() + 2
        # Light grid background
        ax.add_patch(plt.Rectangle((-0.5, -0.5), gx, gy, facecolor="#f3f3f3",
                                   edgecolor="none"))
        for i in range(gx + 1):
            ax.plot([i - 0.5, i - 0.5], [-0.5, gy - 0.5], color="#dddddd", lw=0.3)
        for j in range(gy + 1):
            ax.plot([-0.5, gx - 0.5], [j - 0.5, j - 0.5], color="#dddddd", lw=0.3)

        # Draw order arrows between consecutive edits, faded
        for i in range(1, len(edits)):
            ax.annotate(
                "", xy=(xs[i], ys[i]), xytext=(xs[i - 1], ys[i - 1]),
                arrowprops=dict(arrowstyle="->", color="gray", alpha=0.25, lw=0.5),
            )

        # Plot edits: face color = ARC color, marker shape = extension / seed / recolor
        markers = {"extension": "o", "seed": "s", "recolor": "x"}
        for cls in set(classes):
            mask = [c == cls for c in classes]
            ax.scatter(xs[mask], ys[mask],
                       c=[cs[i] for i, m in enumerate(mask) if m],
                       s=120, marker=markers[cls],
                       edgecolors="black", linewidths=0.8,
                       label=f"{cls} ({sum(mask)})")

        # Number the first 12 edits so order is visible
        for i in range(min(12, len(edits))):
            ax.text(xs[i], ys[i], str(i + 1), fontsize=7, ha="center",
                    va="center", color="white", weight="bold")

        # Metrics for title
        ext_rate = sum(c == "extension" for c in classes) / max(
            sum(c in ("extension", "seed") for c in classes), 1)
        ax.set_title(
            f"{tid}\n"
            f"{len(edits)} edits  |  extension rate: {ext_rate:.0%}",
            fontsize=11, color=border_color,
        )
        ax.set_aspect("equal")
        ax.invert_yaxis()  # match grid[y][x] orientation
        ax.set_xlim(-1, gx - 0.5)
        ax.set_ylim(gy - 0.5, -1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(2)

        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "One representative participant per majority-error task: "
        "circles = extensions, squares = new seeds, × = re-color",
        fontsize=12, y=1.03,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    df = all_style_features()
    os.makedirs(OUT_DIR, exist_ok=True)
    plot_distributions(df, os.path.join(OUT_DIR, "edit_propagation_distributions.png"))
    plot_trajectories(os.path.join(OUT_DIR, "edit_propagation_trajectories.png"))


if __name__ == "__main__":
    main()

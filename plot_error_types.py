"""Visualize motor-vs-cognitive error classification across 75 tasks."""
from __future__ import annotations

import _paths  # noqa: F401  (sys.path bootstrap, safe under both invocations)

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("prior_analysis/error_types.csv")
per_task = pd.read_csv("prior_analysis/error_types_per_task.csv")

os.makedirs("prior_analysis", exist_ok=True)

# ---- Figure 1: per-task motor vs cognitive rate ---------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) stacked bar: 75 tasks sorted by cognitive rate, motor/cognitive/ambig
df_task = per_task.sort_values("mean_cognitive_rate", ascending=True).reset_index(drop=True)
motor = df_task["mean_motor_rate"].values
cog = df_task["mean_cognitive_rate"].values
ambig = 1.0 - motor - cog
x = np.arange(len(df_task))
axes[0].bar(x, motor, color="#1f77b4", label="motor slip")
axes[0].bar(x, cog, bottom=motor, color="#d62728", label="cognitive error")
axes[0].bar(x, ambig, bottom=motor + cog, color="#aaaaaa", label="ambiguous")
axes[0].set_xlim(-0.5, len(df_task) - 0.5)
axes[0].set_ylim(0, 1)
axes[0].set_xlabel("75 tasks (sorted by cognitive-error rate)")
axes[0].set_ylabel("fraction of wrong edits")
axes[0].set_title("Per-task composition of wrong edits")
axes[0].set_xticks([])
axes[0].legend(loc="lower right")

# (b) RT distribution of motor vs cognitive, pooled across trajectories
df_all = pd.read_csv("prior_analysis/error_types.csv")
m = df_all["mean_rt_motor"].dropna().values
c = df_all["mean_rt_cognitive"].dropna().values
bins = np.logspace(1.5, 4.5, 40)
axes[1].hist(m, bins=bins, color="#1f77b4", alpha=0.6, label=f"motor slip  (n={len(m)})")
axes[1].hist(c, bins=bins, color="#d62728", alpha=0.6, label=f"cognitive error (n={len(c)})")
axes[1].set_xscale("log")
axes[1].set_xlabel("mean RT of wrong edits (ms, log)")
axes[1].set_ylabel("trajectories")
axes[1].set_title("RT distribution — motor vs cognitive\n"
                  "Cognitive errors are preceded by longer pauses.")
axes[1].legend(loc="upper right")

fig.tight_layout()
fig.savefig("prior_analysis/error_types_overview.png", dpi=140, bbox_inches="tight")
print("wrote prior_analysis/error_types_overview.png")

# ---- Figure 2: scatter of task-level motor vs cognitive rate --------------
fig2, ax = plt.subplots(figsize=(7, 7))
ax.scatter(per_task["mean_motor_rate"], per_task["mean_cognitive_rate"],
           s=18, alpha=0.6, color="#444")
# Highlight the 3 majority-error tasks
highlight = {"0d87d2a6": "#d62728", "1f0c79e5": "#2ca02c", "834ec97d": "#1f77b4"}
for tid, col in highlight.items():
    if tid in per_task["task_id"].values:
        r = per_task[per_task.task_id == tid].iloc[0]
        ax.scatter(r["mean_motor_rate"], r["mean_cognitive_rate"],
                   s=120, facecolor=col, edgecolor="k", zorder=5, label=tid)
ax.set_xlabel("mean motor-slip rate (per task)")
ax.set_ylabel("mean cognitive-error rate (per task)")
ax.set_title("Task-level error composition\n"
             "(each dot is one of 75 CogARC tasks)", fontsize=11)
ax.set_xlim(-0.02, 0.5)
ax.set_ylim(0, 1.02)
ax.legend(loc="lower left")
fig2.tight_layout()
fig2.savefig("prior_analysis/error_types_scatter.png", dpi=140, bbox_inches="tight")
print("wrote prior_analysis/error_types_scatter.png")

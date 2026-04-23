"""
Individual differences in chunking behavior.

Two questions:

    (A) Reliability: for each chunk-feature, how much of a subject's
        profile is a stable trait vs noise? Split each subject's 75 tasks
        into two random halves, aggregate the feature per subject per
        half, then compute Spearman ρ across subjects between the two
        halves. Repeat N times, report mean ρ and 95% CI. Features with
        ρ > 0.3 are genuine individual-difference dimensions.

    (B) Does chunking style predict performance? Aggregate each subject's
        chunk features across their trajectories, merge with subject-level
        error rates (overall, motor, cognitive), and compute Spearman
        correlations.

Outputs:
    prior_analysis/individual_differences_reliability.csv
    prior_analysis/individual_differences_correlations.csv
    prior_analysis/individual_differences_subject_profile.csv
"""

from __future__ import annotations

import _paths  # noqa: F401
import argparse
import os
import random
from typing import List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# Chunk-level features to test as individual-difference dimensions.
CHUNK_FEATURES = [
    "size",             # cells per chunk (lumper vs splitter)
    "color_homogeneity",
    "is_connected",     # bool -> mean = fraction connected
    "bbox_area",        # spatial extent of chunks
    "fill_ratio",       # compactness within bbox
    "nn_chain_rate",    # draw-order adjacency
    "success_iou_best",
    "n_cells",          # distinct cells per chunk
]

# Also: per-trajectory chunking style needs n_chunks_total (one value per
# (subject, task) pair, not averaged across chunks).
TRAJ_FEATURES = ["n_chunks_total"]


def _aggregate_per_subject_task(chunks_df: pd.DataFrame) -> pd.DataFrame:
    """Mean of chunk features per (subject, task). Keeps n_chunks_total as-is."""
    agg = {c: "mean" for c in CHUNK_FEATURES}
    agg["n_chunks_total"] = "first"  # same across all rows for a (subject, task)
    return chunks_df.groupby(["subject_id", "task_id"]).agg(agg).reset_index()


def _split_half_reliability(per_subj_task: pd.DataFrame, features: List[str],
                            n_iter: int = 100, seed: int = 0) -> pd.DataFrame:
    """For each feature, repeatedly split each subject's tasks in half, average,
    then Spearman across subjects between the two halves."""
    rng = random.Random(seed)
    rows = []
    subjects = sorted(per_subj_task["subject_id"].unique())

    for feat in features:
        rhos = []
        for _ in range(n_iter):
            a_means, b_means = [], []
            for sid in subjects:
                sub = per_subj_task[per_subj_task["subject_id"] == sid]
                if len(sub) < 4:
                    continue
                idx = list(range(len(sub)))
                rng.shuffle(idx)
                half = len(idx) // 2
                a = sub.iloc[idx[:half]][feat].mean()
                b = sub.iloc[idx[half:]][feat].mean()
                if np.isnan(a) or np.isnan(b):
                    continue
                a_means.append(float(a))
                b_means.append(float(b))
            if len(a_means) < 5 or np.std(a_means) == 0 or np.std(b_means) == 0:
                continue
            rho, _ = spearmanr(a_means, b_means)
            if not np.isnan(rho):
                rhos.append(float(rho))
        if not rhos:
            rows.append({"feature": feat, "n_iter": 0,
                         "mean_rho": float("nan"),
                         "ci95_lo": float("nan"), "ci95_hi": float("nan")})
            continue
        rhos = np.array(rhos)
        rows.append({
            "feature": feat,
            "n_iter": len(rhos),
            "mean_rho": float(rhos.mean()),
            "ci95_lo": float(np.percentile(rhos, 2.5)),
            "ci95_hi": float(np.percentile(rhos, 97.5)),
            "sd_rho": float(rhos.std()),
        })
    return pd.DataFrame(rows)


def _per_subject_profile(per_subj_task: pd.DataFrame,
                         features: List[str]) -> pd.DataFrame:
    """Mean of each feature across a subject's tasks."""
    return per_subj_task.groupby("subject_id")[features].mean().reset_index()


def _load_performance(prior_dir: str) -> pd.DataFrame:
    """Merge subject-level error rates (overall, motor, cognitive, per-TE)."""
    # Overall + per-TE from chunk_error_per_subject
    err = pd.read_csv(os.path.join(prior_dir, "chunk_error_per_subject.csv"))
    err = err.rename(columns={
        "error_rate": "overall_error_rate",
    })

    # Motor vs cognitive from error_types (per-edit -> per-subject)
    et = pd.read_csv(os.path.join(prior_dir, "error_types.csv"))
    mc = et.groupby("subject_id").agg(
        n_edits_total=("n_edits", "sum"),
        n_motor=("n_motor", "sum"),
        n_cognitive=("n_cognitive", "sum"),
        n_wrong=("n_wrong", "sum"),
    ).reset_index()
    mc["motor_rate"] = mc["n_motor"] / mc["n_edits_total"].clip(lower=1)
    mc["cognitive_rate"] = mc["n_cognitive"] / mc["n_edits_total"].clip(lower=1)
    mc["motor_over_wrong"] = mc["n_motor"] / mc["n_wrong"].clip(lower=1)
    mc["cognitive_over_wrong"] = mc["n_cognitive"] / mc["n_wrong"].clip(lower=1)

    return err.merge(mc, on="subject_id", how="outer")


def _style_vs_performance(profile: pd.DataFrame, perf: pd.DataFrame,
                          style_features: List[str],
                          perf_metrics: List[str]) -> pd.DataFrame:
    df = profile.merge(perf, on="subject_id", how="inner")
    rows = []
    for s in style_features:
        for m in perf_metrics:
            x = df[s].values
            y = df[m].values
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() < 10:
                rows.append({"style": s, "performance": m,
                             "n": int(ok.sum()),
                             "rho": float("nan"),
                             "p_value": float("nan")})
                continue
            rho, p = spearmanr(x[ok], y[ok])
            rows.append({
                "style": s,
                "performance": m,
                "n": int(ok.sum()),
                "rho": float(rho),
                "p_value": float(p),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prior_dir", default="prior_analysis")
    ap.add_argument("--n_iter", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    chunks_path = os.path.join(args.prior_dir, "chunks_per_trajectory.csv")
    print(f"[load] {chunks_path}")
    chunks = pd.read_csv(chunks_path)
    chunks["is_connected"] = chunks["is_connected"].astype(bool).astype(float)

    print("[agg] per-(subject, task) feature means")
    per_st = _aggregate_per_subject_task(chunks)
    print(f"       n rows = {len(per_st)}  n subjects = {per_st['subject_id'].nunique()}")

    features = CHUNK_FEATURES + TRAJ_FEATURES
    print(f"[A] split-half reliability over {args.n_iter} iterations")
    reliability = _split_half_reliability(per_st, features,
                                          n_iter=args.n_iter, seed=args.seed)
    reliability = reliability.sort_values("mean_rho", ascending=False)
    print(reliability.to_string(index=False))
    reliability.to_csv(os.path.join(args.prior_dir,
                                    "individual_differences_reliability.csv"),
                       index=False)

    print("[profile] per-subject mean of each feature")
    profile = _per_subject_profile(per_st, features)
    profile.to_csv(os.path.join(args.prior_dir,
                                "individual_differences_subject_profile.csv"),
                   index=False)

    print("[load] subject-level performance (error + motor/cognitive)")
    perf = _load_performance(args.prior_dir)
    print(f"       n subjects with performance = {perf['subject_id'].nunique()}")

    perf_metrics = ["overall_error_rate", "te1_rate", "te2_rate", "te3_rate",
                    "motor_rate", "cognitive_rate",
                    "motor_over_wrong", "cognitive_over_wrong"]
    print("[B] chunking-style × performance correlations")
    corr = _style_vs_performance(profile, perf, features, perf_metrics)
    corr_pivot = corr.pivot(index="style", columns="performance", values="rho")
    print(corr_pivot.round(3).to_string())
    corr.to_csv(os.path.join(args.prior_dir,
                             "individual_differences_correlations.csv"),
                index=False)

    # Also dump the merged subject table for the collaborators.
    merged = profile.merge(perf, on="subject_id", how="inner")
    merged.to_csv(os.path.join(args.prior_dir,
                               "individual_differences_merged.csv"),
                  index=False)
    print(f"[done] wrote merged table: {len(merged)} subjects × "
          f"{merged.shape[1]} columns")


if __name__ == "__main__":
    main()

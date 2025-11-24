#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convergence and post-peak behaviour analysis.

Expected input CSV: one row per evaluation point, with at least:
    env, algo, run_id, step, eval_mean_reward
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = Path("results") / "all_eval_curves.csv"  # TODO: 修改成你的路径
OUT_DIR = Path("results") / "convergence_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_run_stats(
    run_df: pd.DataFrame,
    smooth_window: int = 5,
    final_fraction: float = 0.2,
) -> dict:
    """Compute convergence & post-peak stats for a single run."""
    run_df = run_df.sort_values("step").reset_index(drop=True)

    steps = run_df["step"].values
    rewards = run_df["eval_mean_reward"].values

    # Smooth the curve (simple moving average)
    rewards_smooth = (
        pd.Series(rewards)
        .rolling(window=smooth_window, min_periods=1)
        .mean()
        .values
    )

    # Peak performance and time
    R_max = rewards_smooth.max()
    idx_peak = rewards_smooth.argmax()
    t_peak = steps[idx_peak]

    # Final segment (last X% of evaluations)
    n = len(rewards_smooth)
    start_idx = int((1.0 - final_fraction) * n)
    final_segment = rewards_smooth[start_idx:]
    R_final = final_segment.mean()
    sigma_final = final_segment.std(ddof=0)

    # Post-peak drop
    eps = 1e-8
    drop_abs = R_max - R_final
    drop_rel = drop_abs / (abs(R_max) + eps)

    return dict(
        R_max=R_max,
        t_peak=t_peak,
        R_final=R_final,
        drop_abs=drop_abs,
        drop_rel=drop_rel,
        sigma_final=sigma_final,
        n_evals=n,
    )


def classify_run(row: pd.Series) -> str:
    """Assign qualitative convergence regime based on drop & variance."""
    drop_rel = row["drop_rel"]
    sigma_final = row["sigma_final"]
    R_final = row["R_final"]
    eps = 1e-8

    # Normalise sigma by |R_final| to get relative noise level
    noise_rel = sigma_final / (abs(R_final) + eps)

    if drop_rel < 0.05 and noise_rel < 0.05:
        return "stable"
    elif drop_rel < 0.2:
        # small / moderate drop, but either noisy or flat
        return "plateau"
    else:
        return "post_peak_degradation"


def main():
    df = pd.read_csv(DATA_PATH)

    required_cols = {"env", "algo", "run_id", "step", "eval_mean_reward"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

    stats = []
    for (env, algo, run_id), group in df.groupby(["env", "algo", "run_id"]):
        s = compute_run_stats(group)
        s.update(dict(env=env, algo=algo, run_id=run_id))
        stats.append(s)

    stats_df = pd.DataFrame(stats)
    stats_df["regime"] = stats_df.apply(classify_run, axis=1)

    stats_df.to_csv(OUT_DIR / "run_convergence_stats.csv", index=False)
    print(f"Saved per-run stats to {OUT_DIR / 'run_convergence_stats.csv'}")

    # ----- Plot 1: stacked bar (regime proportions) -----
    # Count runs per (env, algo, regime)
    counts = (
        stats_df.groupby(["env", "algo", "regime"])["run_id"]
        .nunique()
        .reset_index(name="num_runs")
    )

    # Convert to proportions per (env, algo)
    total = (
        counts.groupby(["env", "algo"])["num_runs"]
        .transform("sum")
    )
    counts["fraction"] = counts["num_runs"] / total

    pivot = counts.pivot_table(
        index=["env", "algo"],
        columns="regime",
        values="fraction",
        fill_value=0.0,
    )

    pivot.plot(kind="bar", stacked=True)
    plt.ylabel("Fraction of runs")
    plt.title("Convergence regimes per env–algo")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "regime_proportions_per_env_algo.png", dpi=200)
    plt.close()

    # ----- Plot 2: boxplot of post-peak drop -----
    plt.figure()
    # Build a label "env|algo" for x-axis
    stats_df["env_algo"] = stats_df["env"] + " | " + stats_df["algo"]
    stats_df.boxplot(column="drop_rel", by="env_algo")
    plt.ylabel("Relative post-peak drop")
    plt.title("Post-peak degradation per env–algo")
    plt.suptitle("")  # remove default super title
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "post_peak_drop_boxplot.png", dpi=200)
    plt.close()

    # ----- Plot 3 (optional): histogram of t_peak -----
    plt.figure()
    stats_df.boxplot(column="t_peak", by="env_algo")
    plt.ylabel("t_peak (timesteps)")
    plt.title("Peak time distribution per env–algo")
    plt.suptitle("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "t_peak_boxplot.png", dpi=200)
    plt.close()

    print("Plots saved to:", OUT_DIR)


if __name__ == "__main__":
    main()

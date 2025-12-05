#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot high-quality stacked bar chart for relaxed convergence fractions,
formatted for academic publication.

Reads:
    convergence_trend_fractions_relaxed.csv

Writes:
    relaxed_convergence_stacked_bars_clean.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
ANALYSIS_DIR = Path(
    "/homes/sohawan2/reinforcement-learning-thesis/thesis_project"
) / "experiment_analysis" / "convergence_analysis"

CSV_PATH = ANALYSIS_DIR / "convergence_trend_fractions_relaxed.csv"


def main():
    df = pd.read_csv(CSV_PATH)
    print("Loaded columns:", df.columns.tolist())

    # -----------------------------------------------------
    # Environment & algorithm order
    # (Hopper-v4 放最后)
    # -----------------------------------------------------
    env_order = [
        "CartPole-v1",
        "LunarLander-v3",
        "LunarLanderContinuous-v3",
        "Hopper-v4",
    ]
    algo_order = ["a2c", "ppo"]

    # 显示用的环境缩写
    env_short_map = {
        "CartPole-v1": "CP",
        "Hopper-v4": "HP",
        "LunarLander-v3": "LL",
        "LunarLanderContinuous-v3": "LLC",
    }

    labels = []
    conv_vals, post_vals, unc_vals = [], [], []

    for env in env_order:
        for algo in algo_order:
            row = df[(df["env"] == env) & (df["algo"] == algo)]
            if row.empty:
                continue

            r = row.iloc[0]
            short = env_short_map.get(env, env)

            labels.append(f"{short} ({algo.upper()})")
            conv_vals.append(r.get("converged_relaxed", 0.0))
            post_vals.append(r.get("post_peak_degradation", 0.0))
            unc_vals.append(r.get("uncertain", 0.0))

    conv_vals = np.array(conv_vals)
    post_vals = np.array(post_vals)
    unc_vals = np.array(unc_vals)
    x = np.arange(len(labels))

    # -----------------------------------------------------
    # Plot
    # -----------------------------------------------------
    plt.figure(figsize=(9, 5))

    # academic color palette
    color_conv = "#1f77b4"   # deep blue
    color_post = "#ff7f0e"   # deep orange
    color_unc  = "#7f7f7f"   # grey

    bar_width = 0.6

    plt.bar(
        x,
        conv_vals,
        width=bar_width,
        color=color_conv,
        label="Converged (relaxed)",
    )
    plt.bar(
        x,
        post_vals,
        width=bar_width,
        bottom=conv_vals,
        color=color_post,
        label="Post-peak degradation",
    )
    plt.bar(
        x,
        unc_vals,
        width=bar_width,
        bottom=conv_vals + post_vals,
        color=color_unc,
        label="Uncertain",
    )

    # 改成 CP–A2C 这样的标签
    short_labels = [lbl.replace("(", "–").replace(")", "") for lbl in labels]
    plt.xticks(x, short_labels, rotation=20, ha="right", fontsize=10)

    plt.ylabel("Fraction of runs", fontsize=11)
    plt.ylim(0.0, 1.0)

    plt.title(
        "Relaxed convergence classification across environments and algorithms",
        fontsize=12,
    )

    # 仅 y 方向虚线网格，便于读数
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # legend 在底部
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        fontsize=10,
        frameon=False,
    )

    plt.tight_layout()

    out_path = ANALYSIS_DIR / "relaxed_convergence_stacked_bars_clean.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved clean figure to: {out_path}")


if __name__ == "__main__":
    main()

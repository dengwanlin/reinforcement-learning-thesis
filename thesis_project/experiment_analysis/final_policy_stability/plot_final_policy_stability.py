#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot final-policy stability across environments and algorithms.

Input:
    final_policy_stability_seed0.csv
Outputs:
    final_policy_stability_bar_seed0.{pdf,png}
    final_policy_stability_box_seed0.{pdf,png}
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------- 全局路径 -----------------
BASE_DIR = Path(
    "/homes/sohawan2/reinforcement-learning-thesis/thesis_project/experiment_analysis/final_policy_stability"
)

CSV_PATH = BASE_DIR / "final_policy_stability_seed0.csv"


def plot_bar_chart(df: pd.DataFrame, metric: str = "var_final") -> None:
    """论文风格的 mean ± std 柱状图（使用 log-scale）。"""

    algo_order = ["a2c", "ppo"]
    df["algo"] = pd.Categorical(df["algo"], categories=algo_order, ordered=True)

    grouped = (
        df.groupby(["env", "algo"])[metric]
          .agg(["mean", "std", "count"])
          .reset_index()
          .sort_values(["env", "algo"])
    )

    envs = grouped["env"].unique()
    n_envs = len(envs)

    fig, ax = plt.subplots(figsize=(1.6 * n_envs + 1.5, 4.2))

    x_env = range(n_envs)
    bar_width = 0.35

    means_a2c, stds_a2c = [], []
    means_ppo, stds_ppo = [], []

    for env in envs:
        sub = grouped[grouped["env"] == env]
        row_a2c = sub[sub["algo"] == "a2c"]
        row_ppo = sub[sub["algo"] == "ppo"]

        means_a2c.append(row_a2c["mean"].values[0])
        stds_a2c.append(row_a2c["std"].values[0])

        means_ppo.append(row_ppo["mean"].values[0])
        stds_ppo.append(row_ppo["std"].values[0])

    x_a2c = [x - bar_width / 2 for x in x_env]
    x_ppo = [x + bar_width / 2 for x in x_env]

    ax.bar(x_a2c, means_a2c, width=bar_width, label="A2C")
    ax.bar(x_ppo, means_ppo, width=bar_width, label="PPO")

    ax.errorbar(x_a2c, means_a2c, yerr=stds_a2c, fmt="none", capsize=3)
    ax.errorbar(x_ppo, means_ppo, yerr=stds_ppo, fmt="none", capsize=3)

    ax.set_ylabel("Final-policy variance\n$\\mathrm{Var}_{\\mathrm{final}}$", fontsize=11)

    # 🔥🔥🔥 ADD THIS LINE (log scale)
    ax.set_yscale("log")

    ax.set_xticks(list(x_env))
    ax.set_xticklabels(envs, fontsize=10)
    ax.set_title("Final-policy stability, seed 0", fontsize=12)

    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()

    out_pdf = BASE_DIR / "final_policy_stability_bar_seed0.pdf"
    out_png = BASE_DIR / "final_policy_stability_bar_seed0.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=220)

    print(f"[OK] saved bar chart → {out_pdf}")
    print(f"[OK] saved bar chart → {out_png}")

    plt.close(fig)



def plot_boxplots(df: pd.DataFrame, metric: str = "var_final") -> None:
    """论文风格的按环境箱线图。"""

    algo_order = ["a2c", "ppo"]
    df["algo"] = pd.Categorical(df["algo"], categories=algo_order, ordered=True)

    envs = sorted(df["env"].unique())
    n_envs = len(envs)

    n_cols = min(n_envs, 2)
    n_rows = (n_envs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.0 * n_cols, 3.0 * n_rows),
        squeeze=False
    )

    for idx, env in enumerate(envs):
        r = idx // n_cols
        c = idx % n_cols
        ax = axes[r][c]

        sub = df[df["env"] == env]

        data, labels = [], []
        for algo in algo_order:
            vals = sub[sub["algo"] == algo][metric].dropna().values
            data.append(vals)
            labels.append(algo.upper())

        bp = ax.boxplot(
            data,
            showfliers=False,
            widths=0.6
        )
        for element in ["boxes", "whiskers", "caps", "medians"]:
            for line in bp[element]:
                line.set_linewidth(1.2)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel("$\\mathrm{Var}_{\\mathrm{final}}$", fontsize=10)
        ax.set_title(env, fontsize=11)

        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)

    for idx in range(len(envs), n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r][c].set_visible(False)

    fig.suptitle("Final-policy variance distribution per environment", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    out_pdf = BASE_DIR / "final_policy_stability_box_seed0.pdf"
    out_png = BASE_DIR / "final_policy_stability_box_seed0.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=200)

    print(f"[OK] saved boxplots → {out_pdf}")
    print(f"[OK] saved boxplots → {out_png}")

    plt.close(fig)


def main():
    df = pd.read_csv(CSV_PATH)

    # drop runs without valid var
    df = df.dropna(subset=["var_final"])

    plot_bar_chart(df)
    plot_boxplots(df)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot intra-run variance (late-phase) across environments and algorithms.

Outputs both PDF (for LaTeX) and PNG (for quick viewing).

Input:
    intra_run_variance_summary_seed0.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------- 全局路径 -----------------

BASE_DIR = Path(
    "/homes/sohawan2/reinforcement-learning-thesis/thesis_project/experiment_analysis/variance_within"
)
CSV_PATH = BASE_DIR / "intra_run_variance_summary_seed0.csv"


def plot_bar_chart(df: pd.DataFrame, metric: str = "sigma_intra_late") -> None:
    """论文风格的 mean±std 柱状图。"""

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

    # 画布稍宽一些以容纳所有 env
    fig, ax = plt.subplots(figsize=(1.6 * n_envs + 1.5, 4.2))

    # x 轴：让 env 在整数 0,1,2,... 的位置
    x_env = range(n_envs)
    bar_width = 0.35

    # 准备 mean/std 按 algo 分开
    means_a2c, stds_a2c = [], []
    means_ppo, stds_ppo = [], []

    for env in envs:
        sub = grouped[grouped["env"] == env]
        row_a2c = sub[sub["algo"] == "a2c"]
        row_ppo = sub[sub["algo"] == "ppo"]

        if not row_a2c.empty:
            means_a2c.append(row_a2c["mean"].values[0])
            stds_a2c.append(row_a2c["std"].values[0])
        else:
            means_a2c.append(0.0)
            stds_a2c.append(0.0)

        if not row_ppo.empty:
            means_ppo.append(row_ppo["mean"].values[0])
            stds_ppo.append(row_ppo["std"].values[0])
        else:
            means_ppo.append(0.0)
            stds_ppo.append(0.0)

    # A2C / PPO 两组柱，左右偏移
    x_a2c = [x - bar_width / 2 for x in x_env]
    x_ppo = [x + bar_width / 2 for x in x_env]

    # 柱：不设颜色参数，用默认配色，但有 legend
    a2c_bar = ax.bar(x_a2c, means_a2c, width=bar_width, label="A2C")
    ppo_bar = ax.bar(x_ppo, means_ppo, width=bar_width, label="PPO")

    # error bars
    ax.errorbar(
        x_a2c, means_a2c, yerr=stds_a2c,
        fmt="none", capsize=3
    )
    ax.errorbar(
        x_ppo, means_ppo, yerr=stds_ppo,
        fmt="none", capsize=3
    )

    ax.set_ylabel(
        "Late-phase intra-run variance\n"
        "$\\sigma_{\\mathrm{intra, late}}$",
        fontsize=11
    )
    ax.set_xticks(list(x_env))
    ax.set_xticklabels(envs, rotation=0, ha="center", fontsize=10)
    ax.set_title("Intra-run variance (late phase), seed 0", fontsize=12)

    ax.legend(frameon=False, fontsize=10)

    # 加细 y 轴网格，提高可读性
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_axisbelow(True)

    # 让图布局更紧凑
    fig.tight_layout()

    out_bar_pdf = BASE_DIR / "intra_run_variance_bar_seed0.pdf"
    out_bar_png = BASE_DIR / "intra_run_variance_bar_seed0.png"
    fig.savefig(out_bar_pdf, bbox_inches="tight")
    fig.savefig(out_bar_png, bbox_inches="tight", dpi=200)
    print(f"[OK] Saved bar chart to {out_bar_pdf}")
    print(f"[OK] Saved bar chart to {out_bar_png}")

    plt.close(fig)


def plot_boxplots(df: pd.DataFrame, metric: str = "sigma_intra_late") -> None:
    """论文风格的按环境箱线图，不显示离群点。"""

    algo_order = ["a2c", "ppo"]
    df["algo"] = pd.Categorical(df["algo"], categories=algo_order, ordered=True)

    envs = sorted(df["env"].unique())
    n_envs = len(envs)

    # 2 列布局
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
            if len(vals) == 0:
                continue
            data.append(vals)
            labels.append(algo.upper())

        if not data:
            ax.set_visible(False)
            continue

        # 箱线图：不显示离群点，线条稍粗一点
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
        ax.set_ylabel("$\\sigma_{\\mathrm{intra, late}}$", fontsize=10)
        ax.set_title(env, fontsize=11)

        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_axisbelow(True)

    # 隐藏多余子图
    for idx in range(len(envs), n_rows * n_cols):
        r = idx // n_cols
        c = idx % n_cols
        axes[r][c].set_visible(False)

    fig.suptitle(
        "Distribution of late-phase intra-run variance per environment",
        fontsize=12
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    out_box_pdf = BASE_DIR / "intra_run_variance_box_seed0.pdf"
    out_box_png = BASE_DIR / "intra_run_variance_box_seed0.png"
    fig.savefig(out_box_pdf, bbox_inches="tight")
    fig.savefig(out_box_png, bbox_inches="tight", dpi=200)
    print(f"[OK] Saved boxplots to {out_box_pdf}")
    print(f"[OK] Saved boxplots to {out_box_png}")

    plt.close(fig)


def main():
    df = pd.read_csv(CSV_PATH)

    # 只用有效值
    df = df.dropna(subset=["sigma_intra_late"])

    plot_bar_chart(df, metric="sigma_intra_late")
    plot_boxplots(df, metric="sigma_intra_late")


if __name__ == "__main__":
    main()

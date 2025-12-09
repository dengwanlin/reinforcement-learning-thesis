# hyperparameter_interaction_effects/select_representative_interactions.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import ANALYSIS_OUT_DIR


def load_summaries() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    读取全局 ANOVA 和回归 summary。
    需要 interaction_anova_summary.csv 和 interaction_regression_summary.csv
    已由 interaction_analysis.py 生成。
    """
    anova_path = ANALYSIS_OUT_DIR / "interaction_anova_summary.csv"
    regr_path = ANALYSIS_OUT_DIR / "interaction_regression_summary.csv"

    if not anova_path.exists():
        raise FileNotFoundError(f"ANOVA summary not found: {anova_path}")
    if not regr_path.exists():
        raise FileNotFoundError(f"Regression summary not found: {regr_path}")

    anova_df = pd.read_csv(anova_path)
    regr_df = pd.read_csv(regr_path)

    return anova_df, regr_df


def build_ranking_table(anova_df: pd.DataFrame, regr_df: pd.DataFrame) -> pd.DataFrame:
    """
    合并 ANOVA 和回归结果，并计算一个综合的“交互强度得分”。

    约定字段：
      - anova_df: env, algo, h1, h2, p, eta_sq_partial, ...
      - regr_df: env, algo, h1, h2, adj_R2_main, adj_R2_inter,
                 p_change, p_beta_inter, ...
    """

    # 只保留我们需要的列，避免合并时杂乱
    anova_cols_keep = [
        "env",
        "algo",
        "h1",
        "h2",
        "p",
        "eta_sq_partial",
        "F",
        "df_inter",
        "df_resid",
        "n",
    ]
    regr_cols_keep = [
        "env",
        "algo",
        "h1",
        "h2",
        "adj_R2_main",
        "adj_R2_inter",
        "AIC_main",
        "AIC_inter",
        "BIC_main",
        "BIC_inter",
        "beta_inter",
        "p_beta_inter",
        "F_change",
        "p_change",
        "used_log_h1",
        "used_log_h2",
        "n",
    ]

    anova_small = anova_df[anova_cols_keep].copy()
    regr_small = regr_df[regr_cols_keep].copy()

    merged = pd.merge(
        anova_small,
        regr_small,
        on=["env", "algo", "h1", "h2"],
        suffixes=("_anova", "_reg"),
        how="inner",
    )

    # 计算若干辅助指标
    merged["delta_adj_R2"] = merged["adj_R2_inter"] - merged["adj_R2_main"]

    # 显著性标记
    merged["sig_anova"] = merged["p"] < 0.05
    merged["sig_reg_F"] = merged["p_change"] < 0.05
    merged["sig_beta"] = merged["p_beta_inter"] < 0.05

    # 只看正向提升的 R^2
    merged["delta_adj_R2_pos"] = merged["delta_adj_R2"].clip(lower=0.0)

    # 综合得分：
    #   - ANOVA 的部分方差解释度（eta_sq_partial）
    #   - 回归里交互项带来的 R^2 提升（delta_adj_R2_pos）
    #   - 如果显著性 OK，再稍微加一点 bonus
    # 你可以后续根据论文需要调调整权重，这里给一个合理 heuristic。
    base_score = merged["eta_sq_partial"].fillna(0) * merged["delta_adj_R2_pos"].fillna(0)

    bonus = 0.0
    bonus += merged["sig_anova"].astype(float) * 0.1
    bonus += merged["sig_reg_F"].astype(float) * 0.05
    bonus += merged["sig_beta"].astype(float) * 0.05

    merged["interaction_score"] = base_score + bonus

    return merged


def attach_heatmap_path(df: pd.DataFrame) -> pd.DataFrame:
    """
    为每一行加上 heatmap PNG 的路径，方便你直接定位图像文件。

    路径规则：
      results/<env>/<algo>/heatmaps/heatmap_<h1>_<h2>.png
    """
    png_paths = []
    for _, row in df.iterrows():
        env = row["env"]
        algo = row["algo"]
        h1 = row["h1"]
        h2 = row["h2"]

        png_path = (
            ANALYSIS_OUT_DIR
            / env
            / algo
            / "heatmaps"
            / f"heatmap_{h1}_{h2}.png"
        )
        png_paths.append(str(png_path))

    df = df.copy()
    df["heatmap_png"] = png_paths
    return df


def select_top_interactions(
    rank_df: pd.DataFrame,
    top_k_per_env_algo: int = 2,
    min_eta_sq: float = 0.02,
    min_delta_adjR2: float = 0.01,
) -> pd.DataFrame:
    """
    按 env+algo 分组，从 rank_df 中挑选每个组最“代表性”的若干交互对。

    逻辑：
      1. 先对每个 env+algo 单独筛选：
         - eta_sq_partial >= min_eta_sq
         - delta_adj_R2 >= min_delta_adjR2
         - 至少满足 ANOVA 或回归里的一个显著性条件
      2. 如果该组在上述条件下一个都没有：
         - 退而求其次：该 env+algo 组内不做阈值过滤，
           直接按 interaction_score 和 eta_sq_partial 排序取 top K。
      3. 这样可以保证 “每个 env+algo 至少有一些代表性交互对”。

    排序：
      - 先按 interaction_score 从大到小
      - 再按 eta_sq_partial 排序兜底
    """
    df = rank_df.copy()

    # 统一做一次显著性标记，后面复用
    mask_sig_global = df["sig_anova"] | df["sig_reg_F"] | df["sig_beta"]

    selected_rows = []

    for (env, algo), group in df.groupby(["env", "algo"]):
        # 组内条件
        mask_effect = group["eta_sq_partial"].fillna(0) >= min_eta_sq
        mask_r2 = group["delta_adj_R2"].fillna(0) >= min_delta_adjR2
        mask_sig = mask_sig_global.loc[group.index]

        group_filtered = group[mask_effect & mask_r2 & mask_sig].copy()

        if group_filtered.empty:
            # 该 env+algo 在严格条件下没有通过的交互对，
            # 退一步：在该组内全部候选里选相对最好的。
            print(f"[select] No strong interactions for env={env}, algo={algo}, "
                  f"fallback to best-by-score.")
            group_candidate = group.copy()
        else:
            group_candidate = group_filtered

        group_sorted = group_candidate.sort_values(
            by=["interaction_score", "eta_sq_partial"],
            ascending=False,
        )

        top_group = group_sorted.head(top_k_per_env_algo)
        selected_rows.append(top_group)

    if not selected_rows:
        print("[select] No groups produced any selected interactions.")
        return pd.DataFrame()

    selected = pd.concat(selected_rows, ignore_index=True)
    selected = attach_heatmap_path(selected)

    return selected



def save_selection(selected: pd.DataFrame) -> None:
    """
    保存自动挑选出来的代表性交互：
      - 全局一个 CSV：selected_interactions.csv
      - 每个 env+algo 一个简短的 txt 列表，方便在论文里查阅
    """
    out_csv = ANALYSIS_OUT_DIR / "selected_interactions.csv"
    selected.to_csv(out_csv, index=False)
    print(f"[select] Saved global selection CSV -> {out_csv}")

    # 每个 env+algo 输出一个 txt
    for (env, algo), group in selected.groupby(["env", "algo"]):
        out_txt = ANALYSIS_OUT_DIR / env / algo / "selected_interactions.txt"
        lines = []
        lines.append(f"env = {env}, algo = {algo}\n")
        lines.append(f"{'-'*60}\n")
        for _, row in group.iterrows():
            line = (
                f"h1 = {row['h1']}, h2 = {row['h2']}\n"
                f"  interaction_score = {row['interaction_score']:.4f}\n"
                f"  eta_sq_partial    = {row['eta_sq_partial']:.4f}\n"
                f"  delta_adj_R2      = {row['delta_adj_R2']:.4f}\n"
                f"  p(ANOVA)          = {row['p']:.3e}\n"
                f"  p_change(F-test)  = {row['p_change']:.3e}\n"
                f"  p_beta_inter      = {row['p_beta_inter']:.3e}\n"
                f"  heatmap_png       = {row['heatmap_png']}\n\n"
            )
            lines.append(line)

        out_txt.parent.mkdir(parents=True, exist_ok=True)
        with out_txt.open("w") as f:
            f.writelines(lines)

        print(f"[select] Saved per-env/algo selection -> {out_txt}")


def main():
    anova_df, regr_df = load_summaries()
    rank_df = build_ranking_table(anova_df, regr_df)
    selected = select_top_interactions(rank_df, top_k_per_env_algo=2)
    if selected.empty:
        print("[select] No interactions selected. Try relaxing thresholds.")
    else:
        save_selection(selected)


if __name__ == "__main__":
    main()

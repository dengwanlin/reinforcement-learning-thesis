#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarise RELAXED convergence trend labels per environment and algorithm.

Reads:
    convergence_metrics_relaxed.csv  (from ANALYSIS_ROOT)

Writes:
    convergence_trend_counts_relaxed.csv
    convergence_trend_fractions_relaxed.csv
"""

import pandas as pd
from pathlib import Path

# 和 compute_convergence_metrics_relaxed.py 保持一致
ANALYSIS_ROOT = Path(
    "/homes/sohawan2/reinforcement-learning-thesis/thesis_project/experiment_analysis/convergence_analysis"
)
CSV_PATH = ANALYSIS_ROOT / "convergence_metrics_relaxed.csv"


def main():
    df = pd.read_csv(CSV_PATH)

    print("Columns:", df.columns.tolist())

    # 1) 计数：每个 (env, algo, trend_relaxed) 的 run 数
    counts = (
        df.groupby(["env", "algo", "trend_relaxed"])
          .size()
          .rename("count")
          .reset_index()
    )

    print("\n=== Counts per env & algo (relaxed) ===")
    print(
        counts.pivot_table(
            index=["env", "algo"],
            columns="trend_relaxed",
            values="count",
            fill_value=0,
        )
    )

    # 2) 各类在每个 (env, algo) 内的比例
    counts["fraction"] = (
        counts.groupby(["env", "algo"])["count"]
              .transform(lambda x: x / x.sum())
    )

    print("\n=== Fractions per env & algo (relaxed) ===")
    print(counts)

    # 3) 做一个 pivot 后的 fraction 表（便于进 LaTeX）
    frac_pivot = (
        counts.pivot_table(
            index=["env", "algo"],
            columns="trend_relaxed",
            values="fraction",
            fill_value=0.0,
        )
        .reset_index()
    )

    print("\n=== Fraction pivot (for LaTeX, relaxed) ===")
    print(frac_pivot.round(3))

    # 4) 保存到 CSV
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    out_counts = ANALYSIS_ROOT / "convergence_trend_counts_relaxed.csv"
    out_frac = ANALYSIS_ROOT / "convergence_trend_fractions_relaxed.csv"

    counts.to_csv(out_counts, index=False)
    frac_pivot.to_csv(out_frac, index=False)

    print(f"\nSaved relaxed counts   to {out_counts}")
    print(f"Saved relaxed fractions to {out_frac}")


if __name__ == "__main__":
    main()

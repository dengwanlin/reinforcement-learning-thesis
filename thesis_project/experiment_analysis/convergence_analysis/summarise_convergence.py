#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarise convergence categories (strict + relaxed) per environment and algorithm.

Reads:
    convergence_metrics.csv   (from ANALYSIS_DIR)

Writes:
    convergence_trend_counts_strict.csv
    convergence_trend_fractions_strict.csv
    convergence_trend_counts_relaxed.csv
    convergence_trend_fractions_relaxed.csv
"""

from pathlib import Path

import pandas as pd

ANALYSIS_DIR = Path(
    "/homes/sohawan2/reinforcement-learning-thesis/thesis_project"
) / "experiment_analysis" / "convergence_analysis"

CSV_PATH = ANALYSIS_DIR / "convergence_metrics.csv"


def summarise_one(
    df: pd.DataFrame,
    trend_col: str,
    counts_name: str,
    fractions_name: str,
) -> None:
    """
    Summarise a given trend column (trend_strict or trend_relaxed)
    into counts and fractions per (env, algo).
    """
    # 计数
    counts = (
        df.groupby(["env", "algo", trend_col])
          .size()
          .rename("count")
          .reset_index()
    )

    print(f"\n=== Counts per env & algo ({trend_col}) ===")
    print(
        counts.pivot_table(
            index=["env", "algo"],
            columns=trend_col,
            values="count",
            fill_value=0,
        )
    )

    # 归一化为比例
    counts["fraction"] = (
        counts.groupby(["env", "algo"])["count"]
              .transform(lambda x: x / x.sum())
    )

    print(f"\n=== Fractions per env & algo ({trend_col}) ===")
    print(counts)

    # pivot for LaTeX/table
    frac_pivot = (
        counts.pivot_table(
            index=["env", "algo"],
            columns=trend_col,
            values="fraction",
            fill_value=0.0,
        )
        .reset_index()
    )

    print(f"\n=== Fraction pivot (for LaTeX, {trend_col}) ===")
    print(frac_pivot.round(3))

    out_counts = ANALYSIS_DIR / counts_name
    out_frac = ANALYSIS_DIR / fractions_name

    counts.to_csv(out_counts, index=False)
    frac_pivot.to_csv(out_frac, index=False)

    print(f"\nSaved {trend_col} counts    to {out_counts}")
    print(f"Saved {trend_col} fractions to {out_frac}")


def main():
    df = pd.read_csv(CSV_PATH)
    print("Columns:", df.columns.tolist())

    # 严格版
    summarise_one(
        df,
        trend_col="trend_strict",
        counts_name="convergence_trend_counts_strict.csv",
        fractions_name="convergence_trend_fractions_strict.csv",
    )

    # 宽松版（过滤掉 not_computed）
    df_rel = df[df["trend_relaxed"] != "not_computed"].copy()
    if not df_rel.empty:
        summarise_one(
            df_rel,
            trend_col="trend_relaxed",
            counts_name="convergence_trend_counts_relaxed.csv",
            fractions_name="convergence_trend_fractions_relaxed.csv",
        )
    else:
        print("\n[WARN] No relaxed metrics available (trend_relaxed == 'not_computed' for all rows).")


if __name__ == "__main__":
    main()

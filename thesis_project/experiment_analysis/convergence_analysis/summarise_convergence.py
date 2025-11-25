#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Summarise convergence trend labels per environment and algorithm.

Reads:
    convergence_metrics.csv  (from ANALYSIS_ROOT)

Writes:
    convergence_trend_counts.csv
    convergence_trend_fractions.csv
"""

import pandas as pd
from pathlib import Path

# Paths (align with compute_convergence_metrics.py)
ANALYSIS_ROOT = Path(
    "/homes/sohawan2/reinforcement-learning-thesis/thesis_project/experiment_analysis/convergence_analysis"
)
CSV_PATH = ANALYSIS_ROOT / "convergence_metrics.csv"


def main():
    df = pd.read_csv(CSV_PATH)

    print("Columns:", df.columns.tolist())

    # 1) Counts: number of runs for each (env, algo, trend_label)
    counts = (
        df.groupby(["env", "algo", "trend_label"])
          .size()
          .rename("count")
          .reset_index()
    )

    print("\n=== Counts per env & algo ===")
    print(
        counts.pivot_table(
            index=["env", "algo"],
            columns="trend_label",
            values="count",
            fill_value=0,
        )
    )

    # 2) Fractions: within each (env, algo), normalise counts to sum to 1
    counts["fraction"] = (
        counts.groupby(["env", "algo"])["count"]
              .transform(lambda x: x / x.sum())
    )

    print("\n=== Fractions per env & algo ===")
    print(counts)

    # 3) Pivoted fraction table (nice for LaTeX)
    frac_pivot = (
        counts.pivot_table(
            index=["env", "algo"],
            columns="trend_label",
            values="fraction",
            fill_value=0.0,
        )
        .reset_index()
    )

    print("\n=== Fraction pivot (for LaTeX) ===")
    print(frac_pivot.round(3))

    # Save to CSV in analysis dir
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    out_counts = ANALYSIS_ROOT / "convergence_trend_counts.csv"
    out_frac = ANALYSIS_ROOT / "convergence_trend_fractions.csv"

    counts.to_csv(out_counts, index=False)
    frac_pivot.to_csv(out_frac, index=False)

    print(f"\nSaved counts to   {out_counts}")
    print(f"Saved fractions to {out_frac}")


if __name__ == "__main__":
    main()

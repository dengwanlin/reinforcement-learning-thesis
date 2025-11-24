#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_pipeline.py

一键执行第 5.3 节相关的分析流程：

(1) 从 thesis_project/runs 中所有实验日志计算 "T_to_target" 指标；
(2) 在上一步结果基础上，统计每个 env–algo 的最快 run 及其超参数。

在本目录下运行：
    (rl) python run_pipeline.py
"""

from pathlib import Path
import metric_fast_learning
import analyze_learning_speed


def main():
    this_file = Path(__file__).resolve()
    # 目录结构：
    #   .../reinforcement-learning-thesis/thesis_project/experiment_analysis/fastest_learning_speed/run_pipeline.py
    # parents[0] = fastest_learning_speed
    # parents[1] = experiment_analysis
    # parents[2] = thesis_project        ✅
    thesis_project_root = this_file.parents[2]
    runs_root = thesis_project_root / "runs"
    out_dir = this_file.parent                      # -> fastest_learning_speed

    runs_csv = out_dir / "fastest_learning_runs_full.csv"
    summary_csv = out_dir / "fastest_learning_by_env_algo.csv"

    print("=== [Step 1] Compute T_to_target for all runs ===")
    print(f"    Runs root   : {runs_root}")
    print(f"    Output CSV  : {runs_csv}")
    metric_fast_learning.compute_all_runs(runs_root, runs_csv)

    print("\n=== [Step 2] Analyze fastest runs per env–algo ===")
    print(f"    Input CSV   : {runs_csv}")
    print(f"    Output CSV  : {summary_csv}")
    analyze_learning_speed.summarize_fastest(runs_csv, summary_csv)

    print("\n=== Pipeline finished ===")
    print(f"All runs metrics   -> {runs_csv}")
    print(f"Fastest per config -> {summary_csv}")


if __name__ == "__main__":
    main()

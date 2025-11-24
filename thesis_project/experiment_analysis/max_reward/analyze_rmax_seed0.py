#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_rmax_seed0.py

扫描 runs/ 目录中所有算法-环境的 run（仅 seed == 0），
为每个 run 计算最大评估奖励 R_max，并提取完整超参数。

输出：
  - rmax_seed0_runs_full.csv
      每行：一个 run 的 (env, algo, run_id, seed, R_max, t_at_max, 所有超参)
  - rmax_seed0_summary_by_env_algo.csv
      对每个 (env, algo) 的 R_max 统计（count / mean / max / min）
"""

from pathlib import Path
import pandas as pd

from metric_rmax import compute_rmax_for_run
from extract_hyperparams import extract_hyperparams


def discover_runs(runs_root: Path):
    """
    递归发现 runs_root 下的所有 run 目录。
    结构假定为：runs/<env>/<algo>/<run_id>/
    每个 run_dir 下至少包含 config.yml 和 eval/evaluations.npz（如果评估过）。
    """
    for env_dir in runs_root.iterdir():
        if not env_dir.is_dir():
            continue
        for algo_dir in env_dir.iterdir():
            if not algo_dir.is_dir():
                continue
            for run_dir in algo_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                yield env_dir.name, algo_dir.name, run_dir


def main():
    script_path = Path(__file__).resolve()
    # project_root = thesis_project/
    project_root = script_path.parents[2]
    runs_root = project_root / "runs"
    out_dir = project_root / "experiment_analysis" / "max_reward"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] project_root = {project_root}")
    print(f"[INFO] runs_root    = {runs_root}")
    print(f"[INFO] out_dir      = {out_dir}")

    rows = []

    for env, algo, run_dir in discover_runs(runs_root):
        config_path = run_dir / "config.yml"
        eval_path = run_dir / "eval" / "evaluations.npz"

        # 提取种子和全部超参数
        seed, hparams = extract_hyperparams(config_path)
        if seed is None:
            continue

        # 只分析 seed == 0
        if seed != 0:
            continue

        # 计算 R_max
        res = compute_rmax_for_run(eval_path)
        if res is None:
            # 没有 eval 文件或格式有问题的 run 直接跳过
            continue
        R_max, t_at_max = res

        row = {
            "env": env,
            "algo": algo,
            "run_id": run_dir.name,
            "seed": seed,
            "R_max": R_max,
            "t_at_max": t_at_max,
        }
        # 加上所有超参
        row.update(hparams)

        rows.append(row)

    if not rows:
        print("[WARN] No runs found for seed == 0 with valid eval files.")
        return

    # 所有 run 的完整记录
    df = pd.DataFrame(rows)
    full_csv = out_dir / "rmax_seed0_runs_full.csv"
    df.to_csv(full_csv, index=False)
    print(f"[INFO] Saved full hyperparams to {full_csv}")
    print(df.head())

    # 按环境+算法做一个 summary，用于 baseline 部分
    summary = (
        df.groupby(["env", "algo"])["R_max"]
        .agg(["count", "mean", "max", "min"])
        .reset_index()
    )
    summary = summary.rename(
        columns={
            "count": "num_runs",
            "mean": "R_max_mean",
            "max": "R_max_max",
            "min": "R_max_min",
        }
    )
    summary_csv = out_dir / "rmax_seed0_summary_by_env_algo.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"[INFO] Saved summary to {summary_csv}")
    print(summary)


if __name__ == "__main__":
    main()


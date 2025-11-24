#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_learning_speed.py

基于 metric_fast_learning.py 生成的 fastest_learning_runs_full.csv，
统计每个 env–algo 的最快 run 及其超参数。

生成：
  - fastest_learning_by_env_algo.csv
并在终端打印：
  - 每个 env–algo 的成功率
  - 每个 env–algo 的最快 run 及其超参数（展开后的列）
"""

from pathlib import Path
import argparse
import json
import pandas as pd


def _process_hyperparams_json(hp_json: str) -> dict:
    """
    将 hyperparams_json 解析为扁平 dict：
    - 对于标量类型（int/float/bool/str），直接保留；
    - 对于 dict/list 等复杂对象，用 JSON 字符串形式保留（例如 policy_kwargs）。
    """
    if not isinstance(hp_json, str) or not hp_json:
        return {}

    try:
        hp = json.loads(hp_json)
    except Exception:
        return {}

    if not isinstance(hp, dict):
        # 极端情况：不是 dict，就整个转字符串
        return {"hyperparams_raw": json.dumps(hp, ensure_ascii=False)}

    flat = {}
    for k, v in hp.items():
        if isinstance(v, (int, float, bool, str)) or v is None:
            flat[k] = v
        else:
            # 嵌套结构统一存 json 字符串
            flat[k] = json.dumps(v, ensure_ascii=False)
    return flat


def summarize_fastest(input_csv: Path, output_csv: Path):
    """
    读取 input_csv，输出每个 env–algo 的最快 run 到 output_csv，
    并将超参数展开成独立的列。
    """
    if not input_csv.exists():
        print(f"[ERROR] Input CSV not found: {input_csv}")
        return

    df = pd.read_csv(input_csv)

    print("=== Success rate (per env–algo) ===")
    success_stats = (
        df.groupby(["env", "algo"])["success"]
        .agg(["mean", "sum", "count"])
        .rename(columns={"mean": "success_rate", "sum": "num_success", "count": "num_total"})
    )
    print(success_stats)
    print()

    df_success = df[df["success"] == True].copy()
    if df_success.empty:
        print("[WARN] No successful runs at all. Nothing to analyze.")
        return

    # 找每个 env–algo 下 T_to_target 最小的那一行
    idx_min = (
        df_success.groupby(["env", "algo"])["T_to_target"]
        .idxmin()
        .dropna()
        .astype(int)
    )

    best_rows = df_success.loc[idx_min].copy()

    # 先保留基础信息列
    base_cols = [
        "env", "algo", "T_to_target", "run_dir", "seed",
        "total_timesteps", "n_envs", "hyperparams_json",
    ]
    base_cols = [c for c in base_cols if c in best_rows.columns]
    best_rows = best_rows[base_cols]
    best_rows = best_rows.rename(columns={"T_to_target": "best_T_to_target"})

    # ✅ 重置索引，保证后面和 hp_df 行对得上
    best_rows = best_rows.reset_index(drop=True)

    # === 展开 hyperparams_json 成多个列 ===
    hp_dicts = []
    for _, row in best_rows.iterrows():
        hp_json = row.get("hyperparams_json", "")
        hp_flat = _process_hyperparams_json(hp_json)
        hp_dicts.append(hp_flat)

    hp_df = pd.DataFrame(hp_dicts)

    # 合并基础列 + 超参数列（此时两边 index 都是 0..N-1）
    best_expanded = pd.concat(
        [best_rows.drop(columns=["hyperparams_json"]), hp_df],
        axis=1,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    best_expanded.to_csv(output_csv, index=False)
    print(f"[INFO] Saved fastest runs summary (with hyperparams) to {output_csv}")

    # 终端打印一个简要版本，方便 eyeball 检查
    print("\n=== Fastest configs per env–algo ===")
    for _, row in best_expanded.iterrows():
        env = row["env"]
        algo = row["algo"]
        T = row["best_T_to_target"]
        run_dir = row["run_dir"]
        seed = row.get("seed", None)

        # 把所有超参数字段收集一下（排除基础信息列）
        exclude = {
            "env", "algo", "best_T_to_target", "run_dir", "seed",
            "total_timesteps", "n_envs",
        }
        hp_view = {
            k: row[k]
            for k in best_expanded.columns
            if k not in exclude and pd.notna(row[k])
        }

        print(f"\n[{env} / {algo}]")
        print(f"  fastest T_to_target = {T}")
        print(f"  run_dir             = {run_dir}")
        print(f"  seed                = {seed}")
        print(f"  hyperparams:")
        for k, v in hp_view.items():
            print(f"    - {k}: {v}")


def main():
    this_file = Path(__file__).resolve()
    out_dir = this_file.parent
    default_input = out_dir / "fastest_learning_runs_full.csv"
    default_output = out_dir / "fastest_learning_by_env_algo.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        default=str(default_input),
        help="metric_fast_learning 生成的 CSV（默认同目录下的 fastest_learning_runs_full.csv）",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=str(default_output),
        help="输出：每个 env–algo 的最快 run 概要 CSV（含展开的超参数列）",
    )
    args = parser.parse_args()

    summarize_fastest(Path(args.input_csv), Path(args.output_csv))


if __name__ == "__main__":
    main()

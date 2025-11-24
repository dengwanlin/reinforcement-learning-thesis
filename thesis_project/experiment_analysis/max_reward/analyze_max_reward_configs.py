#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_max_reward_configs.py

基于 rmax_seed0_runs_full.csv 做两类分析：

1) 对 CartPole-v1：
      统计所有达到 ~500 reward 的超参数取值频率
      输出：cartpole_max500_hparam_freq.csv

2) 对其它环境：
      找到 R_max 最大的所有 run（可能有多个并列）
      输出：other_envs_best_configs.csv
"""

from pathlib import Path
import pandas as pd


def load_full_csv(csv_path):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def is_cartpole_max500(R):
    return R >= 499.9  # 防止浮点误差


def analyze_cartpole(df, out_dir):
    df_cp = df[df["env"] == "CartPole-v1"]

    # 所有达到500 reward 的 run
    df_success = df_cp[df_cp["R_max"].apply(is_cartpole_max500)]
    num_success = len(df_success)
    print(f"[INFO] CartPole-v1: {len(df_cp)} total, {num_success} reach ~500 reward.")

    cols_hps = [c for c in df.columns
                if c not in ["env", "algo", "run_id", "seed", "R_max", "t_at_max"]]

    rows = []

    for hp in cols_hps:
        counts = df_success.groupby("algo")[hp].value_counts().reset_index(name="count")

        for _, row in counts.iterrows():
            rows.append({
                "env": "CartPole-v1",
                "algo": row["algo"],
                "hyperparam": hp,
                "value": row[hp],
                "count": row["count"],
                "freq": row["count"] / df_success[df_success["algo"] == row["algo"]].shape[0],
                "num_success_configs": df_success[df_success["algo"] == row["algo"]].shape[0],
            })

    out_csv = out_dir / "cartpole_max500_hparam_freq.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[INFO] Saved CartPole hyperparam frequencies to: {out_csv}")


def analyze_other_envs(df, out_dir):
    df_other = df[df["env"] != "CartPole-v1"]

    rows = []
    for (env, algo), group in df_other.groupby(["env", "algo"]):
        max_r = group["R_max"].max()
        winners = group[group["R_max"] == max_r]

        print(f"[INFO] {env} - {algo}: max R_max = {max_r:.3f}, {len(winners)} configs achieve this value.")

        for _, row in winners.iterrows():
            rows.append(row.to_dict())

    out_csv = out_dir / "other_envs_best_configs.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[INFO] Saved best configs for non-CartPole envs to: {out_csv}")


def main():
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    out_dir = project_root / "experiment_analysis" / "max_reward"
    out_dir.mkdir(parents=True, exist_ok=True)

    full_csv = out_dir / "rmax_seed0_runs_full.csv"
    df = load_full_csv(full_csv)

    analyze_cartpole(df, out_dir)
    analyze_other_envs(df, out_dir)


if __name__ == "__main__":
    main()

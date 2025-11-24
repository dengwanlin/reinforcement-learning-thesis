#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
metric_fast_learning.py

计算每个 run 第一次达到目标奖励所需的 environment steps (T_to_target)，
并输出一个汇总 CSV：fastest_learning_runs_full.csv

目录结构假设为：
  thesis_project/
    ├── runs/
    │   └── <ENV>/
    │       └── <ALGO>/
    │           └── <RUN_ID>/
    │               ├── config.yml
    │               └── eval/
    │                   └── evaluations.npz
    └── experiment_analysis/
        └── fastest_learning_speed/
            └── metric_fast_learning.py

既可以直接运行：
    python metric_fast_learning.py
也可以在 run_pipeline.py 中以函数方式调用 compute_all_runs()。
"""

from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import json
import argparse


# === 目标奖励阈值：基于你刚刚确认的四个环境 ===
TARGET_REWARDS = {
    "CartPole-v1": 475.0,
    "LunarLander-v3": 200.0,
    "LunarLanderContinuous-v3": 200.0,
    "Hopper-v4": 2000.0,
}


def parse_seed_from_dirname(dirname: str):
    """
    尝试从 run 目录名中解析 seed，比如:
        '20251107_171248_728087_pid2961473_seed0' -> 0
    解析失败则返回 None。
    """
    parts = dirname.split("_")
    for p in parts:
        if p.startswith("seed"):
            try:
                return int(p.replace("seed", ""))
            except ValueError:
                pass
    return None


def load_config(run_dir: Path) -> dict:
    """
    读取每个 run 下的 config.yml（如果有），失败则返回 {}。
    """
    cfg_path = run_dir / "config.yml"
    if not cfg_path.exists():
        return {}

    try:
        with cfg_path.open("r") as f:
            cfg = yaml.safe_load(f)
        return cfg or {}
    except Exception as e:
        print(f"[WARN] Failed to load config.yml for {run_dir}: {e}")
        return {}


def compute_T_for_run(eval_file: Path):
    """
    对单个 run（通过其 eval/evaluations.npz）计算：
        env, algo, run_dir, T_to_target, success, meta_info_dict

    目录结构：
      runs/<ENV>/<ALGO>/<RUN_ID>/eval/evaluations.npz
    """
    # eval_file: .../runs/<ENV>/<ALGO>/<RUN_ID>/eval/evaluations.npz
    eval_dir = eval_file.parent               # eval/
    run_dir = eval_dir.parent                 # <RUN_ID>/
    algo_dir = run_dir.parent                 # <ALGO>/
    env_dir = algo_dir.parent                 # <ENV>/

    algo = algo_dir.name
    env = env_dir.name

    # 只关心你论文里的四个环境
    if env not in TARGET_REWARDS:
        # 其它环境直接跳过
        # print(f"[INFO] env={env} not in TARGET_REWARDS, skip {run_dir}")
        return None

    target = TARGET_REWARDS[env]

    # 载入 evaluations.npz
    data = np.load(eval_file)
    timesteps = data["timesteps"]   # shape: (n_eval,)
    results = data["results"]       # shape: (n_eval, n_eval_episodes)

    # 计算每次 evaluation 的平均 reward
    mean_rewards = results.mean(axis=1)

    # 找第一次 >= target 的 index
    indices = np.where(mean_rewards >= target)[0]
    if len(indices) > 0:
        first_idx = int(indices[0])
        T = int(timesteps[first_idx])
        success = True
    else:
        # 没达到阈值，把 T 设为最后一次 evaluation 的 timestep
        T = int(timesteps[-1])
        success = False

    # seed 从目录名解析
    seed = parse_seed_from_dirname(run_dir.name)

    # 读取 config.yml，提取超参数等信息
    cfg = load_config(run_dir)

    # ✅ 优先用已解析好的 hyperparams_parsed
    hyperparams = cfg.get("hyperparams_parsed")
    if hyperparams is None:
        # 兜底：尝试 hyperparams / hyperparams_raw
        hyperparams = cfg.get("hyperparams")
        if hyperparams is None:
            hyperparams = cfg.get("hyperparams_raw", {})

    total_timesteps = cfg.get("n_timesteps") or cfg.get("total_timesteps")
    n_envs = cfg.get("n_envs", None)

    meta = {
        "seed": seed,
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        # 把超参数整体转成 JSON 字符串，后面再展开
        "hyperparams_json": json.dumps(hyperparams, ensure_ascii=False),
    }

    return env, algo, run_dir, T, success, meta


def compute_all_runs(runs_root: Path, output_csv: Path):
    """
    核心函数：给定 runs 根目录和输出 CSV 路径，完成全部计算。
    供 run_pipeline.py 调用。

    期待的结构：
      runs/<ENV>/<ALGO>/<RUN_ID>/eval/evaluations.npz
    """
    records = []

    for eval_file in runs_root.rglob("evaluations.npz"):
        result = compute_T_for_run(eval_file)
        if result is None:
            continue
        env, algo, run_dir, T, success, meta = result
        rec = {
            "env": env,
            "algo": algo,
            "run_dir": str(run_dir),
            "T_to_target": T,
            "success": success,
        }
        rec.update(meta)
        records.append(rec)

    if not records:
        print("[ERROR] No valid runs found. Check runs_root and TARGET_REWARDS.")
        return

    df = pd.DataFrame(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Saved {len(df)} records to {output_csv}")


def main():
    """
    standalone 模式：python metric_fast_learning.py 也能跑。
    默认自动推断 runs_root 和 output_csv。
    """
    # 当前文件路径：
    #   .../reinforcement-learning-thesis/thesis_project/experiment_analysis/fastest_learning_speed/metric_fast_learning.py
    this_file = Path(__file__).resolve()
    # parents[0] = fastest_learning_speed
    # parents[1] = experiment_analysis
    # parents[2] = thesis_project          ✅
    thesis_project_root = this_file.parents[2]
    runs_root = thesis_project_root / "runs"
    out_dir = this_file.parent
    output_csv = out_dir / "fastest_learning_runs_full.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, default=str(runs_root),
        help="实验 runs 根目录（默认 <thesis_project>/runs）",
    )
    parser.add_argument(
        "--output_csv", type=str, default=str(output_csv),
        help="输出 CSV 路径",
    )
    args = parser.parse_args()

    compute_all_runs(Path(args.root), Path(args.output_csv))


if __name__ == "__main__":
    main()

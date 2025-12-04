#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Baseline overview plots for default RL Baselines3 Zoo runs (single or multiple seeds).

Current usage:
- For each environment, plot PPO and A2C curves on the same figure.
- Right now only seed0 is available, so we plot a single curve per algorithm.
- If you later add runs_seed1, runs_seed2, the script will automatically:
  - draw all seed curves per algorithm
  - and add a dashed "mean over seeds" curve.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# ------------------------------
# User configuration
# ------------------------------

# 根目录：key 是 seed 标签，value 是该 seed 的日志目录
RUN_BASE_DIRS: Dict[str, str] = {
    "seed0": "/homes/sohawan2/reinforcement-learning-thesis/thesis_project/runs_seed0",
    # 将来如果有更多 seeds 可以加：
    # "seed1": "/homes/sohawan2/reinforcement-learning-thesis/thesis_project/runs_seed1",
    # "seed2": "/homes/sohawan2/reinforcement-learning-thesis/thesis_project/runs_seed2",
}

# 四个环境
ENVIRONMENTS: List[str] = [
    "CartPole-v1",
    "LunarLander-v3",
    "LunarLanderContinuous-v3",
    "Hopper-v4",
]

# 两个算法
ALGORITHMS: List[str] = ["ppo", "a2c"]

# 输出图片目录（相对于本脚本所在目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")


# ------------------------------
# Helper functions
# ------------------------------

def find_latest_run_dir(base_dir: str, env: str, algo: str) -> str:
    """
    在 base_dir/env/algo 下找到最新的那个 run 子目录（按名字排序后取最后一个）。
    返回该 run 的完整路径；如果没有找到，返回空字符串。
    """
    target_dir = os.path.join(base_dir, env, algo)
    if not os.path.isdir(target_dir):
        return ""

    subdirs = [
        os.path.join(target_dir, d)
        for d in os.listdir(target_dir)
        if os.path.isdir(os.path.join(target_dir, d))
    ]
    if not subdirs:
        return ""

    subdirs.sort()
    return subdirs[-1]


def load_evaluation_curve(eval_npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 evaluations.npz 中读取 timesteps 和 mean reward 曲线。

    SB3 evaluations.npz 通常包含:
      - 'timesteps': shape (n_eval,)
      - 'results'  : shape (n_eval, n_envs)
    我们对每次 eval 的 results 在 env 维度上求平均。
    """
    data = np.load(eval_npz_path)
    timesteps = data["timesteps"].squeeze()
    results = data["results"]
    rewards = results.mean(axis=1)
    return timesteps, rewards


def collect_curves_for_env_algo(
    env: str, algo: str
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    对于 (env, algo)，从所有 seed base dir 中收集曲线。
    返回字典: seed_label -> (timesteps, rewards)
    """
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for seed_label, base_dir in RUN_BASE_DIRS.items():
        run_dir = find_latest_run_dir(base_dir, env, algo)
        if not run_dir:
            print(f"[WARN] No run dir for {env} / {algo} / {seed_label}")
            continue

        eval_npz = os.path.join(run_dir, "eval", "evaluations.npz")
        if not os.path.isfile(eval_npz):
            print(f"[WARN] Missing evaluations.npz: {eval_npz}")
            continue

        try:
            ts, rew = load_evaluation_curve(eval_npz)
            curves[seed_label] = (ts, rew)
            print(f"[INFO] Loaded {env} / {algo} / {seed_label} from {eval_npz}")
        except Exception as e:
            print(f"[ERROR] Failed to load {eval_npz}: {e}")

    return curves


def compute_mean_curve(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算跨 seed 的平均曲线。
    只有当 seeds >= 2 时才有意义；调用方自己控制是否使用。
    """
    if len(curves) < 2:
        return np.array([]), np.array([])

    lengths = [len(v[0]) for v in curves.values()]
    min_len = min(lengths)

    # 用第一个 seed 的 timesteps 作为基准
    first_seed = next(iter(curves.keys()))
    base_ts = curves[first_seed][0][:min_len]

    all_rewards = []
    for _, (ts, r) in curves.items():
        all_rewards.append(r[:min_len])
    all_rewards = np.stack(all_rewards, axis=0)  # (n_seeds, min_len)

    mean_rewards = all_rewards.mean(axis=0)
    return base_ts, mean_rewards


def plot_env_with_algorithms(env: str) -> None:
    """
    为单个 environment 生成一张图：
    - x 轴 timesteps
    - y 轴 evaluation reward
    - 每个算法画一条或多条 seed 曲线
    - 若某算法有多个 seed，则再画一条 dashed 的 mean 曲线
    """
    plt.figure(figsize=(6, 4))

    for algo in ALGORITHMS:
        curves = collect_curves_for_env_algo(env, algo)
        if not curves:
            print(f"[WARN] No curves for {env} / {algo}, skip this algo.")
            continue

        # 画每个 seed 的曲线
        for seed_label, (ts, r) in curves.items():
            plt.plot(
                ts,
                r,
                alpha=0.9,
                linewidth=1.8,
                label=f"{algo.upper()} ({seed_label})",
            )

        # 如果该算法有多个 seed，可以加一条平均曲线
        ts_mean, r_mean = compute_mean_curve(curves)
        if ts_mean.size > 0:
            plt.plot(
                ts_mean,
                r_mean,
                linewidth=2.2,
                linestyle="--",
                label=f"{algo.upper()} (mean over seeds)",
            )

    plt.xlabel("Timesteps")
    plt.ylabel("Evaluation reward")
    plt.title(f"Baseline performance on {env}")
    plt.legend()
    plt.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    env_safe = env.replace("-", "_")
    out_path = os.path.join(FIG_DIR, f"baseline_{env_safe}_seed0.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[INFO] Saved figure: {out_path}")


# ------------------------------
# Main
# ------------------------------

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"[INFO] Saving figures to: {FIG_DIR}")

    for env in ENVIRONMENTS:
        print(f"[INFO] Processing environment: {env}")
        plot_env_with_algorithms(env)


if __name__ == "__main__":
    main()

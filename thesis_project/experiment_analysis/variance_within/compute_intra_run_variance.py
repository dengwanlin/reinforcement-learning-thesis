#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute intra-run variance (variance within a single training run)
based on monitor.csv episode rewards.

Directory layout assumed:

/homes/sohawan2/reinforcement-learning-thesis/thesis_project/
    runs_seed0/
        <ENV>/
            <ALGO>/
                <RUN_ID>/
                    monitor.csv
                    results.json
                    ...

This script will scan all runs and write a summary CSV with one row per run:
    intra_run_variance_summary.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json


# ---------- 0. 配置你的路径 ----------

RUNS_ROOT = Path(
    "/homes/sohawan2/reinforcement-learning-thesis/thesis_project/runs_seed0"
)

OUT_DIR = Path(
    "/homes/sohawan2/reinforcement-learning-thesis/thesis_project/experiment_analysis/variance_within"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- 1. 读 monitor.csv，提取 episode rewards ----------

def load_episode_rewards(monitor_path: Path) -> np.ndarray:
    """
    Load per-episode rewards from a Stable-Baselines-style monitor.csv file.
    Returns a 1D numpy array of rewards.

    We only use the root-level monitor.csv (training episodes), not eval/monitor.csv.
    """
    if not monitor_path.exists():
        raise FileNotFoundError(f"monitor.csv not found at {monitor_path}")

    rewards = []
    with monitor_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 跳过注释行和 JSON header 行（以 "#" 开头）
            if line.startswith("#"):
                continue

            parts = line.split(",")
            # 跳过 header 行 "r,l,t"
            if parts[0] == "r":
                continue

            try:
                r = float(parts[0])
                rewards.append(r)
            except ValueError:
                # 出现奇怪行就直接忽略
                continue

    return np.array(rewards, dtype=np.float32)


# ---------- 2. 计算 intra-run variance（全程 + 后期） ----------

def compute_intra_run_variance(
    rewards: np.ndarray,
    late_fraction: float = 0.5
) -> dict:
    """
    Compute intra-run variance metrics from a sequence of episode rewards.

    Returns:
        {
          "sigma_intra_full": float,
          "sigma_intra_late": float,
          "n_episodes": int
        }

    This corresponds to Section 4.3.5 (Variance Within a Run),
    with an additional late-phase variant for analysis in Chapter 5.
    """
    n_eps = int(len(rewards))
    if n_eps == 0:
        return {
            "sigma_intra_full": np.nan,
            "sigma_intra_late": np.nan,
            "n_episodes": 0,
        }

    sigma_full = float(rewards.std())

    # late-phase episodes: e.g. last 50% of training
    late_fraction = max(0.0, min(1.0, late_fraction))
    start_idx = int(n_eps * (1.0 - late_fraction))
    if start_idx >= n_eps:
        sigma_late = np.nan
    else:
        sigma_late = float(rewards[start_idx:].std())

    return {
        "sigma_intra_full": sigma_full,
        "sigma_intra_late": sigma_late,
        "n_episodes": n_eps,
    }


# ---------- 3. 遍历所有 runs ----------

def iter_runs(root: Path):
    """
    Yield (env_id, algo, run_dir) for all runs in the result directory.

    Assumes: <ROOT>/<ENV>/<ALGO>/<RUN_ID>/
    We don't hard-code env names here; just scan whatever is under runs_seed0.
    """
    if not root.exists():
        raise FileNotFoundError(f"RUNS_ROOT does not exist: {root}")

    for env_dir in sorted(root.iterdir()):
        if not env_dir.is_dir():
            continue
        env_id = env_dir.name
        for algo_dir in sorted(env_dir.iterdir()):
            if not algo_dir.is_dir():
                continue
            algo = algo_dir.name
            for run_dir in sorted(algo_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                yield env_id, algo, run_dir


# ---------- 4. 主流程 ----------

def main():
    rows = []

    for env_id, algo, run_dir in iter_runs(RUNS_ROOT):
        monitor_path = run_dir / "monitor.csv"  # 注意：根目录的 monitor.csv
        if not monitor_path.exists():
            print(f"[WARN] no monitor.csv in {run_dir}")
            continue

        try:
            rewards = load_episode_rewards(monitor_path)
        except Exception as e:
            print(f"[WARN] failed to load {monitor_path}: {e}")
            continue

        metrics = compute_intra_run_variance(
            rewards,
            late_fraction=0.5,  # 这里用 “最后 50% episodes”，你可以改
        )

        row = {
            "env": env_id,
            "algo": algo,
            "run_dir": str(run_dir),
            "n_episodes": metrics["n_episodes"],
            "sigma_intra_full": metrics["sigma_intra_full"],
            "sigma_intra_late": metrics["sigma_intra_late"],
        }

        # 可选：从 results.json 中顺手读一点超参数做后续分析
        results_json = run_dir / "results.json"
        if results_json.exists():
            try:
                with results_json.open("r") as f:
                    meta = json.load(f)
                hps = meta.get("hyperparams", {})
                # 下面根据你自己的结构挑你关心的
                lr = hps.get("learning_rate", None)
                ent_coef = hps.get("ent_coef", None)
                n_steps = hps.get("n_steps", None)
                row["learning_rate"] = lr
                row["ent_coef"] = ent_coef
                row["n_steps"] = n_steps
            except Exception as e:
                print(f"[WARN] failed to parse {results_json}: {e}")

        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.head())

    out_path = OUT_DIR / "intra_run_variance_summary_seed0.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] Saved summary to {out_path}")


if __name__ == "__main__":
    main()

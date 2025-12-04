#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute final-policy stability (variance of final evaluation returns)
based on evaluations.npz, following Section 4.3.6.

Directory layout assumed:

/homes/sohawan2/reinforcement-learning-thesis/thesis_project/
    runs_seed0/
        <ENV>/
            <ALGO>/
                <RUN_ID>/
                    eval/
                        evaluations.npz
                    results.json
                    ...

This script scans all runs and writes a summary CSV with one row per run:
    final_policy_stability_seed0.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json


# ---------- 0. 路径 & 超参数 ----------

RUNS_ROOT = Path(
    "/homes/sohawan2/reinforcement-learning-thesis/thesis_project/runs_seed0"
)

OUT_DIR = Path(
    "/homes/sohawan2/reinforcement-learning-thesis/thesis_project/experiment_analysis/final_policy_stability"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 使用最后 K 次 evaluation 来评估最终策略的稳定性
K_EVALS = 5  # 你可以改成 10 等


# ---------- 1. 遍历所有 runs ----------

def iter_runs(root: Path):
    """
    Yield (env_id, algo, run_dir) for all runs in the result directory.

    Assumes directory layout:
        <ROOT>/<ENV>/<ALGO>/<RUN_ID>/
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


# ---------- 2. 从 evaluations.npz 计算 final-policy stability ----------

def compute_final_policy_stability(eval_path: Path, k_evals: int = K_EVALS) -> dict:
    """
    Compute final-policy stability metrics from a Stable-Baselines3 evaluations.npz file.

    Following 4.3.6:
      - Take the last K evaluations
      - Flatten all episode returns into a single vector
      - Compute variance of these returns

    Returns:
        {
          "n_evals": int,
          "n_final_returns": int,
          "var_final": float,
          "mean_final": float,
          "range_final": float,
        }
    """
    if not eval_path.exists():
        raise FileNotFoundError(f"evaluations.npz not found at {eval_path}")

    data = np.load(eval_path)

    # SB3: "results" usually contains evaluation returns, shape (n_evals, n_eval_episodes)
    if "results" not in data:
        raise KeyError(f"'results' not found in {eval_path}")

    results = np.asarray(data["results"])  # shape: (n_evals, n_eval_episodes)
    n_evals = results.shape[0]

    if n_evals == 0:
        return {
            "n_evals": 0,
            "n_final_returns": 0,
            "var_final": np.nan,
            "mean_final": np.nan,
            "range_final": np.nan,
        }

    # 取最后 K 次 evaluation
    k = min(k_evals, n_evals)
    final_block = results[-k:, :]  # shape: (k, n_eval_episodes)
    final_returns = final_block.reshape(-1)  # 展平

    if final_returns.size == 0:
        return {
            "n_evals": n_evals,
            "n_final_returns": 0,
            "var_final": np.nan,
            "mean_final": np.nan,
            "range_final": np.nan,
        }

    var_final = float(np.var(final_returns))     # 与 4.3.6 一致：variance
    mean_final = float(np.mean(final_returns))
    range_final = float(np.max(final_returns) - np.min(final_returns))

    return {
        "n_evals": n_evals,
        "n_final_returns": int(final_returns.size),
        "var_final": var_final,
        "mean_final": mean_final,
        "range_final": range_final,
    }


# ---------- 3. 主流程 ----------

def main():
    rows = []

    for env_id, algo, run_dir in iter_runs(RUNS_ROOT):
        eval_path = run_dir / "eval" / "evaluations.npz"
        if not eval_path.exists():
            print(f"[WARN] no evaluations.npz in {run_dir}")
            continue

        try:
            metrics = compute_final_policy_stability(eval_path, k_evals=K_EVALS)
        except Exception as e:
            print(f"[WARN] failed to process {eval_path}: {e}")
            continue

        row = {
            "env": env_id,
            "algo": algo,
            "run_dir": str(run_dir),
            "n_evals": metrics["n_evals"],
            "n_final_returns": metrics["n_final_returns"],
            "var_final": metrics["var_final"],
            "mean_final": metrics["mean_final"],
            "range_final": metrics["range_final"],
        }

        # 可选：从 results.json 把一些超参数一起抓进来，方便后面关联分析
        results_json = run_dir / "results.json"
        if results_json.exists():
            try:
                with results_json.open("r") as f:
                    meta = json.load(f)
                hps = meta.get("hyperparams", {})
                row["learning_rate"] = hps.get("learning_rate")
                row["ent_coef"] = hps.get("ent_coef")
                row["n_steps"] = hps.get("n_steps")
            except Exception as e:
                print(f"[WARN] failed to parse {results_json}: {e}")

        rows.append(row)

    df = pd.DataFrame(rows)
    print(df.head())

    out_path = OUT_DIR / "final_policy_stability_seed0.csv"
    df.to_csv(out_path, index=False)
    print(f"[OK] Saved summary to {out_path}")


if __name__ == "__main__":
    main()

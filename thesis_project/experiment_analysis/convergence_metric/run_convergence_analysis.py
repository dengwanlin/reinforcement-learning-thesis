#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from compute_convergence_metrics import compute_convergence_stats


ROOT = Path("/homes/sohawan2/reinforcement-learning-thesis/thesis_project/runs_seed0")
OUT_DIR = Path("./results_convergence")
OUT_DIR.mkdir(exist_ok=True, parents=True)

def load_eval_npz(path: Path):
    """Load SB3 evaluations.npz (robust to single-eval runs)."""
    data = np.load(path)

    # 保证至少是一维
    timesteps = np.atleast_1d(data["timesteps"])
    results = np.array(data["results"])

    # 结果维度：
    # (n_eval, n_envs) -> 沿 env 取平均
    # (n_eval,)        -> 直接用
    # ()               -> 单个标量，包装成长度 1
    if results.ndim == 2:
        rewards = results.mean(axis=1)
    elif results.ndim == 1:
        rewards = results
    elif results.ndim == 0:
        rewards = np.array([results.item()])
    else:
        raise ValueError(f"Unexpected results.ndim={results.ndim} in {path}")

    return timesteps.astype(float), rewards.astype(float)



def main():
    records = []

    # traverse: <env>/<algo>/<run_id>/eval/evaluations.npz
    for env_dir in tqdm(list(ROOT.iterdir()), desc="env"):
        if not env_dir.is_dir():
            continue

        env_name = env_dir.name

        for algo_dir in env_dir.iterdir():
            if not algo_dir.is_dir():
                continue

            algo_name = algo_dir.name

            for run_dir in algo_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                eval_path = run_dir / "eval" / "evaluations.npz"
                if not eval_path.exists():
                    continue

                try:
                    t, r = load_eval_npz(eval_path)
                except Exception as e:
                    print(f"Failed loading: {eval_path}, error: {e}")
                    continue

                stats = compute_convergence_stats(t, r)
                stats.update(dict(
                    env=env_name,
                    algo=algo_name,
                    run_id=run_dir.name
                ))

                records.append(stats)

    # save results
    df = pd.DataFrame(records)
    out_csv = OUT_DIR / "convergence_stats.csv"
    df.to_csv(out_csv, index=False)

    print(f"Saved convergence_stats to {out_csv}")

    # Optional: Add categorisation for “stable / plateau / degrade”
    df["regime"] = df.apply(classify_regime, axis=1)
    df.to_csv(OUT_DIR / "convergence_stats_with_regime.csv", index=False)


def classify_regime(row):
    """Simple classification; tune thresholds later."""
    drop = row["delta_post"]
    slope = row["s_final"]
    noise = row["sigma_final"]

    # stable: small drop + small slope + low noise
    if drop > -0.05 * abs(row["R_max"]) and abs(slope) < 0.01 and noise < 0.1 * abs(row["R_end"]):
        return "stable"

    # degrade: final reward much lower than peak
    if drop < -0.2 * abs(row["R_max"]):
        return "post_peak_degrade"

    return "plateau"


if __name__ == "__main__":
    main()

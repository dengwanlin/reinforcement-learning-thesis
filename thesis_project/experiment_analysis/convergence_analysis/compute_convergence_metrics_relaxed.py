#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute relaxed convergence metrics for all runs.

This version is more tolerant than the strict RQ3 metric:
- R_max = 95th percentile instead of global maximum
- R_end = mean of last 5 raw eval returns
- slope computed over at least 10 points
- thresholds relaxed (10% tolerance for delta_post, 2% for slope)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

# -------------------------------
# Configuration
# -------------------------------
ROOT = Path("/homes/sohawan2/reinforcement-learning-thesis/thesis_project/runs_seed0")
EVAL_REL_PATH = Path("eval/evaluations.npz")

SMOOTH_WINDOW = 5
END_WINDOW_RELAXED = 5       # last 5 raw evals
FINAL_SLOPE_MIN_K = 10       # at least 10 points
FINAL_SLOPE_FRAC = 0.10      # or last 10%
DELTA_TOL_FRAC = 0.10        # relaxed: allow 10% drop
SLOPE_TOL_FRAC = 0.02        # relaxed: allow |s_final| <= 2% R_max


# -------------------------------
# Helpers
# -------------------------------

def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def load_eval(evals_path: Path) -> Optional[Dict[str, Any]]:
    if not evals_path.is_file():
        return None
    data = np.load(evals_path, allow_pickle=True)
    timesteps = np.atleast_1d(data["timesteps"])

    if "results" in data.files:
        results = np.asarray(data["results"])
        if results.ndim == 2:
            mean_returns = results.mean(axis=1)
        else:
            mean_returns = results
    else:
        mean_returns = np.atleast_1d(data["mean_returns"])

    return {"timesteps": timesteps, "mean_returns": mean_returns}


# -------------------------------
# Relaxed metric
# -------------------------------

def compute_relaxed_metrics(timesteps, mean_returns):
    n = len(mean_returns)
    if n < 5:
        raise ValueError("Not enough evals for relaxed metrics (need >=5)")

    smooth = moving_average(mean_returns, SMOOTH_WINDOW)

    # ---- relaxed R_max: 95% quantile of smooth curve
    R_max = float(np.quantile(smooth, 0.95))

    # ---- relaxed R_end: last 5 raw evals
    k_end = min(END_WINDOW_RELAXED, n)
    R_end = float(mean_returns[-k_end:].mean())

    delta_post = R_end - R_max

    # ---- relaxed slope: use >=10 points or last 10% of smoothed curve
    k = max(int(np.ceil(FINAL_SLOPE_FRAC * n)), FINAL_SLOPE_MIN_K)
    k = min(k, n)  # cannot exceed total count
    segment = smooth[-k:]
    diffs = np.diff(segment)
    s_final = float(diffs.mean())

    # ---- thresholds
    delta_eps = -DELTA_TOL_FRAC * abs(R_max)
    slope_eps = SLOPE_TOL_FRAC * abs(R_max)

    if (delta_post >= delta_eps) and (abs(s_final) <= slope_eps):
        trend = "converged_relaxed"
    elif delta_post < delta_eps:
        trend = "post_peak_degradation"
    else:
        trend = "uncertain"

    return dict(
        r_max_relaxed=R_max,
        r_end_relaxed=R_end,
        delta_post_relaxed=delta_post,
        s_final_relaxed=s_final,
        delta_eps_relaxed=delta_eps,
        slope_eps_relaxed=slope_eps,
        trend_relaxed=trend,
    )


# -------------------------------
# Iterate all runs
# -------------------------------

def parse_run_id(run_dir: Path):
    name = run_dir.name
    seed = None
    if "seed" in name:
        try:
            seed = int(name.split("seed")[-1])
        except:
            seed = None
    return {"run_id": name, "seed": seed}


def iterate_all_runs(root: Path) -> pd.DataFrame:
    rows = []

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

                eval_file = run_dir / EVAL_REL_PATH
                eval_data = load_eval(eval_file)
                if eval_data is None:
                    continue

                try:
                    metrics = compute_relaxed_metrics(
                        eval_data["timesteps"],
                        eval_data["mean_returns"],
                    )
                except ValueError as e:
                    print(f"[SKIP] {run_dir}: {e}")
                    continue

                idinfo = parse_run_id(run_dir)
                row = {
                    "env": env_id,
                    "algo": algo,
                    **idinfo,
                    **metrics,
                }
                rows.append(row)

    return pd.DataFrame(rows)


# -------------------------------
# Main
# -------------------------------

def main():
    df = iterate_all_runs(ROOT)
    out = ROOT.parent / "experiment_analysis" / "convergence_analysis" / "convergence_metrics_relaxed.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved relaxed metrics → {out}")
    print(df.head())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute convergence and post-peak behaviour metrics for all runs.

Implements the definitions from Section 4.3.3 (RQ3):

- Smooth evaluation curve via moving average
- R_max        : maximum of smoothed returns
- R_end        : mean of last N smoothed evaluations
- delta_post   : R_end - R_max
- s_final      : average slope over the last 10% of evaluations
- post_degrade : whether learning degrades after the peak
- trend_label  : coarse classification of convergence behaviour

Outputs:
    convergence_metrics.csv  (in ANALYSIS_ROOT)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

# =========================
# Paths (ADAPT TO YOUR PROJECT)
# =========================

# Raw runs root
RUNS_ROOT = Path(
    "/homes/sohawan2/reinforcement-learning-thesis/thesis_project/runs_seed0"
)

# Where to store analysis outputs (this script's directory)
ANALYSIS_ROOT = Path(
    "/homes/sohawan2/reinforcement-learning-thesis/thesis_project/experiment_analysis/convergence_analysis"
)

# Relative path to evaluations.npz inside each run directory
EVAL_REL_PATH = Path("eval/evaluations.npz")

# =========================
# Configuration
# =========================

# Moving-average window for smoothing evaluation returns
SMOOTH_WINDOW = 5

# Number of evaluations used to estimate R_end
END_WINDOW = 10

# Fraction of evaluations used for final-phase slope (last 10%)
FINAL_SLOPE_FRAC = 0.10

# Threshold for classifying post-peak degradation:
# we tolerate a small drop, e.g. 5% of |R_max|
DELTA_EPS_FRAC = 0.05

# Threshold for deciding whether the final slope is "flat" (converged) or not.
# 使用 reward 尺度自适应的斜率阈值：每一次 evaluation 允许 1% |R_max| 的变化。
SLOPE_EPS_FRAC = 0.01


# =========================
# Helper functions
# =========================

def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """
    Simple moving average with 'same' length output.
    """
    if window <= 1:
        return x.copy()

    # Use convolution; mode='same' keeps array length unchanged
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="same")


def load_mean_returns(evals_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load timesteps and mean evaluation returns from evaluations.npz.

    Compatible with:
        - 'results': shape (N,) or (N, n_eval_episodes)
        - 'mean_returns': already shape (N,)
    """
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
    elif "mean_returns" in data.files:
        mean_returns = np.atleast_1d(data["mean_returns"])
    else:
        raise KeyError(
            f"{evals_path}: cannot find 'results' or 'mean_returns'"
        )

    assert timesteps.shape[0] == mean_returns.shape[0], \
        f"timesteps and returns length mismatch in {evals_path}"

    return {
        "timesteps": timesteps,
        "mean_returns": mean_returns,
    }


def compute_convergence_metrics(
    timesteps: np.ndarray,
    mean_returns: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute R_max, R_end, delta_post and s_final for a single run,
    plus classification flags.

    Formulas follow Section 4.3.3:

        smooth_returns = moving_average(mean_returns)
        R_max  = max(smooth_returns)
        R_end  = mean(last END_WINDOW of smooth_returns)
        delta_post = R_end - R_max

        s_final = average increment over the last 10% of evaluations

    Additional:
        trend_label in {"converged", "post_peak_degradation",
                        "still_improving", "uncertain"}
    """
    n = len(timesteps)
    if n < 2:
        raise ValueError("Not enough evaluation points to compute metrics")

    # --------- smoothing ----------
    smooth = moving_average(mean_returns, SMOOTH_WINDOW)

    # --------- R_max and peak time ----------
    max_idx = int(np.argmax(smooth))
    r_max = float(smooth[max_idx])
    t_peak = int(timesteps[max_idx])

    # --------- R_end ----------
    end_w = min(END_WINDOW, n)
    r_end = float(smooth[-end_w:].mean())

    # --------- post-peak variation ----------
    delta_post = float(r_end - r_max)

    # --------- final-phase slope s_final ----------
    # 使用最后 max(ceil(10% * n), 2) 个点，确保至少 2 个点。
    k = max(int(np.ceil(FINAL_SLOPE_FRAC * n)), 2)
    k = min(k, n)  # 不能超过总长度
    segment = smooth[-k:]

    if len(segment) >= 2:
        diffs = np.diff(segment)
        if len(diffs) > 0:
            s_final = float(diffs.mean())  # average increment per evaluation
        else:
            s_final = 0.0
    else:
        s_final = 0.0  # too few points to estimate slope reliably

    # --------- classification: does reward go down? ----------
    # positive r_max scale; avoid division by 0 for tiny rewards
    scale = max(abs(r_max), 1.0)

    # tolerance for delta_post (e.g. allow 5% drop)
    eps_delta = -DELTA_EPS_FRAC * scale   # e.g., -5% * |R_max|
    # tolerance for slope being considered "flat"
    eps_slope = SLOPE_EPS_FRAC * scale    # e.g., 1% * |R_max| per eval

    # If delta_post >= eps_delta -> final performance is close to peak
    # If delta_post <  eps_delta -> significant post-peak degradation
    post_degrade = bool(delta_post < eps_delta)

    # --------- high-level trend label ----------
    if (delta_post >= eps_delta) and (abs(s_final) <= eps_slope):
        trend_label = "converged"
    elif post_degrade and (s_final <= eps_slope):
        trend_label = "post_peak_degradation"
    elif (not post_degrade) and (s_final > eps_slope):
        trend_label = "still_improving"
    else:
        trend_label = "uncertain"

    return {
        "r_max": r_max,
        "t_peak": t_peak,
        "r_end": r_end,
        "delta_post": delta_post,
        "s_final": s_final,
        "post_degrade": post_degrade,
        "delta_eps": eps_delta,
        "slope_eps": eps_slope,
        "trend_label": trend_label,
    }


def parse_run_id(run_dir: Path) -> Dict[str, Any]:
    """
    Try to extract seed from run directory name, assuming pattern ..._seedX.
    """
    name = run_dir.name
    seed = None
    if "seed" in name:
        try:
            seed_str = name.split("seed")[-1]
            seed = int(seed_str)
        except ValueError:
            seed = None
    return {"run_id": name, "seed": seed}


def iterate_all_runs(root: Path) -> pd.DataFrame:
    """
    Iterate over RUNS_ROOT/<ENV>/<ALGO>/<RUN_ID>/eval/evaluations.npz
    and collect convergence metrics for each run.
    """
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

                eval_path = run_dir / EVAL_REL_PATH
                eval_data = load_mean_returns(eval_path)
                if eval_data is None:
                    continue

                try:
                    metrics = compute_convergence_metrics(
                        timesteps=eval_data["timesteps"],
                        mean_returns=eval_data["mean_returns"],
                    )
                except ValueError as e:
                    print(f"[SKIP] {run_dir}: {e}")
                    continue

                id_info = parse_run_id(run_dir)

                row = {
                    "env": env_id,
                    "algo": algo,
                    **id_info,
                    **metrics,
                }
                rows.append(row)

    return pd.DataFrame(rows)


def main():
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    df = iterate_all_runs(RUNS_ROOT)
    out_path = ANALYSIS_ROOT / "convergence_metrics.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved convergence metrics to: {out_path}")
    print(df.head())


if __name__ == "__main__":
    main()

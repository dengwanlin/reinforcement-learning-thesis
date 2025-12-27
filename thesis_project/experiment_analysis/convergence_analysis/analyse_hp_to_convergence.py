#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analyse hyperparameters -> relaxed convergence.

Inputs:
  - hp_table.csv (from extract_hp_table.py)
  - convergence_metrics.csv (from compute_convergence_metrics.py)

Outputs:
  - hp_convergence_1d_rates.csv
  - hp_convergence_top_features.csv
  - optional plots per env/algo/hp

Main goal (thesis-friendly):
  "Converged rate by HP value" per environment and algorithm.

We define:
  y = 1 if trend_relaxed == 'converged_relaxed' else 0
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ANALYSIS_DIR = Path("/homes/sohawan2/reinforcement-learning-thesis/thesis_project/experiment_analysis/convergence_analysis")
HP_TABLE = ANALYSIS_DIR / "hp_table.csv"
CONV_TABLE = ANALYSIS_DIR / "convergence_metrics.csv"

# --- choose which HP keys to analyse (edit if needed) ---
# Tip: keep it to the "grid-search HPs", not logging paths etc.
DEFAULT_HP_KEYS = [
    # common
    "learning_rate",
    "n_steps",
    "gamma",
    "gae_lambda",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "policy_kwargs.net_arch",   # might exist in your config.yml
    # PPO-only (will be NaN for A2C)
    "batch_size",
    "n_epochs",
    "clip_range",
    "target_kl",
]


def pick_existing_hp_keys(df: pd.DataFrame, hp_keys: List[str]) -> List[str]:
    existing = [k for k in hp_keys if k in df.columns]
    # also accept partial matches if your keys are prefixed (depends on YAML structure)
    if not existing:
        # try find best matches by suffix
        for k in hp_keys:
            matches = [c for c in df.columns if c.endswith(k)]
            if matches:
                existing.append(matches[0])
    return list(dict.fromkeys(existing))


def make_binary_target(df: pd.DataFrame) -> pd.Series:
    return (df["trend_relaxed"] == "converged_relaxed").astype(int)


def normalise_value(v: Any) -> Any:
    """
    Make HP values stable for grouping:
    - floats -> rounded scientific-ish if very small
    - lists in JSON -> keep string
    """
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        # keep small floats readable (learning rates etc.)
        if v != 0 and (abs(v) < 1e-2 or abs(v) >= 1e3):
            return f"{v:.0e}"
        return float(round(v, 6))
    return v


def compute_1d_rates(df: pd.DataFrame, hp_cols: List[str]) -> pd.DataFrame:
    """
    For each env/algo and each hp_col value:
      rate = mean(y)
      n = count
    """
    rows = []
    y = make_binary_target(df)

    for hp in hp_cols:
        tmp = df[["env", "algo", hp]].copy()
        tmp["y"] = y.values
        tmp["hp_value"] = tmp[hp].apply(normalise_value)

        g = tmp.dropna(subset=["hp_value"]).groupby(["env", "algo", "hp_value"], dropna=True)
        agg = g["y"].agg(["count", "mean"]).reset_index()
        agg["hp"] = hp
        agg.rename(columns={"mean": "converged_rate", "count": "n"}, inplace=True)
        rows.append(agg)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return out[["env", "algo", "hp", "hp_value", "n", "converged_rate"]]


def compute_top_features(rate_df: pd.DataFrame, min_group_n: int = 50) -> pd.DataFrame:
    """
    Produce a simple "importance" proxy that is thesis-friendly:

    For each (env, algo, hp):
      lift = max(converged_rate) - min(converged_rate) over hp values
      (only considering hp values with n >= min_group_n)

    This is NOT a causal claim; it's a descriptive driver.
    """
    rows = []
    for (env, algo, hp), sub in rate_df.groupby(["env", "algo", "hp"]):
        sub2 = sub[sub["n"] >= min_group_n].copy()
        if sub2.empty:
            continue
        mx = float(sub2["converged_rate"].max())
        mn = float(sub2["converged_rate"].min())
        lift = mx - mn
        rows.append({
            "env": env,
            "algo": algo,
            "hp": hp,
            "lift": lift,
            "max_rate": mx,
            "min_rate": mn,
            "n_values_kept": int(sub2.shape[0]),
        })
    out = pd.DataFrame(rows).sort_values(["env", "algo", "lift"], ascending=[True, True, False])
    return out


def plot_1d_bar(rate_df: pd.DataFrame, env: str, algo: str, hp: str, out_dir: Path, min_group_n: int = 50):
    """
    Plot converged_rate by hp_value (bars). Only keep groups with n>=min_group_n.
    """
    sub = rate_df[(rate_df["env"] == env) & (rate_df["algo"] == algo) & (rate_df["hp"] == hp)].copy()
    sub = sub[sub["n"] >= min_group_n]
    if sub.empty:
        return

    # sort: numeric hp_value first if possible
    def sort_key(x):
        try:
            return float(str(x).replace("e", "E"))
        except Exception:
            return str(x)

    sub["sort_key"] = sub["hp_value"].apply(sort_key)
    sub = sub.sort_values("sort_key")

    xlabels = sub["hp_value"].astype(str).tolist()
    vals = sub["converged_rate"].values

    plt.figure(figsize=(7, 4))
    plt.bar(np.arange(len(vals)), vals)
    plt.xticks(np.arange(len(vals)), xlabels, rotation=25, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Converged rate (relaxed)")
    plt.title(f"{env} — {algo.upper()} — {hp}")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hp1d_{env}_{algo}_{hp.replace('.', '_')}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_2d_heatmap(df: pd.DataFrame, env: str, algo: str, hp_x: str, hp_y: str, out_dir: Path, min_cell_n: int = 30):
    """
    2D interaction: convergence rate over hp_x × hp_y.
    Good for Hopper: learning_rate × n_steps.

    We keep cells with at least min_cell_n runs.
    """
    sub = df[(df["env"] == env) & (df["algo"] == algo)].copy()
    if hp_x not in sub.columns or hp_y not in sub.columns:
        return

    sub["y"] = make_binary_target(sub)
    sub = sub.dropna(subset=[hp_x, hp_y])

    sub["xv"] = sub[hp_x].apply(normalise_value)
    sub["yv"] = sub[hp_y].apply(normalise_value)

    pivot_rate = sub.pivot_table(index="yv", columns="xv", values="y", aggfunc="mean")
    pivot_n = sub.pivot_table(index="yv", columns="xv", values="y", aggfunc="count")

    # mask low-count cells
    rate = pivot_rate.copy()
    rate[pivot_n < min_cell_n] = np.nan

    if rate.dropna(how="all").empty:
        return

    plt.figure(figsize=(7, 5))
    plt.imshow(rate.values, aspect="auto", interpolation="nearest")
    plt.colorbar(label="Converged rate (relaxed)")

    plt.xticks(np.arange(rate.shape[1]), [str(c) for c in rate.columns], rotation=25, ha="right")
    plt.yticks(np.arange(rate.shape[0]), [str(i) for i in rate.index])

    plt.xlabel(hp_x)
    plt.ylabel(hp_y)
    plt.title(f"{env} — {algo.upper()} — interaction: {hp_x} × {hp_y}")

    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hp2d_{env}_{algo}_{hp_x.replace('.', '_')}_vs_{hp_y.replace('.', '_')}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    if not HP_TABLE.is_file():
        raise FileNotFoundError(f"Missing {HP_TABLE}. Run extract_hp_table.py first.")
    if not CONV_TABLE.is_file():
        raise FileNotFoundError(f"Missing {CONV_TABLE}. Run compute_convergence_metrics.py first.")

    hp = pd.read_csv(HP_TABLE)
    conv = pd.read_csv(CONV_TABLE)

    # join on (env, algo, run_id)
    df = conv.merge(hp, on=["env", "algo", "run_id"], how="left", suffixes=("", "_hp"))
    if df["trend_relaxed"].isna().any():
        df = df.dropna(subset=["trend_relaxed"])

    # pick HP columns that exist in your hp_table
    hp_cols = pick_existing_hp_keys(df, DEFAULT_HP_KEYS)
    if not hp_cols:
        # fallback: auto-detect numeric HP-like columns (avoid id columns)
        id_cols = {"env", "algo", "run_id", "seed"}
        candidates = [c for c in df.columns if c not in id_cols and df[c].notna().any()]
        # keep numeric-ish
        hp_cols = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c]) or df[c].astype(str).str.contains(r"e-|\[", regex=True).any()]

    print("Using HP columns:", hp_cols)

    # 1D rates
    rate_df = compute_1d_rates(df, hp_cols)
    out_rate = ANALYSIS_DIR / "hp_convergence_1d_rates.csv"
    rate_df.to_csv(out_rate, index=False)
    print("Saved 1D rates to:", out_rate)

    # top features
    top_df = compute_top_features(rate_df, min_group_n=50)
    out_top = ANALYSIS_DIR / "hp_convergence_top_features.csv"
    top_df.to_csv(out_top, index=False)
    print("Saved top features to:", out_top)

    # plots (optional but very useful for 5.4 appendix or short subsection)
    plot_dir = ANALYSIS_DIR / "hp_convergence_plots"
    envs = sorted(df["env"].unique().tolist())
    algos = sorted(df["algo"].unique().tolist())

    # Plot only top-3 HP per env/algo (keeps it small & thesis-friendly)
    for env in envs:
        for algo in algos:
            sub_top = top_df[(top_df["env"] == env) & (top_df["algo"] == algo)].head(3)
            for _, r in sub_top.iterrows():
                plot_1d_bar(rate_df, env, algo, r["hp"], plot_dir, min_group_n=50)

            # Special: Hopper interaction plot (if columns exist)
            if env == "Hopper-v4":
                # try both canonical names and suffix matches
                candidates = {c: c for c in df.columns}
                lr_col = next((c for c in df.columns if c.endswith("learning_rate") or c == "learning_rate"), None)
                ns_col = next((c for c in df.columns if c.endswith("n_steps") or c == "n_steps"), None)
                if lr_col and ns_col:
                    plot_2d_heatmap(df, env, algo, lr_col, ns_col, plot_dir, min_cell_n=30)

    print("Saved plots to:", plot_dir)


if __name__ == "__main__":
    main()

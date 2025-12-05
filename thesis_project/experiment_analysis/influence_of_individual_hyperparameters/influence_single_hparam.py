import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import kruskal
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess

# -------------------
# Paths
# -------------------
ROOT = Path("/homes/sohawan2/reinforcement-learning-thesis/thesis_project")
RUNS = ROOT / "runs_seed0"
OUT_BASE = ROOT / "experiment_analysis/influence_of_individual_hyperparameters/single_hparam_results"

OUT_BASE.mkdir(exist_ok=True, parents=True)

# -------------------
# Helper: load one run
# -------------------
def load_run(run_dir):
    """Load R_max and hyperparams from a run directory."""
    config_path = run_dir / "config.yml"
    eval_path = run_dir / "eval" / "evaluations.npz"

    if not config_path.exists() or not eval_path.exists():
        return None

    # Load config
    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    env = cfg.get("env_id")
    algo = cfg.get("algo")
    seed = cfg.get("seed", None)

    hyper = cfg.get("hyperparams_parsed", {})
    # Convert net_arch (list) to a str for grouping
    if "policy_kwargs" in hyper and isinstance(hyper["policy_kwargs"], dict):
        na = hyper["policy_kwargs"].get("net_arch", None)
        if isinstance(na, list):
            hyper["net_arch"] = "-".join(str(x) for x in na)
        else:
            hyper["net_arch"] = str(na)
        del hyper["policy_kwargs"]
    else:
        hyper["net_arch"] = None

    # Load evaluations
    data = np.load(eval_path)
    results = data["results"]  # shape [evals, episodes]
    mean_return = results.mean(axis=1)
    R_max = float(mean_return.max())

    row = {"env": env, "algo": algo, "seed": seed, "R_max": R_max}
    row.update(hyper)
    return row


# -------------------
# Load all runs
# -------------------
all_rows = []
print("Loading runs from:", RUNS)

for env_dir in RUNS.iterdir():
    if not env_dir.is_dir():
        continue

    for algo_dir in env_dir.iterdir():
        if not algo_dir.is_dir():
            continue

        for run_dir in algo_dir.iterdir():
            if not run_dir.is_dir():
                continue

            row = load_run(run_dir)
            if row is not None:
                all_rows.append(row)

df_all = pd.DataFrame(all_rows)
print("Loaded runs:", len(df_all))

# -------------------
# Group by env + algo
# -------------------
env_algos = df_all.groupby(["env", "algo"])

# -------------------
# Utility functions
# -------------------
def make_plots_for_hparam(df, env, algo, hp, outdir):
    """Generate slice plots and boxplots."""
    sub = df[["R_max", hp]].dropna()
    if sub[hp].nunique() < 2:
        return

    # Line plot
    plt.figure(figsize=(5, 3))
    sns.pointplot(data=sub, x=hp, y="R_max")
    plt.title(f"{env}-{algo}: {hp} (line)")
    plt.tight_layout()
    plt.savefig(outdir / f"slices_{hp}.png")
    plt.close()

    # Box plot
    plt.figure(figsize=(5, 3))
    sns.boxplot(data=sub, x=hp, y="R_max")
    plt.title(f"{env}-{algo}: {hp} (box)")
    plt.tight_layout()
    plt.savefig(outdir / f"slices_box_{hp}.png")
    plt.close()


def compute_importance(df, hp):
    """Normalized between-slice variance."""
    if df[hp].nunique() < 2:
        return None

    total_var = df["R_max"].var()
    if total_var <= 1e-12:
        return None

    means = df.groupby(hp)["R_max"].mean()
    between_var = means.var()
    return float(between_var / total_var)


def make_global_pdp(df, env, algo, hp, outdir):
    """Scatter + LOWESS smooth."""
    sub = df[["R_max", hp]].dropna()
    if sub[hp].nunique() < 2:
        return

    x = sub[hp].values
    y = sub["R_max"].values

    plt.figure(figsize=(5, 3))
    plt.scatter(x, y, alpha=0.5, s=15)

    # LOWESS
    try:
        sm = lowess(y, x, frac=0.6)
        plt.plot(sm[:, 0], sm[:, 1], color="red")
    except Exception:
        pass

    plt.title(f"{env}-{algo}: PDP (global) {hp}")
    plt.tight_layout()
    plt.savefig(outdir / f"pdp_global_{hp}.png")
    plt.close()


# -------------------
# Main loop
# -------------------
for (env, algo), df in env_algos:
    print(f"\n=== Processing {env} / {algo} ({len(df)} runs) ===")

    outdir = OUT_BASE / f"{env}_{algo}"
    outdir.mkdir(exist_ok=True)

    hparams = [c for c in df.columns if c not in ["env", "algo", "seed", "R_max"]]

    # Skip if R_max variance = 0 (e.g. CartPole PPO)
    if df["R_max"].var() < 1e-12:
        print(f"  -> R_max variance = 0, skipping importance computation.")
        imp_df = pd.DataFrame([{
            "hyperparameter": "N/A",
            "importance_score": 0.0,
            "n_values": 0
        }])
        imp_df.to_csv(outdir / "importance_R_max.csv", index=False)
        continue

    rows_imp = []
    rows_kw = []

    for hp in hparams:
        nunique = df[hp].nunique()
        if nunique < 2:
            continue

        # Plots
        make_plots_for_hparam(df, env, algo, hp, outdir)
        make_global_pdp(df, env, algo, hp, outdir)

        # Importance
        imp = compute_importance(df, hp)
        if imp is not None:
            rows_imp.append({
                "hyperparameter": hp,
                "importance_score": imp,
                "n_values": nunique
            })

        # Kruskal test
        groups = [g["R_max"].values for _, g in df.groupby(hp)]
        try:
            H, p = kruskal(*groups)
        except Exception:
            H, p = None, None
        rows_kw.append({
            "hyperparameter": hp,
            "H_stat": H,
            "p_value": p,
            "n_groups": nunique
        })

    # Save importance
    imp_df = pd.DataFrame(rows_imp)
    imp_df.to_csv(outdir / "importance_R_max.csv", index=False)

    # Save Kruskal
    kw_df = pd.DataFrame(rows_kw)
    kw_df.to_csv(outdir / "kruskal_R_max.csv", index=False)

    print(f"Saved outputs → {outdir}")

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import kruskal
import yaml

# 尝试导入 LOWESS 平滑，没有也没关系
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    HAS_LOWESS = True
except Exception:
    HAS_LOWESS = False

# ============================================================
# Paths
# ============================================================
ROOT = Path("/homes/sohawan2/reinforcement-learning-thesis/thesis_project")
RUNS = ROOT / "runs_seed0"

OUT = ROOT / "experiment_analysis/influence_of_individual_hyperparameters/single_hparam_results"
OUT.mkdir(exist_ok=True, parents=True)


# ============================================================
# 1) Load runs into DataFrame
# ============================================================
records = []

for env_dir in RUNS.iterdir():
    if not env_dir.is_dir():
        continue
    env = env_dir.name

    for algo_dir in env_dir.iterdir():
        if not algo_dir.is_dir():
            continue
        algo = algo_dir.name

        for run_dir in algo_dir.iterdir():
            if not run_dir.is_dir():
                continue

            config_file = run_dir / "config.yml"
            eval_file = run_dir / "eval" / "evaluations.npz"

            if not config_file.exists() or not eval_file.exists():
                continue

            # config
            with open(config_file, "r") as f:
                cfg = yaml.safe_load(f)

            # eval data
            data = np.load(eval_file)
            results = data["results"]          # [n_eval, n_envs]
            timesteps = data["timesteps"]      # unused currently

            mean_returns = results.mean(axis=1)
            R_max = float(mean_returns.max())
            if len(mean_returns) >= 10:
                R_final = float(mean_returns[-10:].mean())
            else:
                R_final = float(mean_returns[-1])

            record = {
                "env": env,
                "algo": algo,
                "run_dir": str(run_dir),
                "R_max": R_max,
                "R_final": R_final,
                "seed": cfg.get("seed", None),
            }

            # flatten hyperparams
            hpp = cfg.get("hyperparams_parsed", {})
            for k, v in hpp.items():
                if k == "policy_kwargs" and isinstance(v, dict):
                    net_arch = v.get("net_arch", None)
                    if net_arch is not None:
                        record["net_arch"] = "-".join(str(x) for x in net_arch)
                else:
                    record[k] = v

            records.append(record)

df = pd.DataFrame(records)
print("Loaded:", len(df), "runs")
if df.empty:
    raise SystemExit("No runs found. Check directory paths.")


# ============================================================
# 2) Hyperparams per algorithm
# ============================================================
hyperparams_by_algo = {
    "a2c": [
        "learning_rate", "n_steps", "gamma", "gae_lambda",
        "ent_coef", "vf_coef", "max_grad_norm",
        "use_rms_prop", "rms_prop_eps",
        "net_arch"
    ],
    "ppo": [
        "learning_rate", "n_steps", "batch_size", "n_epochs",
        "gamma", "clip_range", "gae_lambda", "ent_coef",
        "vf_coef", "max_grad_norm", "target_kl",
        "net_arch"
    ],
}

for algo in hyperparams_by_algo:
    hyperparams_by_algo[algo] = [
        h for h in hyperparams_by_algo[algo] if h in df.columns
    ]

print("\nHyperparams by algo:")
for algo, hps in hyperparams_by_algo.items():
    print(f"  {algo}: {hps}")


def is_numeric(series: pd.Series) -> bool:
    """判断列是否为数值类型（用于 global scatter PDP）"""
    return np.issubdtype(series.dtype, np.number)


# ============================================================
# 3) Per-env × per-algo analysis
# ============================================================
envs = sorted(df["env"].unique())
algos = sorted(df["algo"].unique())

for env in envs:
    for algo in algos:
        sub = df[(df["env"] == env) & (df["algo"] == algo)]
        if len(sub) == 0:
            continue

        if algo not in hyperparams_by_algo:
            continue
        hparams = hyperparams_by_algo[algo]
        if not hparams:
            continue

        print(f"\n=== Processing {env} / {algo} (n={len(sub)}) ===")

        out_dir = OUT / f"{env}_{algo}"
        out_dir.mkdir(exist_ok=True, parents=True)

        # ------------------------------------------------------------
        # 3.1 Global slices (R_max): line+error + boxplot + Kruskal
        # ------------------------------------------------------------
        kw_rows = []

        # line+error + boxplot + global scatter
        for h in hparams:
            col = sub[h]

            # ---------- line + errorbar ----------
            grp = sub.groupby(h)["R_max"].agg(["mean", "std", "count"]).reset_index()
            grp.to_csv(out_dir / f"slices_{h}.csv", index=False)

            plt.figure(figsize=(6, 4))
            plt.errorbar(grp[h], grp["mean"], yerr=grp["std"], fmt="o-", capsize=4)
            plt.xlabel(h)
            plt.ylabel("R_max")
            plt.title(f"{env} / {algo} – Slice: {h}")
            plt.tight_layout()
            plt.savefig(out_dir / f"slices_{h}.png", dpi=200)
            plt.close()

            # ---------- boxplot ----------
            values = grp[h].values
            data_groups = [sub[sub[h] == v]["R_max"].values for v in values]

            plt.figure(figsize=(6, 4))
            positions = np.arange(len(values))
            plt.boxplot(
                data_groups,
                positions=positions,
                widths=0.6,
                showfliers=True
            )
            plt.xticks(positions, [str(v) for v in values], rotation=30, ha="right")
            plt.xlabel(h)
            plt.ylabel("R_max")
            plt.title(f"{env} / {algo} – Boxplot slice: {h}")
            plt.tight_layout()
            plt.savefig(out_dir / f"slices_box_{h}.png", dpi=200)
            plt.close()

            # ---------- Kruskal ----------
            groups = [g["R_max"].values for _, g in sub.groupby(h)]
            if len(groups) > 1:
                all_vals = np.concatenate(groups)
                if not np.allclose(all_vals, all_vals[0]):
                    try:
                        H, p = kruskal(*groups)
                    except ValueError:
                        H, p = None, None
                    else:
                        kw_rows.append({
                            "hyperparameter": h,
                            "H_stat": H,
                            "p_value": p,
                            "n_groups": len(groups)
                        })

            # ---------- global scatter + smooth PDP ----------
            if is_numeric(col) and col.nunique() >= 3:
                x = col.values
                y = sub["R_max"].values

                plt.figure(figsize=(6, 4))
                plt.scatter(x, y, alpha=0.3, s=10)

                if HAS_LOWESS:
                    try:
                        sm = lowess(y, x, frac=0.6, return_sorted=True)
                        plt.plot(sm[:, 0], sm[:, 1], linewidth=2)
                    except Exception:
                        pass
                else:
                    # fallback: simple moving average
                    order = np.argsort(x)
                    x_sorted = x[order]
                    y_sorted = y[order]
                    window = max(3, len(y_sorted) // 10)
                    if window > 1:
                        kernel = np.ones(window) / window
                        y_smooth = np.convolve(y_sorted, kernel, mode="valid")
                        x_smooth = x_sorted[window // 2: window // 2 + len(y_smooth)]
                        plt.plot(x_smooth, y_smooth, linewidth=2)

                plt.xlabel(h)
                plt.ylabel("R_max")
                plt.title(f"{env} / {algo} – Global PDP (scatter): {h}")
                plt.tight_layout()
                plt.savefig(out_dir / f"pdp_global_{h}.png", dpi=200)
                plt.close()

        if kw_rows:
            pd.DataFrame(kw_rows).to_csv(out_dir / "kruskal_R_max.csv", index=False)
        print("Saved slices (line+box), Kruskal tests, and global PDP scatter.")

        # ------------------------------------------------------------
        # 3.1c Variance-based importance scores (always write CSV)
        # ------------------------------------------------------------
        importance_rows = []
        total_var = sub["R_max"].var()

        if total_var > 0:
            for h in hparams:
                if sub[h].nunique() <= 1:
                    continue
                means = sub.groupby(h)["R_max"].mean()
                var_between = means.var()
                imp = float(var_between / total_var)
                importance_rows.append({
                    "hyperparameter": h,
                    "importance_score": imp,
                    "n_values": int(sub[h].nunique())
                })
        else:
            # 例如 CartPole-v1 / ppo: R_max 完全一样 → variance 为 0
            importance_rows.append({
                "hyperparameter": "N/A",
                "importance_score": 0.0,
                "n_values": 0
            })

        imp_df = pd.DataFrame(importance_rows)
        imp_df.to_csv(out_dir / "importance_R_max.csv", index=False)
        print("Saved importance scores.")

        # ------------------------------------------------------------
        # 3.2 Local PDP (around best config, exact-match)
        # ------------------------------------------------------------
        ref = sub.sort_values("R_max", ascending=False).iloc[0]

        for h in hparams:
            cond = np.ones(len(sub), dtype=bool)
            for hp in hparams:
                if hp != h and hp in sub.columns:
                    cond &= (sub[hp] == ref[hp])

            local = sub[cond]
            if local[h].nunique() <= 1:
                continue

            grp_local = local.groupby(h)["R_max"].mean().reset_index()

            plt.figure(figsize=(6, 4))
            plt.plot(grp_local[h], grp_local["R_max"], "o-")
            plt.xlabel(h)
            plt.ylabel("R_max")
            plt.title(f"{env} / {algo} – PDP local (exact match): {h}")
            plt.tight_layout()
            plt.savefig(out_dir / f"pdp_local_{h}.png", dpi=200)
            plt.close()

        print("Saved local PDPs.")

import numpy as np
import pandas as pd
from pathlib import Path
import yaml

# 跟 influence_single_hparam.py 一样的 ROOT / RUNS
ROOT = Path("/homes/sohawan2/reinforcement-learning-thesis/thesis_project")
RUNS = ROOT / "runs_seed0"

# 和之前一致的 PPO 超参列表
PPO_HPARAMS = [
    "learning_rate", "n_steps", "batch_size", "n_epochs",
    "gamma", "clip_range", "gae_lambda", "ent_coef",
    "vf_coef", "max_grad_norm", "target_kl",
    "net_arch"
]

# =====================================================
# 1) 重新构建 df（只关心 CartPole-v1 / PPO 也可以）
# =====================================================

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

            with open(config_file, "r") as f:
                cfg = yaml.safe_load(f)

            data = np.load(eval_file)
            results = data["results"]
            mean_returns = results.mean(axis=1)
            R_max = float(mean_returns.max())

            record = {
                "env": env,
                "algo": algo,
                "run_dir": str(run_dir),
                "R_max": R_max,
            }

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
print("Total runs loaded:", len(df))

# 只看 CartPole-v1 / ppo
sub = df[(df["env"] == "CartPole-v1") & (df["algo"] == "ppo")].copy()
print("\n=== CartPole-v1 / ppo summary ===")
print("n_runs:", len(sub))

if len(sub) == 0:
    raise SystemExit("No CartPole-v1 / ppo runs found, check RUNS path.")

print("R_max variance:", sub["R_max"].var())

# 只保留实际存在的超参列
hparams = [h for h in PPO_HPARAMS if h in sub.columns]
print("\nHyperparameters present in DataFrame:", hparams)

print("\nPer-hyperparameter unique values:")
for h in hparams:
    vals = sub[h].unique()
    print(f"- {h}: nunique={len(vals)}, values={vals}")

# =====================================================
# 2) 模拟 importance 计算，看看理论上应该不会都是 0 吧？
# =====================================================
print("\nApproximate importance scores (same formula as script):")
total_var = sub["R_max"].var()
if total_var <= 0:
    print("Total variance of R_max is 0 or negative; importance will be empty.")
else:
    rows = []
    for h in hparams:
        if sub[h].nunique() <= 1:
            rows.append((h, 0.0))
            continue
        means = sub.groupby(h)["R_max"].mean()
        var_between = means.var()
        imp = float(var_between / total_var)
        rows.append((h, imp))

    rows_sorted = sorted(rows, key=lambda x: x[1], reverse=True)
    for h, imp in rows_sorted:
        print(f"- {h:12s}  importance = {imp:.6f}")

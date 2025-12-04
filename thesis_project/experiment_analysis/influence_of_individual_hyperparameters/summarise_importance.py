import pandas as pd
from pathlib import Path

ROOT = Path("/homes/sohawan2/reinforcement-learning-thesis/thesis_project")
BASE = ROOT / "experiment_analysis/influence_of_individual_hyperparameters/single_hparam_results"

out_rows = []

print("Scanning:", BASE)

for subdir in sorted(BASE.iterdir()):
    if not subdir.is_dir():
        continue

    name = subdir.name  # e.g. "Hopper-v4_ppo"
    if "_" not in name:
        continue
    env, algo = name.rsplit("_", 1)

    imp_file = subdir / "importance_R_max.csv"
    if not imp_file.exists():
        print(f"[WARN] {name}: no importance_R_max.csv, skip.")
        continue

    # keep_default_na=False: 不要把 "N/A" 自动当成 NaN
    df = pd.read_csv(imp_file, keep_default_na=False)

    if "importance_score" not in df.columns or "hyperparameter" not in df.columns:
        print(f"[WARN] {name}: malformed importance_R_max.csv, skip.")
        continue

    # 跳过占位行：hyperparameter == "N/A" 或 空 / NaN
    df = df[df["hyperparameter"].notna()]
    df = df[df["hyperparameter"] != "N/A"]

    if df.empty:
        print(f"\n=== {env} / {algo} ===")
        print("  No valid importance entries (e.g. all N/A or variance=0).")
        continue

    df_sorted = df.sort_values("importance_score", ascending=False)
    top3 = df_sorted.head(3)

    print(f"\n=== {env} / {algo} ===")
    for rank, (_, row) in enumerate(top3.iterrows(), start=1):
        hp = row["hyperparameter"]
        score = float(row["importance_score"])
        n_vals = row.get("n_values", None)
        if pd.isna(n_vals):
            n_vals_str = ""
        else:
            n_vals_str = f" (n_values={int(n_vals)})"
        # 强制转成字符串再对齐，防止类型问题
        hp_str = str(hp)
        print(f"  #{rank}: {hp_str:15s} importance={score:.4f}{n_vals_str}")

        out_rows.append({
            "env": env,
            "algo": algo,
            "rank": rank,
            "hyperparameter": hp_str,
            "importance_score": score,
            "n_values": int(n_vals) if n_vals == n_vals else None,
        })

# 汇总总表
if out_rows:
    out_df = pd.DataFrame(out_rows)
    out_path = BASE / "top3_importance_R_max_all_envs.csv"
    out_df.to_csv(out_path, index=False)
    print("\nSaved combined top-3 table to:", out_path)
else:
    print("\nNo importance data found (all env–algo pairs were N/A or empty).")

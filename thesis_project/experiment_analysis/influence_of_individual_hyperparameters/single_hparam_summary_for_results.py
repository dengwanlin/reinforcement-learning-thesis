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

    # importance：过滤掉 N/A 占位
    imp_df = pd.read_csv(imp_file, keep_default_na=False)
    if "importance_score" not in imp_df.columns or "hyperparameter" not in imp_df.columns:
        print(f"[WARN] {name}: malformed importance_R_max.csv, skip.")
        continue

    imp_df = imp_df[imp_df["hyperparameter"].notna()]
    imp_df = imp_df[imp_df["hyperparameter"] != "N/A"]
    if imp_df.empty:
        print(f"\n=== {env} / {algo} ===")
        print("  No valid importance entries (variance=0 or only N/A).")
        continue

    # 只要 top-3
    imp_top3 = imp_df.sort_values("importance_score", ascending=False).head(3).copy()
    imp_top3["hyperparameter"] = imp_top3["hyperparameter"].astype(str)

    # 尝试读 Kruskal 结果
    kw_file = subdir / "kruskal_R_max.csv"
    if kw_file.exists():
        kw_df = pd.read_csv(kw_file)
        if "hyperparameter" in kw_df.columns:
            kw_df["hyperparameter"] = kw_df["hyperparameter"].astype(str)
            merged = pd.merge(
                imp_top3,
                kw_df,
                on="hyperparameter",
                how="left",
                suffixes=("", "_kw"),
            )
        else:
            merged = imp_top3.copy()
            merged["H_stat"] = None
            merged["p_value"] = None
            merged["n_groups"] = None
    else:
        merged = imp_top3.copy()
        merged["H_stat"] = None
        merged["p_value"] = None
        merged["n_groups"] = None

    print(f"\n=== {env} / {algo} ===")
    for rank, (_, row) in enumerate(merged.iterrows(), start=1):
        hp = row["hyperparameter"]
        score = float(row["importance_score"])
        n_vals = row.get("n_values", None)
        H = row.get("H_stat", None)
        p = row.get("p_value", None)

        n_vals_str = ""
        if pd.notna(n_vals):
            n_vals_str = f" (n_values={int(n_vals)})"

        print(f"  #{rank}: {hp:15s} importance={score:.4f}{n_vals_str}, "
              f"H={H}, p={p}")

        out_rows.append({
            "env": env,
            "algo": algo,
            "rank": rank,
            "hyperparameter": hp,
            "importance_score": score,
            "n_values": int(n_vals) if pd.notna(n_vals) else None,
            "H_stat": H,
            "p_value": p,
        })

# 汇总到一个大 CSV
if out_rows:
    out_df = pd.DataFrame(out_rows)
    out_path = BASE / "single_hparam_summary_R_max_all_envs.csv"
    out_df.to_csv(out_path, index=False)
    print("\nSaved combined summary to:", out_path)
else:
    print("\nNo summary rows collected.")

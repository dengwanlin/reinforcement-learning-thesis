# hyperparameter_interaction_effects/interaction_analysis.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from .config import ANALYSIS_OUT_DIR, METRIC_COL, HP_PAIRS, ENVS


# 合并后的 metrics 表路径
DATA_PATH = ANALYSIS_OUT_DIR / "interaction_metrics.csv"


# --------------------------------------------------------------------------- #
# 0. 读取数据
# --------------------------------------------------------------------------- #
def load_data() -> pd.DataFrame:
    """
    Load the consolidated metrics table produced by collect_runs.py.

    Expected columns:
      - 'env', 'algo'
      - METRIC_COL (e.g. 'max_eval_return')
      - hyperparameter columns (e.g. 'learning_rate', 'n_steps', ...)
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Consolidated metrics file not found: {DATA_PATH}\n"
            f"Run collect_runs.build_consolidated_table() first."
        )

    df = pd.read_csv(DATA_PATH)
    return df


# --------------------------------------------------------------------------- #
# 1. Heatmap / 2D performance surface
# --------------------------------------------------------------------------- #
def compute_heatmap(
    df_ea: pd.DataFrame, h1: str, h2: str, metric_col: str = METRIC_COL
) -> pd.DataFrame:
    """
    Compute a 2D performance surface for a given env+algo and hyperparameter pair.

    Returns a pivot table: index = h2, columns = h1, values = mean(metric_col).
    """
    if h1 not in df_ea.columns or h2 not in df_ea.columns:
        raise KeyError(f"Missing hyperparameter columns: {h1} or {h2}")

    grouped = (
        df_ea
        .groupby([h1, h2], as_index=False)[metric_col]
        .mean()
    )
    if grouped.empty:
        return pd.DataFrame()

    pivot = grouped.pivot(index=h2, columns=h1, values=metric_col)
    return pivot


def save_heatmap(
    pivot: pd.DataFrame,
    env: str,
    algo: str,
    h1: str,
    h2: str,
) -> None:
    """
    Save the 2D performance surface as CSV and a PNG heatmap.

    目录结构：
        results/
          <env>/
            <algo>/
              heatmaps/
                heatmap_<h1>_<h2>.csv
                heatmap_<h1>_<h2>.png
    """
    if pivot.empty:
        print(f"[heatmap] Empty pivot for {env} / {algo} / ({h1}, {h2}), skipped.")
        return

    out_dir = ANALYSIS_OUT_DIR / env / algo / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"heatmap_{h1}_{h2}.csv"
    png_path = out_dir / f"heatmap_{h1}_{h2}.png"

    # 保存数值数据
    pivot.to_csv(csv_path)

    # 用 matplotlib 画 heatmap（简单够用，后面你想美化可以单独写脚本）
    plt.figure(figsize=(6, 5))
    plt.imshow(pivot.values, origin="lower", aspect="auto")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label=METRIC_COL)
    plt.xlabel(pivot.columns.name or h1)
    plt.ylabel(pivot.index.name or h2)
    plt.title(f"{env} / {algo}: {METRIC_COL} over {h1} × {h2}")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    print(f"[heatmap] Saved CSV -> {csv_path}")
    print(f"[heatmap] Saved PNG -> {png_path}")


# --------------------------------------------------------------------------- #
# 2. Two-way ANOVA
# --------------------------------------------------------------------------- #
def run_two_way_anova(
    df_ea: pd.DataFrame,
    h1: str,
    h2: str,
    metric_col: str = METRIC_COL,
) -> Optional[Dict[str, Any]]:
    """
    Run a two-way ANOVA with interaction for the given hyperparameter pair.

    Model:
        metric ~ C(h1) * C(h2)

    Returns
    -------
    result : dict or None
        If successful, returns dictionary with F, p, partial eta^2, etc.
        Returns None if data is insufficient.
    """
    cols_needed = [metric_col, h1, h2]
    for c in cols_needed:
        if c not in df_ea.columns:
            print(f"[ANOVA] Column {c} not found, skipping.")
            return None

    sub = df_ea[cols_needed].dropna().copy()
    if sub.empty:
        print(f"[ANOVA] No data for ({h1}, {h2}), skipping.")
        return None

    formula = f"{metric_col} ~ C({h1}) * C({h2})"
    try:
        model = smf.ols(formula=formula, data=sub).fit()
        anova_res = anova_lm(model, typ=2)
    except Exception as e:
        print(f"[ANOVA] Failed for ({h1}, {h2}): {e}")
        return None

    inter_keys = [idx for idx in anova_res.index if ":" in idx]
    if not inter_keys:
        print(f"[ANOVA] No interaction term detected in ANOVA index for ({h1}, {h2}).")
        return None

    inter_key = inter_keys[0]
    ss_inter = anova_res.loc[inter_key, "sum_sq"]
    df_inter = anova_res.loc[inter_key, "df"]
    f_inter = anova_res.loc[inter_key, "F"]
    p_inter = anova_res.loc[inter_key, "PR(>F)"]

    ss_resid = anova_res.loc["Residual", "sum_sq"]
    df_resid = anova_res.loc["Residual", "df"]

    if (ss_inter + ss_resid) > 0:
        eta_sq_partial = ss_inter / (ss_inter + ss_resid)
    else:
        eta_sq_partial = np.nan

    result: Dict[str, Any] = {
        "term": inter_key,
        "F": float(f_inter),
        "p": float(p_inter),
        "eta_sq_partial": float(eta_sq_partial),
        "df_inter": float(df_inter),
        "df_resid": float(df_resid),
        "n": int(len(sub)),
        "n_levels_h1": int(sub[h1].nunique()),
        "n_levels_h2": int(sub[h2].nunique()),
    }
    return result


def append_local_anova_row(row: Dict[str, Any], env: str, algo: str) -> None:
    """
    把单条 ANOVA 结果追加到对应 env/algo 下的 anova_summary.csv 中。

    目录：
        results/<env>/<algo>/anova/anova_summary.csv
    """
    out_dir = ANALYSIS_OUT_DIR / env / algo / "anova"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "anova_summary.csv"

    df_new = pd.DataFrame([row])
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(csv_path, index=False)


# --------------------------------------------------------------------------- #
# 3. Regression: main-effect vs interaction model
# --------------------------------------------------------------------------- #
def make_numeric_features(
    df_ea: pd.DataFrame,
    h1: str,
    h2: str,
) -> Tuple[pd.DataFrame, Dict[str, bool]]:
    """
    Construct numeric features x1, x2 from h1, h2 for regression.

    Heuristic:
      - If a hyperparameter spans > 10x range and is strictly positive,
        apply log10 transform.
    """
    cols_needed = [METRIC_COL, h1, h2]
    sub = df_ea[cols_needed].dropna().copy()
    if sub.empty:
        return sub, {"log_h1": False, "log_h2": False}

    def transform(col: str):
        vals = sub[col].astype(float)
        vmin, vmax = vals.min(), vals.max()
        if vmin > 0 and vmax / max(vmin, 1e-12) > 10:
            return np.log10(vals), True
        else:
            return vals, False

    sub["x1"], log1 = transform(h1)
    sub["x2"], log2 = transform(h2)
    return sub, {"log_h1": log1, "log_h2": log2}


def run_regression_comparison(
    df_ea: pd.DataFrame,
    h1: str,
    h2: str,
) -> Optional[Dict[str, Any]]:
    """
    Compare main-effect vs interaction regression models:

      main-effect:    metric ~ x1 + x2
      interaction:    metric ~ x1 * x2

    Returns a dict with adj_R2, AIC, BIC, interaction term significance, etc.
    """
    for c in (METRIC_COL, h1, h2):
        if c not in df_ea.columns:
            print(f"[REG] Column {c} not found, skipping.")
            return None

    sub, info = make_numeric_features(df_ea, h1, h2)
    if sub.empty:
        print(f"[REG] No numeric data for ({h1}, {h2}), skipping.")
        return None

    metric_col = METRIC_COL

    try:
        model_main = smf.ols(f"{metric_col} ~ x1 + x2", data=sub).fit()
        model_int = smf.ols(f"{metric_col} ~ x1 * x2", data=sub).fit()
    except Exception as e:
        print(f"[REG] Regression failed for ({h1}, {h2}): {e}")
        return None

    # F-test: does interaction model significantly improve fit?
    try:
        anova_cmp = anova_lm(model_main, model_int)
        F_change = float(anova_cmp["F"].iloc[-1])
        p_change = float(anova_cmp["Pr(>F)"].iloc[-1])
    except Exception:
        F_change, p_change = np.nan, np.nan

    inter_terms = [name for name in model_int.params.index if ":" in name]
    if inter_terms:
        inter_term = inter_terms[0]
        beta_inter = float(model_int.params[inter_term])
        p_beta_inter = float(model_int.pvalues[inter_term])
    else:
        beta_inter, p_beta_inter = np.nan, np.nan

    result: Dict[str, Any] = {
        "adj_R2_main": float(model_main.rsquared_adj),
        "adj_R2_inter": float(model_int.rsquared_adj),
        "AIC_main": float(model_main.aic),
        "AIC_inter": float(model_int.aic),
        "BIC_main": float(model_main.bic),
        "BIC_inter": float(model_int.bic),
        "beta_inter": beta_inter,
        "p_beta_inter": p_beta_inter,
        "F_change": F_change,
        "p_change": p_change,
        "used_log_h1": bool(info["log_h1"]),
        "used_log_h2": bool(info["log_h2"]),
        "n": int(len(sub)),
    }
    return result


def append_local_regression_row(row: Dict[str, Any], env: str, algo: str) -> None:
    """
    把单条回归比较结果追加到对应 env/algo 的 regression_summary.csv 中。

    目录：
        results/<env>/<algo>/regression/regression_summary.csv
    """
    out_dir = ANALYSIS_OUT_DIR / env / algo / "regression"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "regression_summary.csv"

    df_new = pd.DataFrame([row])
    if csv_path.exists():
        df_old = pd.read_csv(csv_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(csv_path, index=False)


# --------------------------------------------------------------------------- #
# 4. 主入口：对所有 env + algo + 超参对 运行分析
# --------------------------------------------------------------------------- #
def run_interaction_pipeline() -> None:
    """
    全流程：

      - Load consolidated metrics.
      - For each env, algo, hyperparameter pair:
          * compute & save heatmap (PNG + CSV) under env/algo/heatmaps/
          * run ANOVA (本地 per-env/algo + 全局汇总)
          * run regression comparison (本地 per-env/algo + 全局汇总)
      - Save global summary CSVs (interaction_anova_summary.csv,
        interaction_regression_summary.csv).
    """
    df = load_data()

    if "env" not in df.columns or "algo" not in df.columns:
        raise KeyError("Data must contain 'env' and 'algo' columns.")

    anova_rows = []
    regr_rows = []

    for env in ENVS:
        df_env = df[df["env"] == env].copy()
        if df_env.empty:
            print(f"[PIPELINE] No data for env={env}, skipping.")
            continue

        for algo, pairs in HP_PAIRS.items():
            df_ea = df_env[df_env["algo"] == algo].copy()
            if df_ea.empty:
                print(f"[PIPELINE] No data for env={env}, algo={algo}, skipping.")
                continue

            for (h1, h2) in pairs:
                print(f"\n[PIPELINE] env={env}, algo={algo}, pair=({h1}, {h2})")

                # 1) Heatmap
                try:
                    pivot = compute_heatmap(df_ea, h1, h2)
                    save_heatmap(pivot, env, algo, h1, h2)
                except Exception as e:
                    print(f"[PIPELINE] Heatmap failed for ({h1}, {h2}): {e}")

                # 2) ANOVA
                anova_res = run_two_way_anova(df_ea, h1, h2)
                if anova_res is not None:
                    anova_res.update({"env": env, "algo": algo, "h1": h1, "h2": h2})
                    anova_rows.append(anova_res)
                    append_local_anova_row(anova_res, env, algo)

                # 3) Regression comparison
                regr_res = run_regression_comparison(df_ea, h1, h2)
                if regr_res is not None:
                    regr_res.update({"env": env, "algo": algo, "h1": h1, "h2": h2})
                    regr_rows.append(regr_res)
                    append_local_regression_row(regr_res, env, algo)

    # 全局汇总表（跨 env+algo）
    if anova_rows:
        anova_df = pd.DataFrame(anova_rows)
        anova_path = ANALYSIS_OUT_DIR / "interaction_anova_summary.csv"
        anova_df.to_csv(anova_path, index=False)
        print(f"\n[PIPELINE] Saved global ANOVA summary -> {anova_path}")
    else:
        print("\n[PIPELINE] No ANOVA results were produced.")

    if regr_rows:
        regr_df = pd.DataFrame(regr_rows)
        regr_path = ANALYSIS_OUT_DIR / "interaction_regression_summary.csv"
        regr_df.to_csv(regr_path, index=False)
        print(f"[PIPELINE] Saved global regression summary -> {regr_path}")
    else:
        print("[PIPELINE] No regression results were produced.")


if __name__ == "__main__":
    run_interaction_pipeline()

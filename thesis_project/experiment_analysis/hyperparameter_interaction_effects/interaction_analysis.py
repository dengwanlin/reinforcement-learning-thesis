"""
Hyperparameter interaction analysis:
- 读取 consolidated metrics 表（由 collect_runs.py 生成）
- 对每个 env + algo + (h1, h2) 组合：
  - 构建 2D performance surface（heatmap）
  - 进行 two-way ANOVA 检验交互项显著性
  - 用回归模型比较 main-effect vs interaction 模型
- 输出：
  - heatmap_*.csv：2D surface 数据，供画图
  - plots/heatmap_*.pdf：简单热力图，可预览
  - interaction_anova_summary.csv：ANOVA 结果汇总
  - interaction_regression_summary.csv：回归结果汇总
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

from .config import ANALYSIS_OUT_DIR, METRIC_COL, HP_PAIRS, ENVS


# 路径设置
DATA_PATH = ANALYSIS_OUT_DIR / "interaction_metrics.csv"
PLOTS_DIR = ANALYSIS_OUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 基础：读取数据
# --------------------------------------------------------------------------- #
def load_data() -> pd.DataFrame:
    """
    Load the consolidated metrics table produced by collect_runs.py.

    Expected columns include at least:
      - 'env', 'algo'
      - METRIC_COL (e.g. 'max_eval_return')
      - various hyperparameter columns (e.g. 'learning_rate', 'n_steps', ...)
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Consolidated metrics file not found: {DATA_PATH}\n"
            f"Run collect_runs.build_consolidated_table() first."
        )

    df = pd.read_csv(DATA_PATH)

    # 可选：在这里对 seed 做平均（若你希望在这里平均）
    # 目前先保持每个 seed 一行，后续 groupby 时自动平均

    return df


# --------------------------------------------------------------------------- #
# 1. Heatmap / 2D performance surface
# --------------------------------------------------------------------------- #
def compute_heatmap(
    df_ea: pd.DataFrame, h1: str, h2: str, metric_col: str = METRIC_COL
) -> pd.DataFrame:
    """
    Compute a 2D performance surface for a given env+algo and hyperparameter pair.

    Parameters
    ----------
    df_ea : DataFrame
        Filtered data for a single (env, algo).
    h1, h2 : str
        Column names of the two hyperparameters.
    metric_col : str
        Column name of the performance metric.

    Returns
    -------
    pivot : DataFrame
        2D table with index = h2, columns = h1, values = mean(metric_col).
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
    Save the 2D performance surface as CSV and a simple PDF heatmap.
    """
    if pivot.empty:
        print(f"[heatmap] Empty pivot for {env} / {algo} / ({h1}, {h2}), skipped.")
        return

    # 保存 matrix CSV
    mat_path = ANALYSIS_OUT_DIR / f"heatmap_{env}_{algo}_{h1}_{h2}.csv"
    pivot.to_csv(mat_path)

    # 简单画一个 matplotlib heatmap 供预览
    fig_path = PLOTS_DIR / f"heatmap_{env}_{algo}_{h1}_{h2}.pdf"

    plt.figure()
    plt.imshow(pivot.values, origin="lower", aspect="auto")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.colorbar(label=METRIC_COL)
    plt.xlabel(pivot.columns.name)
    plt.ylabel(pivot.index.name)
    plt.title(f"{env} {algo}: {METRIC_COL} over {h1} × {h2}")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    print(f"[heatmap] Saved CSV -> {mat_path}")
    print(f"[heatmap] Saved PDF -> {fig_path}")


# --------------------------------------------------------------------------- #
# 2. Two-way ANOVA
# --------------------------------------------------------------------------- #
def run_two_way_anova(
    df_ea: pd.DataFrame,
    h1: str,
    h2: str,
    metric_col: str = METRIC_COL,
) -> Optional[Dict]:
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

    # interaction term key e.g. 'C(h1):C(h2)'
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

    result = {
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


# --------------------------------------------------------------------------- #
# 3. Regression: main-effect vs interaction model
# --------------------------------------------------------------------------- #
def make_numeric_features(
    df_ea: pd.DataFrame,
    h1: str,
    h2: str,
) -> tuple[pd.DataFrame, Dict[str, bool]]:
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
) -> Optional[Dict]:
    """
    Compare main-effect vs interaction regression models:

      main-effect:    metric ~ x1 + x2
      interaction:    metric ~ x1 * x2  (i.e. x1 + x2 + x1:x2)

    Returns
    -------
    result : dict or None
        Contains adj_R2, AIC, BIC for both models, interaction coefficient,
        and F-test for model comparison.
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

    result = {
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


# --------------------------------------------------------------------------- #
# 4. 主入口：对所有 env + algo + 超参对 运行分析
# --------------------------------------------------------------------------- #
def run_interaction_pipeline() -> None:
    """
    Run the full interaction analysis pipeline:

      - Load consolidated metrics.
      - For each env, algo, hyperparameter pair:
          * compute and save heatmap
          * run two-way ANOVA
          * run regression comparison
      - Save summary CSVs for ANOVA and regression.
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

                # Heatmap
                try:
                    pivot = compute_heatmap(df_ea, h1, h2)
                    save_heatmap(pivot, env, algo, h1, h2)
                except Exception as e:
                    print(f"[PIPELINE] Heatmap failed for ({h1}, {h2}): {e}")

                # ANOVA
                anova_res = run_two_way_anova(df_ea, h1, h2)
                if anova_res is not None:
                    anova_res.update({"env": env, "algo": algo, "h1": h1, "h2": h2})
                    anova_rows.append(anova_res)

                # Regression comparison
                regr_res = run_regression_comparison(df_ea, h1, h2)
                if regr_res is not None:
                    regr_res.update({"env": env, "algo": algo, "h1": h1, "h2": h2})
                    regr_rows.append(regr_res)

    # Save summaries
    if anova_rows:
        anova_df = pd.DataFrame(anova_rows)
        anova_path = ANALYSIS_OUT_DIR / "interaction_anova_summary.csv"
        anova_df.to_csv(anova_path, index=False)
        print(f"\n[PIPELINE] Saved ANOVA summary -> {anova_path}")
    else:
        print("\n[PIPELINE] No ANOVA results were produced.")

    if regr_rows:
        regr_df = pd.DataFrame(regr_rows)
        regr_path = ANALYSIS_OUT_DIR / "interaction_regression_summary.csv"
        regr_df.to_csv(regr_path, index=False)
        print(f"[PIPELINE] Saved regression summary -> {regr_path}")
    else:
        print("[PIPELINE] No regression results were produced.")


if __name__ == "__main__":
    run_interaction_pipeline()

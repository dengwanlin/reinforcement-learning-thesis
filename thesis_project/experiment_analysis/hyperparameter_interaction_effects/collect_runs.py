# hyperparameter_interaction_effects/collect_runs.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd
import yaml

from .config import LOG_ROOT, ANALYSIS_OUT_DIR, METRIC_COL


# --------------------------------------------------------------------------- #
# 辅助转换函数
# --------------------------------------------------------------------------- #

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 解析 config.yml
# --------------------------------------------------------------------------- #

def parse_config(config_path: Path) -> Dict[str, Any]:
    """
    针对当前 thesis 项目的 config.yml 结构量身定制的解析函数。

    典型结构示例：

    env_id: Hopper-v4
    algo: a2c
    seed: 0
    ...
    hyperparams_parsed:
      learning_rate: 0.0007
      n_steps: 128
      gamma: 0.99
      gae_lambda: 0.9
      ent_coef: 0.0
      clip_range: 0.1         # 仅在 PPO 中存在
      batch_size: 128         # 仅在 PPO 中存在
      ...

    我们直接从 hyperparams_parsed 中取需要的超参数。
    """
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)

    # env / algo / seed
    env_id = cfg.get("env_id", None)  # 注意：你的配置里字段名是 env_id
    algo = cfg.get("algo", None)
    seed = cfg.get("seed", None)

    # 关键超参数都在 hyperparams_parsed 里面
    hp = cfg.get("hyperparams_parsed", {}) or {}

    learning_rate = _safe_float(hp.get("learning_rate", None))
    n_steps = _safe_int(hp.get("n_steps", None))
    gamma = _safe_float(hp.get("gamma", None))
    gae_lambda = _safe_float(hp.get("gae_lambda", None))
    ent_coef = _safe_float(hp.get("ent_coef", None))
    clip_range = _safe_float(hp.get("clip_range", None))   # A2C 里可能不存在 → None
    batch_size = _safe_int(hp.get("batch_size", None))     # A2C 里也可能不存在

    row: Dict[str, Any] = {
        "env": env_id,   # 统一称为 env，后面分析用这个字段
        "algo": algo,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "batch_size": batch_size,
        "seed": _safe_int(seed) if seed is not None else None,
    }
    return row


# --------------------------------------------------------------------------- #
# 解析 eval/evaluations.npz -> max_eval_return
# --------------------------------------------------------------------------- #

def parse_evaluations(eval_path: Path) -> Optional[float]:
    """
    解析 eval/evaluations.npz，计算 max_eval_return：
      - 从 'results' 中取每次评估的平均回报（按 env 维度平均）
      - 再取整个训练过程中的最大值
    """
    if not eval_path.exists():
        return None

    data = np.load(eval_path)
    results = data.get("results", None)
    if results is None:
        return None

    # results: shape (n_eval, n_envs)
    mean_over_envs = results.mean(axis=1)   # shape: (n_eval,)
    max_eval = float(mean_over_envs.max())
    return max_eval


# --------------------------------------------------------------------------- #
# 从单个 run 目录收集一行数据
# --------------------------------------------------------------------------- #

def collect_single_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    """
    从单个 run 目录抽取一行记录：
      env, algo, seed, hyperparams, metric 等。

    期望目录结构：
      LOG_ROOT /
        env /
          algo /
            run_id /
              config.yml
              eval/evaluations.npz
              results.json (可选)
              ...
    """
    # 从路径推一份 env/algo 作为 fallback，以防 config 里缺失
    try:
        algo_from_dir = run_dir.parent.name       # .../env/algo/run_id
        env_from_dir = run_dir.parent.parent.name
    except Exception:
        env_from_dir, algo_from_dir = None, None

    config_path = run_dir / "config.yml"
    eval_path = run_dir / "eval" / "evaluations.npz"
    results_json_path = run_dir / "results.json"

    if not config_path.exists():
        print(f"[collect] No config.yml in {run_dir}, skip.")
        return None

    cfg_info = parse_config(config_path)

    # 如果 config 没有 env / algo，就退回用目录名
    if cfg_info["env"] is None:
        cfg_info["env"] = env_from_dir
    if cfg_info["algo"] is None:
        cfg_info["algo"] = algo_from_dir

    # metric: max_eval_return
    max_eval_return = parse_evaluations(eval_path)
    cfg_info[METRIC_COL] = max_eval_return

    # 如果你在 results.json 里有额外 summary，也可以在这里补充
    if results_json_path.exists():
        try:
            with results_json_path.open("r") as f:
                res = json.load(f)
            # 示例：如果以后想用，可以取消注释
            # cfg_info["final_train_return"] = res.get("final_train_return", None)
        except Exception:
            pass

    cfg_info["run_dir"] = str(run_dir)
    return cfg_info


# --------------------------------------------------------------------------- #
# 扫描所有 run 目录
# --------------------------------------------------------------------------- #


def scan_all_runs() -> pd.DataFrame:
    """
    遍历所有 runs_seed* 目录下的 env/algo/run_id 结构，收集成一个 DataFrame。

    期望总结构：
      <PROJECT_ROOT>/
        runs_seed0/
          <env>/
            <algo>/
              <run_id>/
                config.yml
                eval/evaluations.npz
        runs_seed1/
          <env>/
            <algo>/
              <run_id>/
                ...
        ...

    这里约定 LOG_ROOT 指向的是包含 runs_seed* 的上级目录，
    比如：/homes/.../thesis_project
    """
    rows: list[Dict[str, Any]] = []

    if not LOG_ROOT.exists():
        raise FileNotFoundError(f"LOG_ROOT does not exist: {LOG_ROOT}")

    # 遍历所有 runs_seed* 目录
    seed_dirs = [d for d in LOG_ROOT.glob("runs_seed*") if d.is_dir()]
    if not seed_dirs:
        print(f"[collect] No runs_seed* directories found under {LOG_ROOT}")
        return pd.DataFrame()

    for seed_dir in seed_dirs:
        print(f"[collect] Scanning seed dir: {seed_dir}")
        for env_dir in seed_dir.iterdir():
            if not env_dir.is_dir():
                continue
            for algo_dir in env_dir.iterdir():
                if not algo_dir.is_dir():
                    continue
                for run_dir in algo_dir.iterdir():
                    if not run_dir.is_dir():
                        continue
                    row = collect_single_run(run_dir)
                    if row is not None:
                        rows.append(row)

    if not rows:
        print("[collect] No runs found under runs_seed* directories, DataFrame is empty.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df



# --------------------------------------------------------------------------- #
# 主函数：构建汇总表
# --------------------------------------------------------------------------- #

def build_consolidated_table() -> None:
    """
    主入口：扫描所有 run，生成 interaction_metrics.csv。
    """
    df = scan_all_runs()
    out_path = ANALYSIS_OUT_DIR / "interaction_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"[collect] Saved consolidated metrics to {out_path}")


if __name__ == "__main__":
    build_consolidated_table()
print("[collect_runs] Done."   )

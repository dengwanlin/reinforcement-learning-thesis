# hyperparameter_interaction_effects/collect_runs.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import yaml

from .config import LOG_ROOT, ANALYSIS_OUT_DIR, METRIC_COL


def parse_config(config_path: Path) -> Dict[str, Any]:
    """
    解析单个 run 的 config.yml，提取 env、algo 和关键超参数。
    配合 RL Baselines3 Zoo 风格的配置做了默认假设，你可以按实际字段名微调。
    """
    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)

    env_id = cfg.get("env", None)
    algo = cfg.get("algo", None)

    # 超参数通常在 "hyperparams" 下面
    hp = cfg.get("hyperparams", {}) or {}

    # 这里假设 config.yml 中 key 就是这些名字，如果不一样你可以调整
    learning_rate = hp.get("learning_rate", None)
    n_steps = hp.get("n_steps", None)
    gamma = hp.get("gamma", None)
    gae_lambda = hp.get("gae_lambda", None)
    ent_coef = hp.get("ent_coef", None)
    clip_range = hp.get("clip_range", None)
    batch_size = hp.get("batch_size", None)

    seed = cfg.get("seed", None)

    row = {
        "env": env_id,
        "algo": algo,
        "learning_rate": float(learning_rate) if learning_rate is not None else None,
        "n_steps": int(n_steps) if n_steps is not None else None,
        "gamma": float(gamma) if gamma is not None else None,
        "gae_lambda": float(gae_lambda) if gae_lambda is not None else None,
        "ent_coef": float(ent_coef) if ent_coef is not None else None,
        "clip_range": float(clip_range) if clip_range is not None else None,
        "batch_size": int(batch_size) if batch_size is not None else None,
        "seed": int(seed) if seed is not None else None,
    }
    return row


def parse_evaluations(eval_path: Path) -> Optional[float]:
    """
    解析 eval/evaluations.npz，计算 max_eval_return：
        对 results 按 env 平均，然后取最大值。
    """
    if not eval_path.exists():
        return None

    data = np.load(eval_path)
    results = data.get("results", None)
    if results is None:
        return None

    mean_over_envs = results.mean(axis=1)   # shape: (n_eval,)
    max_eval = float(mean_over_envs.max())
    return max_eval


def collect_single_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    """
    从单个 run 目录抽取一行记录：
      env, algo, seed, hyperparams, metric 等。
    """
    # 从路径推一份 env/algo，防止 config 里没有写
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

    # 补 env / algo
    if cfg_info["env"] is None:
        cfg_info["env"] = env_from_dir
    if cfg_info["algo"] is None:
        cfg_info["algo"] = algo_from_dir

    # metric
    max_eval_return = parse_evaluations(eval_path)
    cfg_info[METRIC_COL] = max_eval_return

    # 你如果在 results.json 里存了别的 summary，可以在这里补充：
    if results_json_path.exists():
        try:
            with results_json_path.open("r") as f:
                res = json.load(f)
            # 例如：cfg_info["episode_reward_mean"] = res.get("ep_rew_mean", None)
        except Exception:
            pass

    cfg_info["run_dir"] = str(run_dir)
    return cfg_info


def scan_all_runs() -> pd.DataFrame:
    """
    遍历 LOG_ROOT 下所有 env/algo/run_id 目录，收集成一个 DataFrame。
    期望结构：
      LOG_ROOT /
        env /
          algo /
            run_id /
              config.yml
              eval/evaluations.npz
              ...
    """
    rows = []

    if not LOG_ROOT.exists():
        raise FileNotFoundError(f"LOG_ROOT does not exist: {LOG_ROOT}")

    for env_dir in LOG_ROOT.iterdir():
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
        print("[collect] No runs found under LOG_ROOT, resulting DataFrame is empty.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def build_consolidated_table() -> None:
    """
    主函数：扫描所有 run，生成 interaction_metrics.csv。
    """
    df = scan_all_runs()
    out_path = ANALYSIS_OUT_DIR / "interaction_metrics.csv"
    df.to_csv(out_path, index=False)
    print(f"[collect] Saved consolidated metrics to {out_path}")


if __name__ == "__main__":
    build_consolidated_table()

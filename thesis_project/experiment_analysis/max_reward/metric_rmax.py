#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
metric_rmax.py

负责计算一个 run（目录）下的 `eval/evaluations.npz` 中的最大评估奖励 R_max。
返回：
    R_max, t_at_max
"""

from pathlib import Path
import numpy as np


def compute_rmax_for_run(eval_file: Path):
    """
    Input:
        eval_file: Path('.../eval/evaluations.npz')

    Returns:
        (R_max, t_at_max)
        或 None (文件不存在或格式不正确)
    """
    if not eval_file.exists():
        return None

    try:
        data = np.load(eval_file, allow_pickle=True)
    except:
        return None

    # SB3 evaluations.npz 常见字段
    #   results   -> shape (num_eval, num_episode_per_eval)
    #   timesteps -> shape (num_eval,)
    if "results" not in data or "timesteps" not in data:
        return None

    results = data["results"]          # 每次评估的多次 episode reward
    timesteps = data["timesteps"]      # 每次评估对应的 step

    # 计算每次评估的平均奖励
    eval_means = results.mean(axis=1)  # shape (num_eval,)

    # 找最大 reward
    idx = int(np.argmax(eval_means))
    R_max = float(eval_means[idx])
    t_at_max = int(timesteps[idx])

    return R_max, t_at_max

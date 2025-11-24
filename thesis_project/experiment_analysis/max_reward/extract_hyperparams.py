#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_hyperparams.py

负责从 config.yml 中提取所有关心的超参数：
  learning_rate
  n_steps
  batch_size
  n_epochs
  gamma
  clip_range
  gae_lambda
  ent_coef
  vf_coef
  max_grad_norm
  target_kl
  clip_range_vf
  use_sde
  sde_sample_freq

并展开 policy_kwargs：
  policy_kwargs.net_arch
  policy_kwargs.ortho_init
  policy_kwargs.log_std_init
"""

from pathlib import Path
import yaml
import json


# 你确认的最终需要分析的完整超参数清单
HYPERPARAM_KEYS = [
    "learning_rate",
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "clip_range",
    "gae_lambda",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "target_kl",
    "clip_range_vf",
    "use_sde",
    "sde_sample_freq",
]

# 需要展开的 policy_kwargs 子字段
POLICY_KWARGS_KEYS = [
    "net_arch",
    "ortho_init",
    "log_std_init",
]


def load_config_yaml(path: Path):
    """加载 YAML 文件并返回 dict，失败返回 None"""
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except:
        return None


def normalize_net_arch(value):
    """
    将 net_arch 标准化为字符串形式。
    例如：
        [64, 64] → "[64,64]"
        [256, 256] → "[256,256]"
    """
    if value is None:
        return None
    try:
        if isinstance(value, (list, tuple)):
            return json.dumps(value)
        return str(value)
    except:
        return str(value)


def extract_hyperparams(config_path: Path):
    """
    输入：
        config_path → run_dir/config.yml

    返回：
        (seed, hyperparams_dict)

    hyperparams_dict 会包含全部超参数字段 + policy_kwargs 子字段
    """
    cfg = load_config_yaml(config_path)
    if cfg is None:
        return None, {}

    seed = cfg.get("seed", None)

    # SB3 runner 的超参一般在 hyperparams_parsed
    hp_raw = cfg.get("hyperparams_parsed", {})
    hp = {}

    # 1. 导出你确认的所有顶层超参
    for key in HYPERPARAM_KEYS:
        hp[key] = hp_raw.get(key, None)

    # 2. 展开 policy_kwargs
    pk = hp_raw.get("policy_kwargs", {})
    for key in POLICY_KWARGS_KEYS:
        val = pk.get(key, None)
        if key == "net_arch":
            val = normalize_net_arch(val)
        hp[f"policy_kwargs.{key}"] = val

    return seed, hp

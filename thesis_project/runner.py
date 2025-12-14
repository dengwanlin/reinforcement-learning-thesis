#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Custom Stable-Baselines3 runner
--------------------------------
Creates unique run directory:
    #<ROOT>/<ENV>/<ALGO>/<YYYYmmdd_HHMMSS_%f_pidXXXX_seedY>
    <ROOT>/<ENV>/<ALGO>/<YYYYmmdd_HHMMSS>/<arguments_tags>
Automatically saves:
    - config.yml
    - model.zip
    - monitor.csv / vecmonitor.csv
    - evaluations.npz
    - tb/ (TensorBoard logs)
    - results.json
    -log/progress.log (stdout log)

Supported algorithms: A2C, PPO

Examples:
    python runner.py --algo ppo --env LunarLander-v3
    python runner.py --algo ppo --env LunarLanderContinuous-v2 \
        --hyperparams learning_rate:3e-4 ent_coef:0.0 policy_kwargs:dict(net_arch=[256,256])--seed 42
"""

from __future__ import annotations
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import yaml
import warnings
import uuid
import sys
import gymnasium as gym

import random
import numpy as np
import torch

from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure as sb3_logger_config

# Root directory for all runs
#ROOT = Path("/homes/sohawan2/reinforcement-learning-thesis/thesis_project/runs")
BASE_ROOT = Path("/homes/sohawan2/reinforcement-learning-thesis/thesis_project")


ALGO_MAP = dict(ppo=PPO, a2c=A2C)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def linear_schedule(initial_value: float):
    """Return a callable that linearly decreases from the initial value."""
    initial_value = float(initial_value)
    def func(progress_remaining: float):
        return progress_remaining * initial_value
    return func

SAFE_EVAL_ENV = {"__builtins__": None, "linear_schedule": linear_schedule}

# Environment compatibility mapping
ENV_COMPAT = {
    "LunarLander-v2": "LunarLander-v3",
    "LunarLanderContinuous-v2": "LunarLanderContinuous-v3",
}


def resolve_env_id(e: str) -> str:
    """Map outdated env IDs to the latest version if needed."""
    r = ENV_COMPAT.get(e, e)
    if r != e:
        warnings.warn(f"[compat] {e} -> {r}")
    return r


def parse_kv_list(kv_list):
    """
    Parse ["key:val", "k2:val2"] into a dictionary.
    Supports automatic parsing of numeric, bool, None, and simple expressions, e.g.:
        learning_rate:linear_schedule(3e-4)
        policy_kwargs:dict(net_arch=[256,256])
    """
    out = {}
    if not kv_list:
        return out

    SAFE_EVAL_ENV = {
        "__builtins__": None,
        "linear_schedule": linear_schedule,
        "True": True,
        "False": False,
        "None": None,
        "dict": dict,
        "list": list,
    }

    for item in kv_list:
        if ":" not in item:
            raise ValueError(f"--hyperparams item must be key:val, got: {item}")
        k, v = item.split(":", 1)
        k = k.strip()
        v = v.strip()

        # Evaluate expressions or booleans
        if v.lower() in {"true", "false", "none"} or any(ch in v for ch in "([{'") or "(" in v:
            try:
                out[k] = eval(v, SAFE_EVAL_ENV, {})
            except Exception as e:
                raise ValueError(f"Error parsing hyperparam {k}:{v} -> {e}")
        else:
            # Try numeric, fallback to string
            try:
                out[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
            except ValueError:
                out[k] = v
    return out


def build_env(env_id: str, seed: int, normalize: bool, monitor_dir: Path):
    """
    Create a monitored (and optionally normalized) VecEnv with a single environment.
    - Monitor writes per-episode rewards and lengths to monitor.csv
    - VecNormalize optionally normalizes observations and rewards
    """
    monitor_dir.mkdir(parents=True, exist_ok=True)

    def _make():
        env = gym.make(env_id)
        try:
            env.reset(seed=seed)
        except TypeError:
            pass
        env = Monitor(env, str(monitor_dir / "monitor.csv"))
        return env

    venv = DummyVecEnv([_make])
    venv = VecMonitor(venv, filename=str(monitor_dir / "vecmonitor.csv"))

    if normalize:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return venv


def to_yaml_safe(obj):
    """Convert arbitrary Python objects to YAML-safe representation."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_yaml_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_yaml_safe(v) for k, v in obj.items()}
    if callable(obj):
        name = getattr(obj, "__name__", None) or obj.__class__.__name__
        if name == "func":
            name = "linear_schedule()"
        return f"<callable:{name}>"
    return repr(obj)

# ---------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser("Custom SB3 Runner")
    parser.add_argument("--algo", required=True, choices=ALGO_MAP.keys(),
                        help="Algorithm to use, e.g. ppo or a2c.")
    parser.add_argument("--env", required=True,
                        help="Environment ID, e.g. CartPole-v1 or LunarLander-v3.")
    parser.add_argument("--hyperparams", nargs="+", default=[],
                        help="Optional hyperparameters in key:val format.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Fixed defaults (not exposed as CLI args)
    # -----------------------------------------------------------------------
    seed = args.seed
    device = "auto"
    normalize = False
    eval_freq = 20_000
    eval_episodes = 10
    save_freq = 100_000
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    # automatically set steps based on algo and env
    DEFAULT_STEPS = {
        ("ppo", "CartPole-v1"): 200_000,
        ("a2c", "CartPole-v1"): 300_000,
        ("ppo", "LunarLander-v3"): 1_000_000,
        ("a2c", "LunarLander-v3"): 1_500_000,
        ("ppo", "LunarLanderContinuous-v3"): 1_500_000,
        ("a2c", "LunarLanderContinuous-v3"): 2_000_000,
        ("ppo", "Hopper-v4"): 2_000_000,
        ("a2c", "Hopper-v4"): 3_000_000,
    }
    FALLBACK_STEPS = 1_000_000

    # -----------------------------------------------------------------------
    # Unique run directory
    # -----------------------------------------------------------------------
    seed_root = BASE_ROOT / f"runs_seed{seed}"    
    seed_root.mkdir(parents=True, exist_ok=True)
    env_id = resolve_env_id(args.env)
    now = datetime.now()
    ts_base = now.strftime("%Y%m%d_%H%M%S_%f")
    unique_tag = f"{ts_base}_pid{os.getpid()}_seed{seed}"




    #run_dir = ROOT / env_id / args.algo / unique_tag
    run_dir = seed_root / env_id / args.algo / unique_tag
    run_dir.mkdir(parents=True, exist_ok=True)


    log_dir = run_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / "progress.log"

    print(f"[run dir] {run_dir}")

    # -----------------------------------------------------------------------
    # auto-configure n_timesteps
    # -----------------------------------------------------------------------
    n_timesteps = DEFAULT_STEPS.get((args.algo, env_id), FALLBACK_STEPS)
    print(f"[auto-config] n_timesteps = {n_timesteps:,}  ({args.algo} @ {env_id})")

    # -----------------------------------------------------------------------
    # in the meantime, redirect stdout to both console and log file
    # -----------------------------------------------------------------------
    class Tee(object):
        """Simple stdout/file duplicator"""
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
    #sys.stdout = Tee(sys.__stdout__, open(log_file_path, "w", encoding="utf-8"))
    log_f = open(log_file_path, "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_f)

    # -----------------------------------------------------------------------
    # Parse and save configuration
    # -----------------------------------------------------------------------
    user_hparams = parse_kv_list(args.hyperparams)
    base_cfg = {
        "env_id": env_id,
        "algo": args.algo,
        "seed": seed,
        "device": device,
        "n_timesteps": n_timesteps,
        "normalize": normalize,
        "eval_freq": eval_freq,
        "eval_episodes": eval_episodes,
        "save_freq": save_freq,
        "timestamp": ts_base,
        "pid": os.getpid(),
        "runner": "custom-sb3",
        "hyperparams_raw": args.hyperparams,
        "hyperparams_parsed": to_yaml_safe(user_hparams),
    }
    with open(run_dir / "config.yml", "w") as f:
        yaml.safe_dump(base_cfg, f, allow_unicode=True, sort_keys=False)

    # -----------------------------------------------------------------------
    # Create training and evaluation environments
    # -----------------------------------------------------------------------
    env = build_env(env_id, seed, normalize, run_dir)
    eval_env = build_env(env_id, seed + 100, normalize, run_dir / "eval")

    if normalize:
        eval_env.obs_rms = env.obs_rms
        eval_env.ret_rms = env.ret_rms
        eval_env.training = False

    # -----------------------------------------------------------------------
    # Initialize model
    # -----------------------------------------------------------------------
    Algo = ALGO_MAP[args.algo]
    model_kwargs = dict(
        policy="MlpPolicy",
        env=env,
        tensorboard_log=str(run_dir),
        device=device,
        seed=seed,
        verbose=1,  # default verbose level
        **user_hparams,
    )
    model = Algo(**model_kwargs)

    # -----------------------------------------------------------------------
    # Configure logger
    # -----------------------------------------------------------------------
    logger = sb3_logger_config(
        folder=str(run_dir),
        format_strings=["stdout", "csv", "tensorboard"]
    )
    model.set_logger(logger)

    # -----------------------------------------------------------------------
    # Callbacks
    # -----------------------------------------------------------------------
    callbacks = []
    (run_dir / "best").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best"),
        log_path=str(run_dir / "eval"),
        eval_freq=eval_freq,
        n_eval_episodes=eval_episodes,
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(run_dir / "ckpt"),
        name_prefix="model",
        verbose=1,
    )
    callbacks.extend([eval_cb, ckpt_cb])

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    print(f"[run] {env_id} | {args.algo} | total_timesteps={n_timesteps:,}")
    model.learn(
        total_timesteps=int(n_timesteps),
        tb_log_name="tb",
        callback=callbacks,
        progress_bar=True,   # show progress bar
    )

    # -----------------------------------------------------------------------
    # Evaluation & Saving
    # -----------------------------------------------------------------------
    mean_r, std_r = evaluate_policy(
        model, eval_env, n_eval_episodes=eval_episodes, deterministic=True
    )
    print(f"[eval] mean={mean_r:.2f} ± {std_r:.2f} over {eval_episodes} episodes")

    model.save(str(run_dir / f"{args.env}.zip"))
    if normalize:
        env.save(str(run_dir / "vecnormalize.pkl"))

    summary = {"mean_reward": float(mean_r), "std_reward": float(std_r)}
    with open(run_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] TensorBoard logs saved to: {run_dir}")


if __name__ == "__main__":
    main()

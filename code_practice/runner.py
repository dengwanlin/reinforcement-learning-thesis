#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Custom Stable-Baselines3 runner
--------------------------------
Creates unique run directory:
    <ROOT>/<ENV>/<ALGO>/<YYYYmmdd_HHMMSS_%f_pidXXXX_seedY>

Automatically saves:
    - config.yml
    - model.zip
    - monitor.csv / vecmonitor.csv
    - evaluations.npz
    - tb/ (TensorBoard logs)
    - results.json

Supported algorithms: A2C, PPO, DQN, SAC, TD3

Examples:
    python runner.py --algo ppo --env CartPole-v1 -n 1e5 --progress
    python runner.py --algo a2c --env LunarLander-v2 -n 2e5 --eval-freq 10000 --progress
    python runner.py --algo sac --env Hopper-v4 -n 2e6 --device cuda --progress
    python runner.py --algo ppo --env LunarLanderContinuous-v2 -n 1e6 \
        --hyperparams learning_rate:3e-4 ent_coef:0.0 policy_kwargs:dict(net_arch=[256,256])
"""

from __future__ import annotations
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import yaml
import warnings

import gymnasium as gym
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure as sb3_logger_config

# Root directory for all runs
ROOT = Path("/homes/sohawan2/reinforcement-learning-thesis/code_practice/runs")

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
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("Custom SB3 Runner")
    parser.add_argument("--algo", required=True, choices=ALGO_MAP.keys())
    parser.add_argument("--env", required=True)
    parser.add_argument("-n", "--n-timesteps", default="1e6",
                        help="Total timesteps (accepts scientific notation, e.g., 1e6)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--normalize", action="store_true",
                        help="Enable VecNormalize for obs/reward normalization")
    parser.add_argument("--eval-freq", type=int, default=None,
                        help="Evaluate every N steps and save best model")
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--save-freq", type=int, default=None,
                        help="Save checkpoints every N steps")
    parser.add_argument("--hyperparams", nargs="+", default=[],
                        help="Override hyperparameters as key:val pairs")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Unique run directory
    # -----------------------------------------------------------------------
    env_id = resolve_env_id(args.env)
    now = datetime.now()
    print(f"printing now {now}")
    ts_base = now.strftime("%Y%m%d_%H%M%S_%f_%f")# use the uuid to replace pid and seed 
    unique_tag = f"{ts_base}_pid{os.getpid()}_seed{args.seed}"
    run_dir = ROOT / env_id / args.algo / unique_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[unique tag] {unique_tag}")
    print(f"[run dir] {run_dir}")
    #add print to check the run directory
    # -----------------------------------------------------------------------
    # Parse and save configuration
    # -----------------------------------------------------------------------
    user_hparams = parse_kv_list(args.hyperparams)
    base_cfg = {
        "env_id": env_id,
        "algo": args.algo,
        "seed": args.seed,
        "device": args.device,
        "n_timesteps": args.n_timesteps,
        "normalize": args.normalize,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "save_freq": args.save_freq,
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
    env = build_env(env_id, args.seed, args.normalize, run_dir)
    eval_env = build_env(env_id, args.seed + 100, args.normalize, run_dir / "eval")

    # Share normalization stats between train and eval
    if args.normalize:
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
        device=args.device,
        seed=args.seed,
        verbose=1 if args.progress else 0,
        **user_hparams,
    )
    model = Algo(**model_kwargs)

    # -----------------------------------------------------------------------
    # Configure SB3 logger
    # -----------------------------------------------------------------------
    logger = sb3_logger_config(
        folder=str(run_dir),
        format_strings=["stdout", "csv", "tensorboard"]
    )
    model.set_logger(logger)

    # -----------------------------------------------------------------------
    # Prepare callbacks
    # -----------------------------------------------------------------------
    callbacks = []
    (run_dir / "best").mkdir(parents=True, exist_ok=True)
    (run_dir / "eval").mkdir(parents=True, exist_ok=True)
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)

    if args.eval_freq:
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(run_dir / "best"),
            log_path=str(run_dir / "eval"),
            eval_freq=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            deterministic=True,
            render=False,
        )
        callbacks.append(eval_cb)
    if args.save_freq:
        ckpt_cb = CheckpointCallback(
            save_freq=args.save_freq,
            save_path=str(run_dir / "ckpt"),
            name_prefix="model",
            verbose=1,
        )
        callbacks.append(ckpt_cb)

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    print(f"[run] {env_id} | {args.algo} | saving to: {run_dir}")
    total_timesteps = int(float(args.n_timesteps))
    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name="tb",
        callback=callbacks or None,
        progress_bar=args.progress,
    )

    # -----------------------------------------------------------------------
    # Evaluation and saving
    # -----------------------------------------------------------------------
    mean_r, std_r = evaluate_policy(
        model, eval_env, n_eval_episodes=args.eval_episodes, deterministic=True
    )
    print(f"[eval] mean={mean_r:.2f} ± {std_r:.2f} over {args.eval_episodes} episodes")

    model.save(str(run_dir / f"{args.env}.zip"))

    if args.normalize:
        env.save(str(run_dir / "vecnormalize.pkl"))

    summary = {"mean_reward": float(mean_r), "std_reward": float(std_r)}
    with open(run_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] TensorBoard: tensorboard --logdir {ROOT}")


if __name__ == "__main__":
    main()

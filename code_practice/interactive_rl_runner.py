#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive SB3 runner (CartPole-v1 & LunarLander-v3)
- Terminal prompts for env/algorithm/num_envs/VecNormalize
- Per (env, algo) default hyperparameters
- Optional VecNormalize
- Output: {PROJECT_ROOT}/runs/{ENV}/{ALGO}/{TIMESTAMP[-run_name]}/...

Project root resolution priority:
  1) Env var RL_PROJECT_ROOT (if set and writable)
  2) Preferred path: /homes/sohawan2/reinforcement-learning-thesis/code_practice
  3) Fallback: <script_dir>/rl_runs_root
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.evaluation import evaluate_policy

# ---------- Project root resolution ----------
PREFERRED_ROOT = Path("/homes/sohawan2/reinforcement-learning-thesis/code_practice")

def is_writable(path: Path) -> bool:
    """Check directory writability by creating/removing a temp file."""
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_ok.tmp"
        with open(probe, "w") as f:
            f.write("ok")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False

def resolve_project_root() -> Path:
    env_root = os.getenv("RL_PROJECT_ROOT")
    if env_root:
        cand = Path(env_root).expanduser().resolve()
        if is_writable(cand):
            print(f"[Path] Using RL_PROJECT_ROOT: {cand}")
            return cand
        else:
            print(f"[Path] RL_PROJECT_ROOT not writable, ignoring: {cand}")
    pref = PREFERRED_ROOT.expanduser().resolve()
    if is_writable(pref):
        print(f"[Path] Using preferred root: {pref}")
        return pref
    print(f"[Path] Preferred root not usable, falling back: {pref}")
    fallback = (Path(os.path.dirname(os.path.abspath(__file__))) / "rl_runs_root").resolve()
    if is_writable(fallback):
        print(f"[Path] Fallback root: {fallback}")
        return fallback
    print("[Path] ERROR: No writable project root found.", file=sys.stderr)
    sys.exit(1)

PROJECT_ROOT = resolve_project_root()

# ---------- Options ----------
ENVS  = ["CartPole-v1", "LunarLander-v3"]
ALGOS = ["DQN", "PPO", "A2C"]

CONFIGS = {
    "CartPole-v1": {
        "stop_threshold": 475.0,
        "DQN": {
            "model_kwargs": dict(
                learning_rate=1e-3,
                buffer_size=50_000,
                learning_starts=1_000,
                batch_size=64,
                gamma=0.99,
                train_freq=(4, "step"),
                gradient_steps=1,
                target_update_interval=500,
                exploration_fraction=0.20,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.02,
                policy_kwargs=dict(net_arch=[256, 256]),
            ),
            "train": dict(total_timesteps=300_000, eval_freq=10_000, ckpt_freq=50_000),
        },
        "PPO": {
            "model_kwargs": dict(
                n_steps=1024,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                learning_rate=3e-4,
                clip_range=0.2,
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=[128, 128]),
            ),
            "train": dict(total_timesteps=300_000, eval_freq=10_000, ckpt_freq=50_000),
        },
        "A2C": {
            "model_kwargs": dict(
                n_steps=5,
                gamma=0.99,
                learning_rate=7e-4,
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=[128, 128]),
            ),
            "train": dict(total_timesteps=300_000, eval_freq=10_000, ckpt_freq=50_000),
        },
    },
    "LunarLander-v3": {
        "stop_threshold": 200.0,
        "DQN": {
            "model_kwargs": dict(
                learning_rate=5e-4,
                buffer_size=100_000,
                learning_starts=10_000,
                batch_size=64,
                gamma=0.99,
                train_freq=(4, "step"),
                gradient_steps=1,
                target_update_interval=1_000,
                exploration_fraction=0.30,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                policy_kwargs=dict(net_arch=[256, 256]),
            ),
            "train": dict(total_timesteps=1_000_000, eval_freq=20_000, ckpt_freq=100_000),
        },
        "PPO": {
            "model_kwargs": dict(
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                learning_rate=3e-4,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=[256, 256]),
            ),
            "train": dict(total_timesteps=1_000_000, eval_freq=20_000, ckpt_freq=100_000),
        },
        "A2C": {
            "model_kwargs": dict(
                n_steps=5,
                gamma=0.99,
                learning_rate=7e-4,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=[256, 256]),
            ),
            "train": dict(total_timesteps=800_000, eval_freq=20_000, ckpt_freq=100_000),
        },
    },
}

# ---------- Interactive helpers ----------
def ask_choice(title: str, options: list[str]) -> str:
    print(f"\n== Choose {title} ==")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    while True:
        raw = input(f"Enter index 1~{len(options)}: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("Invalid input, try again.")

def ask_int(title: str, default: int) -> int:
    raw = input(f"{title} (default {default}): ").strip()
    if raw == "":
        return default
    try:
        return int(raw.replace("_", ""))
    except Exception:
        print("Invalid input, using default.")
        return default

def ask_yes_no(title: str, default: bool=False) -> bool:
    raw = input(f"{title} (y/n, default {'y' if default else 'n'}): ").strip().lower()
    if raw == "": return default
    return raw in ["y","yes","1"]

def ask_str(title: str, default: str|None=None) -> str|None:
    raw = input(f"{title}{'' if default is None else f' (default {default})'}: ").strip()
    return default if raw=="" else raw

# ---------- Builders ----------
def make_env(env_id: str, seed: int=0):
    """Return a bare gym env factory (VecMonitor will handle logging)."""
    def _init():
        env = gym.make(env_id)
        env.reset(seed=seed)
        return env
    return _init

def build_paths(env_id: str, algo: str, run_name: str|None):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    env_tag = env_id.replace("-", "_")
    tag = ts if not run_name else f"{ts}-{run_name}"
    base = PROJECT_ROOT / "runs" / env_tag / algo.upper() / tag
    paths = {
        "base": base,
        "tb": base / "tb",
        "best": base / "best",
        "ckpt": base / "ckpt",
        "models": base / "models",
        "monitor_train": base / "monitor_train.csv",
        "monitor_eval": base / "monitor_eval.csv",
        "vecnormalize": base / "vecnormalize.pkl",
        "config": base / "config.json",
    }
    for k in ["tb","best","ckpt","models"]:
        paths[k].mkdir(parents=True, exist_ok=True)
    return paths

def build_model(algo: str, env, tb_log: Path, seed: int, model_kwargs: dict):
    """DQN/PPO on auto device; A2C on CPU by default (faster for MLP)."""
    common = dict(tensorboard_log=str(tb_log), verbose=1, seed=seed)
    if algo=="DQN": return DQN("MlpPolicy", env, device="auto", **model_kwargs, **common)
    if algo=="PPO": return PPO("MlpPolicy", env, device="auto", **model_kwargs, **common)
    if algo=="A2C": return A2C("MlpPolicy", env, device="cpu",  **model_kwargs, **common)
    raise ValueError(f"Unsupported algo: {algo}")

# ---------- Main ----------
def main():
    print("\n===== Stable-Baselines3 Interactive Runner =====")
    print(f"[Path] Project root: {PROJECT_ROOT}")

    env_id = ask_choice("Environment", ENVS)
    algo   = ask_choice("Algorithm", ALGOS)

    env_cfg  = CONFIGS[env_id]
    algo_cfg = env_cfg[algo]
    stop_threshold = env_cfg["stop_threshold"]

    total_timesteps = ask_int("Total timesteps", algo_cfg["train"]["total_timesteps"])
    seed            = ask_int("Random seed", 0)
    eval_freq       = ask_int("Eval frequency",  algo_cfg["train"]["eval_freq"])
    ckpt_freq       = ask_int("Checkpoint frequency", algo_cfg["train"]["ckpt_freq"])
    num_envs        = ask_int("Number of parallel envs", 1)
    use_vecnorm     = ask_yes_no("Enable VecNormalize (obs/reward normalization)?", False)
    run_name        = ask_str("Optional run name suffix", None)

    paths = build_paths(env_id, algo, run_name)

    # Build training vec env
    env_fns = [make_env(env_id, seed=seed+i) for i in range(num_envs)]
    train_env = SubprocVecEnv(env_fns) if num_envs>1 else DummyVecEnv(env_fns)
    train_env = VecMonitor(train_env, filename=str(paths["monitor_train"]))

    # Build evaluation vec env (single env is enough)
    eval_env = DummyVecEnv([make_env(env_id, seed=seed+100)])
    eval_env = VecMonitor(eval_env, filename=str(paths["monitor_eval"]))

    # Optional normalization
    if use_vecnorm:
        train_env = VecNormalize(train_env, norm_obs=True,  norm_reward=True,  clip_obs=10.0)
        eval_env  = VecNormalize(eval_env,  norm_obs=True,  norm_reward=False, training=False)

    # Build model and callbacks
    model = build_model(algo, train_env, paths["tb"], seed, algo_cfg["model_kwargs"])

    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=stop_threshold, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(paths["best"]),
        n_eval_episodes=5,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        callback_after_eval=stop_cb,
        verbose=1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=ckpt_freq,
        save_path=str(paths["ckpt"]),
        name_prefix=f"{algo.lower()}_{env_id.replace('-', '_')}",
    )

    # Save run config (use default=str to avoid non-serializable objects)
    run_config = {
        "env_id": env_id,
        "algo": algo,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "eval_freq": eval_freq,
        "ckpt_freq": ckpt_freq,
        "num_envs": num_envs,
        "use_vecnorm": use_vecnorm,
        "stop_threshold": stop_threshold,
        "model_kwargs": algo_cfg["model_kwargs"],
        "paths": {k: str(v) for k, v in paths.items()},
        "time": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
    }
    with open(paths["config"], "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2, default=str)

    print("\n===== Training Start =====")
    print(json.dumps(
        {k: run_config[k] for k in ["env_id","algo","seed","total_timesteps","num_envs","use_vecnorm"]},
        indent=2
    ))

    # Train
    model.learn(total_timesteps=total_timesteps, callback=[eval_cb, ckpt_cb], progress_bar=True)

    # Save VecNormalize stats if used
    if use_vecnorm:
        train_env.save(str(paths["vecnormalize"]))
        print(f"VecNormalize stats saved to {paths['vecnormalize']}")

    # Final evaluation
    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
    print(f"\n[{algo}] Final Eval on {env_id}: mean={mean_r:.2f} ± {std_r:.2f}")

    # Save last model
    last_path = paths["models"] / f"last_{algo.lower()}"
    model.save(str(last_path))
    print(f"Saved last model to: {last_path}.zip")

    # Cleanup
    train_env.close()
    eval_env.close()
    print("===== Training End =====\n")


if __name__ == "__main__":
    main()

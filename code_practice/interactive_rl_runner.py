#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive SB3 runner in different environments with different algorithms
- Envs: CartPole-v1 / LunarLander-v3 / SuperMarioBros-1-1-v3 with Super Mario Bros (Gym→Gymnasium bridge, built‑in wrappers)
- Choose env/algorithm/num_envs/VecNormalize via prompts
- Auto CNN policy for image-based envs (Mario), MLP for classic control
- Unified output: {PROJECT_ROOT}/runs/{ENV}/{ALGO}/{TIMESTAMP[-run_name]}/...

Mario stack notes:
- Tested combo: numpy==1.26.4, opencv-python<4.12 (e.g. 4.11.0.86), nes-py==8.1.8, gym-super-mario-bros==7.3.0
- We bridge JoypadSpace (old gym wrapper) → gymnasium.Env via GymToGymnasiumEnv
- Image pipeline (built-in fallback if gymnasium wrappers missing):
  FrameSkipCompat → GrayScaleCompat → ResizeCompat(84,84) → FrameStackCompat → VecTransposeImage
  If gymnasium wrappers exist we still prefer our compat wrappers to avoid version drift.
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import gymnasium as gym
import gym as oldgym  # for space conversion in the adapter

from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
    VecTransposeImage,
)
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.evaluation import evaluate_policy

# --- Try to import Mario core (only core pkgs) ---
try:
    import gym_super_mario_bros  # core
    from nes_py.wrappers import JoypadSpace  # core
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT  # core
    _MARIO_AVAILABLE = True
except Exception:
    _MARIO_AVAILABLE = False

# ---------- Project root resolution ----------
PREFERRED_ROOT = Path("/homes/sohawan2/reinforcement-learning-thesis/code_practice")

def is_writable(path: Path) -> bool:
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
ENVS  = ["CartPole-v1", "LunarLander-v3", "SuperMarioBros-1-1-v3"]  # ← for Mario we use v3
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
    "SuperMarioBros-1-1-v3": {
        "stop_threshold": float("inf"),
        "DQN": {
            "model_kwargs": dict(
                learning_rate=1e-4,
                buffer_size=100_000,
                learning_starts=20_000,
                batch_size=32,
                gamma=0.99,
                train_freq=(4, "step"),
                gradient_steps=1,
                target_update_interval=10_000,
                exploration_fraction=0.10,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.01,
                policy_kwargs=dict(),  # default Nature-CNN
            ),
            "train": dict(total_timesteps=3_000_000, eval_freq=200_000, ckpt_freq=200_000),
        },
        "PPO": {
            "model_kwargs": dict(
                n_steps=128,
                batch_size=256,
                n_epochs=4,
                gamma=0.99,
                gae_lambda=0.95,
                learning_rate=2.5e-4,
                clip_range=0.1,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=dict(),  # default CNN
            ),
            "train": dict(total_timesteps=2_000_000, eval_freq=200_000, ckpt_freq=200_000),
        },
        "A2C": {
            "model_kwargs": dict(
                n_steps=5,
                gamma=0.99,
                learning_rate=7e-4,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=dict(),  # default CNN
            ),
            "train": dict(total_timesteps=2_000_000, eval_freq=200_000, ckpt_freq=200_000),
        },
    },
}

# ---------- Helpers ----------
def is_image_env(env_id: str) -> bool:
    return env_id.startswith("SuperMarioBros")

def policy_for(env_id: str) -> str:
    return "CnnPolicy" if is_image_env(env_id) else "MlpPolicy"

# ---------- Gym→Gymnasium adapter (bridge JoypadSpace to gymnasium.Env) ----------
from gymnasium.spaces import (
    Box as GmBox, Discrete as GmDiscrete,
    MultiBinary as GmMultiBinary, MultiDiscrete as GmMultiDiscrete
)
def _to_gymnasium_space(space):
    if isinstance(space, oldgym.spaces.Box):
        return GmBox(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
    if isinstance(space, oldgym.spaces.Discrete):
        return GmDiscrete(space.n)
    if isinstance(space, oldgym.spaces.MultiBinary):
        return GmMultiBinary(space.n)
    if isinstance(space, oldgym.spaces.MultiDiscrete):
        return GmMultiDiscrete(space.nvec)
    return space

class GymToGymnasiumEnv(gym.Env):
    """Wrap a Gym (0.26.x) env so Gymnasium wrappers accept it."""
    metadata = {"render_modes": []}
    def __init__(self, old_env):
        super().__init__()
        self._env = old_env
        self.observation_space = _to_gymnasium_space(getattr(old_env, "observation_space", None))
        self.action_space      = _to_gymnasium_space(getattr(old_env, "action_space", None))
        self.metadata    = getattr(old_env, "metadata", self.metadata)
        self.reward_range = getattr(old_env, "reward_range", (-float("inf"), float("inf")))
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            try: self._env.reset(seed=seed)
            except TypeError:
                try: self._env.seed(seed)
                except Exception: pass
        out = self._env.reset()
        if isinstance(out, tuple) and len(out) == 2:
            return out
        return out, {}
    def step(self, action):
        out = self._env.step(action)
        if len(out) == 5:  # already gymnasium-style
            return out
        obs, reward, done, info = out
        terminated = bool(done)
        truncated  = bool(info.get("TimeLimit.truncated", False))
        return obs, reward, terminated, truncated, info
    def render(self, *a, **k): return self._env.render(*a, **k)
    def close(self): return self._env.close()
    def __getattr__(self, name): return getattr(self._env, name)

# ---------- Built-in pixel wrappers (compat, independent of gymnasium wrappers) ----------
try:
    import cv2
except Exception as e:
    raise RuntimeError("OpenCV is required (pip install opencv-python<4.12 is recommended).") from e

def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if isinstance(arr, np.ndarray) and arr.dtype == np.uint8:
        return arr
    a = np.array(arr)
    if a.dtype.kind == "f":
        a = np.clip(a, 0.0, 255.0)
        if a.max() <= 1.0:
            a = a * 255.0
        a = a.astype(np.uint8)
    else:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a

class SimpleFrameSkip(gym.Wrapper):
    """Repeat the same action `skip` times. Sum rewards, stop early if done."""
    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        assert skip >= 1
        self._skip = skip
    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

class GrayScaleCompat(gym.ObservationWrapper):
    """Convert RGB frames to grayscale. keep_dim=True -> (H,W,1) uint8."""
    def __init__(self, env: gym.Env, keep_dim: bool = True):
        super().__init__(env)
        self.keep_dim = keep_dim
        old = self.observation_space
        assert isinstance(old, gym.spaces.Box) and len(old.shape) == 3
        h, w, _ = old.shape
        shape = (h, w, 1) if keep_dim else (h, w)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    def observation(self, obs):
        arr = _to_uint8(np.array(obs))
        if arr.ndim != 3 or arr.shape[2] == 1:
            gray = arr
        else:
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        if self.keep_dim:
            gray = gray[..., None]
        return _to_uint8(gray)

class ResizeCompat(gym.ObservationWrapper):
    """Resize frames to target size (H,W), keep channels."""
    def __init__(self, env: gym.Env, size=(84, 84)):
        super().__init__(env)
        self.size = tuple(size)
        old = self.observation_space
        assert isinstance(old, gym.spaces.Box)
        s = old.shape
        if len(s) == 3:
            h, w, c = s; shape = (self.size[0], self.size[1], c)
        else:
            h, w = s; shape = (self.size[0], self.size[1])
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)
    def observation(self, obs):
        arr = _to_uint8(np.array(obs))
        resized = cv2.resize(arr, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
        return _to_uint8(resized)

class FrameStackCompat(gym.Wrapper):
    """Stack last k frames channel-last: (H,W,k) if single-channel."""
    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env)
        assert k >= 1
        from collections import deque
        self.k = k
        self.frames = deque(maxlen=k)
        old = self.observation_space
        assert isinstance(old, gym.spaces.Box)
        s = old.shape
        if len(s) == 2:
            h, w = s; new_shape = (h, w, k)
        else:
            h, w, c = s; new_shape = (h, w, c * k)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        arr = _to_uint8(np.array(obs))
        self.frames.clear()
        for _ in range(self.k):
            self.frames.append(arr)
        return self._get_obs(), info
    def step(self, action):
        obs, r, t, tr, info = self.env.step(action)
        self.frames.append(_to_uint8(np.array(obs)))
        return self._get_obs(), r, t, tr, info
    def _get_obs(self):
        frames = list(self.frames)
        x = frames[0]
        if x.ndim == 2:
            stacked = np.stack(frames, axis=-1)
        else:
            stacked = np.concatenate(frames, axis=2)
        return _to_uint8(stacked)

# ---------- Env builders ----------
def make_env(env_id: str, seed: int=0):
    def _init():
        if is_image_env(env_id):
            if not _MARIO_AVAILABLE:
                raise RuntimeError(
                    "SuperMarioBros requested but gym-super-mario-bros/nes-py not installed. "
                    "Run: pip install gym-super-mario-bros nes-py opencv-python"
                )
            # Raw env + joypad (old gym), then bridge to gymnasium
            env = gym_super_mario_bros.make(env_id, apply_api_compatibility=True, render_mode=None)
            env = JoypadSpace(env, SIMPLE_MOVEMENT)      # old gym wrapper
            env = GymToGymnasiumEnv(env)                 # bridge → gymnasium.Env

            # Compose preprocessing: skip -> gray -> resize(84,84) -> stack(4)
            env = SimpleFrameSkip(env, skip=4)
            env = GrayScaleCompat(env, keep_dim=True)    # (H,W,1)
            env = ResizeCompat(env, size=(84, 84))       # (84,84,1)
            env = FrameStackCompat(env, k=4)             # (84,84,4)
        else:
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

def build_model(algo: str, env, tb_log: Path, seed: int, model_kwargs: dict, policy: str):
    common = dict(tensorboard_log=str(tb_log), verbose=1, seed=seed, policy=policy)
    if algo=="DQN": return DQN(env=env, device="auto", **model_kwargs, **common)
    if algo=="PPO": return PPO(env=env, device="auto", **model_kwargs, **common)
    if algo=="A2C":
        device = "auto" if policy == "CnnPolicy" else "cpu"
        return A2C(env=env, device=device, **model_kwargs, **common)
    raise ValueError(f"Unsupported algo: {algo}")

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

    # Disable VecNormalize for pixel envs
    if is_image_env(env_id) and use_vecnorm:
        print("[Warn] VecNormalize is not recommended for pixel-based envs; disabling it.")
        use_vecnorm = False

    paths = build_paths(env_id, algo, run_name)

    # Build training vec env
    env_fns = [make_env(env_id, seed=seed+i) for i in range(num_envs)]
    train_env = SubprocVecEnv(env_fns) if num_envs>1 else DummyVecEnv(env_fns)
    train_env = VecMonitor(train_env, filename=str(paths["monitor_train"]))
    if is_image_env(env_id) and not isinstance(train_env, VecTransposeImage):
        train_env = VecTransposeImage(train_env)  # (H,W,C) -> (C,H,W)

    # Build evaluation vec env (single env)
    eval_env = DummyVecEnv([make_env(env_id, seed=seed+100)])
    eval_env = VecMonitor(eval_env, filename=str(paths["monitor_eval"]))
    if is_image_env(env_id) and not isinstance(eval_env, VecTransposeImage):
        eval_env = VecTransposeImage(eval_env)

    # Optional normalization (not for Mario)
    if use_vecnorm:
        train_env = VecNormalize(train_env, norm_obs=True,  norm_reward=True,  clip_obs=10.0)
        eval_env  = VecNormalize(eval_env,  norm_obs=True,  norm_reward=False, training=False)

    policy = policy_for(env_id)
    model = build_model(algo, train_env, paths["tb"], seed, algo_cfg["model_kwargs"], policy)

    stop_cb = None if stop_threshold == float("inf") else StopTrainingOnRewardThreshold(
        reward_threshold=stop_threshold, verbose=1
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(paths["best"]),
        n_eval_episodes=3 if is_image_env(env_id) else 5,
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

    run_config = {
        "env_id": env_id,
        "algo": algo,
        "policy": policy,
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
        "mario_available": _MARIO_AVAILABLE,
    }
    with open(paths["config"], "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2, default=str)

    print("\n===== Training Start =====")
    print(json.dumps(
        {k: run_config[k] for k in ["env_id","algo","policy","seed","total_timesteps","num_envs","use_vecnorm"]},
        indent=2
    ))

    model.learn(total_timesteps=total_timesteps, callback=[eval_cb, ckpt_cb], progress_bar=True)

    if use_vecnorm:
        train_env.save(str(paths["vecnormalize"]))
        print(f"VecNormalize stats saved to {paths['vecnormalize']}")

    mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=3 if is_image_env(env_id) else 10, deterministic=True)
    print(f"\n[{algo}/{policy}] Final Eval on {env_id}: mean={mean_r:.2f} ± {std_r:.2f}")

    last_path = paths["models"] / f"last_{algo.lower()}"
    model.save(str(last_path))
    print(f"Saved last model to: {last_path}.zip")

    train_env.close()
    eval_env.close()
    print("===== Training End =====\n")


if __name__ == "__main__":
    main()

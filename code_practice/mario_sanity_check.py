#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mario sanity check (one-file version)
- Works with: numpy==1.26.4, nes-py==8.1.8, gym-super-mario-bros==7.3.0,
              opencv-python<4.12 (e.g., 4.11.0.86), gym==0.26.2, gymnasium==1.x
- No external wrappers needed. Includes:
  * GymToGymnasiumEnv (adapter: gym.Env -> gymnasium.Env, with space conversion)
  * FrameSkipCompat / GrayScaleCompat / ResizeCompat / FrameStackCompat
- Expected output: obs shape (84, 84, 4), dtype uint8; 5 steps OK.

Run:
  /homes/sohawan2/miniconda3/envs/rl/bin/python mario_sanity_check.py
"""

from __future__ import annotations

import numpy as np
import cv2
import gymnasium as gym
import gym as oldgym  # old Gym needed for space type checks (nes_py/JoypadSpace)
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# =========================
# Adapter: gym -> gymnasium
# =========================
from gymnasium.spaces import (
    Box as GmBox,
    Discrete as GmDiscrete,
    MultiBinary as GmMultiBinary,
    MultiDiscrete as GmMultiDiscrete,
)

def _to_gymnasium_space(space):
    """Convert old Gym spaces to Gymnasium spaces."""
    if isinstance(space, oldgym.spaces.Box):
        return GmBox(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
    if isinstance(space, oldgym.spaces.Discrete):
        return GmDiscrete(space.n)
    if isinstance(space, oldgym.spaces.MultiBinary):
        return GmMultiBinary(space.n)
    if isinstance(space, oldgym.spaces.MultiDiscrete):
        return GmMultiDiscrete(space.nvec)
    return space  # fallback

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
        if len(out) == 5:
            return out  # already gymnasium style
        obs, reward, done, info = out
        terminated = bool(done)
        truncated  = bool(info.get("TimeLimit.truncated", False))
        return obs, reward, terminated, truncated, info
    def render(self, *a, **k): return self._env.render(*a, **k)
    def close(self): return self._env.close()
    def __getattr__(self, name): return getattr(self._env, name)

# =========================
# Image wrappers (compat)
# =========================
def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    a = arr
    if a.dtype.kind == "f":
        a = np.clip(a, 0.0, 255.0)
        if a.max() <= 1.0:
            a = a * 255.0
        a = a.astype(np.uint8)
    else:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return a

class FrameSkipCompat(gym.Wrapper):
    """Repeat the same action `skip` times. Sum rewards, break on done."""
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env); assert skip >= 1; self._skip = skip
    def step(self, action):
        total_reward = 0.0; terminated = False; truncated = False; info = {}; obs = None
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info

class GrayScaleCompat(gym.ObservationWrapper):
    """RGB -> GRAY; if keep_dim=True: (H,W,1) uint8."""
    def __init__(self, env: gym.Env, keep_dim: bool = True):
        super().__init__(env); self.keep_dim = keep_dim
        old = self.observation_space
        assert isinstance(old, gym.spaces.Box), f"obs space must be Box, got {type(old)}"
        assert len(old.shape) == 3, f"expect (H,W,C), got {old.shape}"
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
    """Resize to (H,W); keeps channels as-is."""
    def __init__(self, env: gym.Env, size=(84, 84)):
        super().__init__(env); self.size = tuple(size)
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
    """Stack last k frames channel-last: (H,W,k) for 1ch input."""
    def __init__(self, env: gym.Env, k: int = 4):
        super().__init__(env); assert k >= 1; self.k = k
        from collections import deque
        self.frames = deque(maxlen=k)
        old = self.observation_space; assert isinstance(old, gym.spaces.Box)
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
        for _ in range(self.k): self.frames.append(arr)
        return self._get_obs(), info
    def step(self, action):
        obs, r, t, tr, info = self.env.step(action)
        self.frames.append(_to_uint8(np.array(obs)))
        return self._get_obs(), r, t, tr, info
    def _get_obs(self):
        frames = list(self.frames); x = frames[0]
        if x.ndim == 2:
            stacked = np.stack(frames, axis=-1)      # (H,W,k)
        else:
            stacked = np.concatenate(frames, axis=2) # (H,W,C*k)
        return _to_uint8(stacked)

# ============
# Main: sanity
# ============
ENV_ID = "SuperMarioBros-1-1-v3"

env = gym_super_mario_bros.make(ENV_ID, apply_api_compatibility=True, render_mode=None)
env = JoypadSpace(env, SIMPLE_MOVEMENT)  # old gym wrapper
env = GymToGymnasiumEnv(env)             # bridge to gymnasium.Env

# Preprocess: skip -> gray -> resize(84) -> stack(4) => (84,84,4) uint8
env = FrameSkipCompat(env, skip=4)
env = GrayScaleCompat(env, keep_dim=True)
env = ResizeCompat(env, size=(84, 84))
env = FrameStackCompat(env, k=4)

obs, info = env.reset(seed=0)
arr = np.array(obs)
print("reset:", arr.shape, arr.dtype, "action_space:", env.action_space)

for i in range(5):
    a = env.action_space.sample()
    obs, r, terminated, truncated, info = env.step(a)
    arr = np.array(obs)
    print(f"step {i}: {arr.shape}, {arr.dtype}, r={r:.2f}, done={terminated or truncated}")
    if terminated or truncated:
        env.reset()

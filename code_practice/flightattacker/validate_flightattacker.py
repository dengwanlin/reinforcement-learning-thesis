#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate a Gymnasium env (default: FlightAttacker-v0):
- Print spec: id, entry_point, max_episode_steps, render_modes
- Print observation/action space details (dtype, shape, bounds)
- Reset with seed, show sample obs stats
- Do a short random rollout and report FPS and done flags
- (Optional) --render to open human render if supported
"""

from __future__ import annotations
import argparse
import time
from typing import Any

import numpy as np
import gymnasium as gym


def space_info(space: gym.Space) -> dict[str, Any]:
    info = {"type": type(space).__name__}
    try:
        info["shape"] = getattr(space, "shape", None)
    except Exception:
        info["shape"] = None

    # dtype
    dtype = getattr(space, "dtype", None)
    if dtype is None:
        # MultiDiscrete/MultiBinary 没有 dtype 概念
        if isinstance(space, gym.spaces.MultiDiscrete):
            dtype = np.int64
        elif isinstance(space, gym.spaces.MultiBinary):
            dtype = np.int8
    info["dtype"] = str(dtype) if dtype is not None else None

    # bounds/choices
    if isinstance(space, gym.spaces.Box):
        try:
            low = np.min(space.low)
            high = np.max(space.high)
            info["low_min"] = float(low)
            info["high_max"] = float(high)
        except Exception:
            info["low_min"] = None
            info["high_max"] = None
    elif isinstance(space, gym.spaces.Discrete):
        info["n"] = space.n
    elif isinstance(space, gym.spaces.MultiDiscrete):
        info["nvec"] = space.nvec.tolist()
    elif isinstance(space, gym.spaces.MultiBinary):
        info["n"] = space.n

    return info


def guess_pixel_obs(obs: np.ndarray) -> dict[str, Any]:
    """
    粗略判断是否像素观测，以及通道位置（HWC/CHW）。
    """
    result = {
        "is_numpy": isinstance(obs, np.ndarray),
        "dtype": str(getattr(obs, "dtype", None)),
        "shape": tuple(getattr(obs, "shape", [])),
        "likely_pixel": False,
        "channel_order_guess": None,  # "HWC" or "CHW" or None
        "value_range": None,
    }
    if not isinstance(obs, np.ndarray):
        return result

    result["value_range"] = (int(np.min(obs)), int(np.max(obs))) if obs.size else None

    # 像素常见 dtype 为 uint8，数值范围 0~255
    if obs.dtype == np.uint8 and obs.ndim in (2, 3):
        result["likely_pixel"] = True
        if obs.ndim == 3:
            h, w, c = obs.shape
            # 常见通道数：1/3/4
            if c in (1, 3, 4):
                result["channel_order_guess"] = "HWC"
        elif obs.ndim == 2:
            result["channel_order_guess"] = "HW"
    else:
        # 有些环境输出 float32 像素（0~1 或 0~255）
        if obs.ndim in (3, 4):
            vals = (float(np.min(obs)), float(np.max(obs))) if obs.size else (None, None)
            if vals[0] is not None:
                # 粗略判断：范围像素化
                if (0.0 <= vals[0] and vals[1] <= 1.0) or (0.0 <= vals[0] and vals[1] <= 255.0):
                    result["likely_pixel"] = True
                    if obs.ndim == 3:
                        if obs.shape[-1] in (1, 3, 4):
                            result["channel_order_guess"] = "HWC"
                        elif obs.shape[0] in (1, 3, 4):
                            result["channel_order_guess"] = "CHW"
                    elif obs.ndim == 4:
                        # 可能是 FrameStack 类输出 (H,W,stack) 或 (stack,H,W)
                        if obs.shape[-1] in (1, 3, 4):
                            result["channel_order_guess"] = "HWC/stack-last?"
                        elif obs.shape[1] in (1, 3, 4):
                            result["channel_order_guess"] = "CHW/stack-first?"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="FlightAttacker-v0", help="Gymnasium env id")
    parser.add_argument("--steps", type=int, default=200, help="Random rollout steps")
    parser.add_argument("--seed", type=int, default=42, help="Seed for reset()")
    parser.add_argument("--render", action="store_true", help="Try human render if supported")
    args = parser.parse_args()

    # 如果需要渲染，优先尝试用 render_mode="human" 创建
    render_mode = "human" if args.render else None
    try:
        env = gym.make(args.env_id, render_mode=render_mode)
    except TypeError:
        # 某些环境不接受 render_mode 作为 make kwarg
        env = gym.make(args.env_id)
    except Exception as e:
        raise RuntimeError(f"Failed to make env '{args.env_id}'. Is it registered and installed?") from e

    spec = env.spec
    print("=== Env Spec ===")
    if spec is not None:
        print(f"id: {spec.id}")
        print(f"entry_point: {spec.entry_point}")
        print(f"max_episode_steps: {getattr(spec, 'max_episode_steps', None)}")
        print(f"reward_threshold: {getattr(spec, 'reward_threshold', None)}")
        print(f"kwargs: {getattr(spec, 'kwargs', None)}")
        try:
            print(f"render_modes: {getattr(spec, 'render_modes', None)}")
        except Exception:
            print("render_modes: <unavailable on this Gymnasium version>")
    else:
        print("spec: None (some custom envs do not set spec)")

    print("\n=== Spaces ===")
    print("Observation space:", env.observation_space)
    print("Action space     :", env.action_space)
    print("Obs space info   :", space_info(env.observation_space))
    print("Act space info   :", space_info(env.action_space))

    print("\n=== Reset & Sample Observation ===")
    obs, info = env.reset(seed=args.seed)
    print("reset: ok")
    try:
        print("info keys:", list(info.keys()))
    except Exception:
        print("info: <non-dict>")

    obs_info = guess_pixel_obs(np.array(obs) if not isinstance(obs, np.ndarray) else obs)
    print("obs sample info:", obs_info)

    if render_mode == "human":
        try:
            env.render()
            print("render(human): called once successfully.")
        except Exception as e:
            print(f"render(human) failed: {e}")

    print("\n=== Random Rollout ===")
    n_steps = max(1, args.steps)
    t0 = time.perf_counter()
    term_count = 0
    trunc_count = 0
    rewards = []

    for i in range(n_steps):
        action = env.action_space.sample()
        step_out = env.step(action)
        if len(step_out) != 5:
            raise RuntimeError(f"Env.step() must return 5 items (obs, reward, terminated, truncated, info), got {len(step_out)}")
        obs, reward, terminated, truncated, info = step_out
        rewards.append(float(reward))
        if terminated:
            term_count += 1
            obs, info = env.reset()
        if truncated:
            trunc_count += 1
            obs, info = env.reset()

    dt = time.perf_counter() - t0
    fps = n_steps / dt if dt > 0 else float("inf")
    print(f"steps: {n_steps}, elapsed: {dt:.3f}s, ~FPS: {fps:.1f}")
    print(f"terminated: {term_count} times, truncated: {trunc_count} times")
    print(f"reward stats over rollout: mean={np.mean(rewards):.3f}, std={np.std(rewards):.3f}, min={np.min(rewards):.3f}, max={np.max(rewards):.3f}")

    # 再次检测一下观测（若中途形状有变更）
    obs_info2 = guess_pixel_obs(np.array(obs) if not isinstance(obs, np.ndarray) else obs)
    if obs_info2["shape"] != obs_info["shape"] or obs_info2["dtype"] != obs_info["dtype"]:
        print("\n[Note] Observation shape/dtype changed during rollout.")
        print("obs sample info (end):", obs_info2)

    env.close()
    print("\nValidation finished ✔")


if __name__ == "__main__":
    main()

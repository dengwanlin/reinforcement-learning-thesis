#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stable-Baselines3 runner (non-interactive, CLI-only version)
------------------------------------------------------------

- 支持命令行参数： --env, --algo
- 不再有任何 input() 或交互提示
- 所有参数从 rl_config.py 加载
"""

from __future__ import annotations
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import gymnasium as gym
from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
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

# 读取配置
from rl_config import CONFIGS, ENVS


# --------------------------------------------------------------------------- #
# Path Manager
# --------------------------------------------------------------------------- #
class PathManager:
    PREFERRED_ROOT = Path("/homes/sohawan2/reinforcement-learning-thesis/code_practice")

    @staticmethod
    def is_writable(path: Path) -> bool:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".tmp"
            probe.write_text("ok")
            probe.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    def __init__(self):
        self.project_root = self._resolve_root()

    def _resolve_root(self) -> Path:
        env_root = os.getenv("RL_PROJECT_ROOT")
        if env_root and self.is_writable(Path(env_root)):
            return Path(env_root).expanduser().resolve()
        pref = self.PREFERRED_ROOT.expanduser().resolve()
        if self.is_writable(pref):
            return pref
        fallback = Path(os.path.dirname(os.path.abspath(__file__))) / "rl_runs_root"
        fallback.mkdir(exist_ok=True)
        return fallback

    def build_paths(self, env_id: str, algo: str) -> dict[str, Path]:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        env_tag = env_id.replace("-", "_")
        base = self.project_root / "runs" / env_tag / algo.upper() / ts
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
        for k in ("tb", "best", "ckpt", "models"):
            paths[k].mkdir(parents=True, exist_ok=True)
        return paths


# --------------------------------------------------------------------------- #
# Env Factory
# --------------------------------------------------------------------------- #
class EnvFactory:
    def make_env(self, env_id: str, seed: int = 0):
        def _init():
            env = gym.make(env_id)
            env.reset(seed=seed)
            return env
        return _init

    @staticmethod
    def algo_options_for_space(action_space):
        if isinstance(action_space, gym.spaces.Discrete):
            return ["DQN", "PPO", "A2C"]
        elif isinstance(action_space, gym.spaces.Box):
            return ["PPO", "A2C", "SAC", "TD3"]
        else:
            return ["PPO", "A2C"]


# --------------------------------------------------------------------------- #
# Model Factory
# --------------------------------------------------------------------------- #
class ModelFactory:
    @staticmethod
    def build_model(algo: str, env, tb_log, seed: int, model_kwargs: dict):
        policy = "MlpPolicy"
        common = dict(tensorboard_log=str(tb_log), verbose=1, seed=seed, policy=policy)
        if algo == "DQN":
            return DQN(env=env, device="auto", **model_kwargs, **common)
        if algo == "PPO":
            return PPO(env=env, device="auto", **model_kwargs, **common)
        if algo == "A2C":
            return A2C(env=env, device="cpu", **model_kwargs, **common)
        if algo == "SAC":
            return SAC(env=env, device="auto", **model_kwargs, **common)
        if algo == "TD3":
            return TD3(env=env, device="auto", **model_kwargs, **common)
        raise ValueError(f"Unsupported algo: {algo}")


# --------------------------------------------------------------------------- #
# Callback Factory
# --------------------------------------------------------------------------- #
class CallbackFactory:
    @staticmethod
    def make_callbacks(eval_env, best_path, eval_freq, stop_threshold, ckpt_path, ckpt_freq, algo, env_id):
        stop_cb = None
        if stop_threshold != float("inf"):
            stop_cb = StopTrainingOnRewardThreshold(reward_threshold=stop_threshold, verbose=1)

        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=str(best_path),
            n_eval_episodes=5,
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            callback_after_eval=stop_cb,
            verbose=1,
        )
        ckpt_cb = CheckpointCallback(
            save_freq=ckpt_freq,
            save_path=str(ckpt_path),
            name_prefix=f"{algo.lower()}_{env_id.replace('-', '_')}",
        )
        return [eval_cb, ckpt_cb]


# --------------------------------------------------------------------------- #
# Runner (no interaction)
# --------------------------------------------------------------------------- #
class Runner:
    def __init__(self, env_id: str, algo: str):
        self.env_id = env_id
        self.algo = algo
        self.path_manager = PathManager()
        self.env_factory = EnvFactory()
        self.model_factory = ModelFactory()
        self.callback_factory = CallbackFactory()

        if env_id not in CONFIGS:
            raise ValueError(f"Unknown environment: {env_id}")
        if algo not in CONFIGS[env_id]:
            raise ValueError(f"Algorithm {algo} not configured for {env_id}")

        cfg = CONFIGS[env_id][algo]
        self.stop_threshold = CONFIGS[env_id]["stop_threshold"]
        self.total_timesteps = cfg["train"]["total_timesteps"]
        self.eval_freq = cfg["train"]["eval_freq"]
        self.ckpt_freq = cfg["train"]["ckpt_freq"]
        self.seed = 0
        self.num_envs = cfg["train"].get("n_envs", 1)
        self.use_vecnorm = cfg["train"].get("normalize", False)
        self.model_kwargs = cfg["model_kwargs"]
        self.paths = self.path_manager.build_paths(env_id, algo)

    def _build_envs(self):
        env_fns = [self.env_factory.make_env(self.env_id, seed=self.seed + i) for i in range(self.num_envs)]
        train_env = SubprocVecEnv(env_fns) if self.num_envs > 1 else DummyVecEnv(env_fns)
        train_env = VecMonitor(train_env, filename=str(self.paths["monitor_train"]))

        eval_env = DummyVecEnv([self.env_factory.make_env(self.env_id, seed=self.seed + 100)])
        eval_env = VecMonitor(eval_env, filename=str(self.paths["monitor_eval"]))

        if self.use_vecnorm:
            train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

        self.train_env = train_env
        self.eval_env = eval_env

    def _dump_config(self):
        run_config = {
            "env_id": self.env_id,
            "algo": self.algo,
            "seed": self.seed,
            "total_timesteps": self.total_timesteps,
            "eval_freq": self.eval_freq,
            "ckpt_freq": self.ckpt_freq,
            "num_envs": self.num_envs,
            "use_vecnorm": self.use_vecnorm,
            "stop_threshold": self.stop_threshold,
            "model_kwargs": self.model_kwargs,
            "paths": {k: str(v) for k, v in self.paths.items()},
            "time": datetime.now().isoformat(timespec="seconds"),
            "project_root": str(self.path_manager.project_root),
        }
        with open(self.paths["config"], "w", encoding="utf-8") as f:
            json.dump(run_config, f, indent=2, ensure_ascii=False, default=str)

    def run(self):
        print(f"\n=== Training {self.algo} on {self.env_id} ===")
        self._build_envs()
        model = self.model_factory.build_model(self.algo, self.train_env, self.paths["tb"], self.seed, self.model_kwargs)
        callbacks = self.callback_factory.make_callbacks(
            self.eval_env, self.paths["best"], self.eval_freq,
            self.stop_threshold, self.paths["ckpt"], self.ckpt_freq, self.algo, self.env_id
        )
        self._dump_config()

        model.learn(total_timesteps=self.total_timesteps, callback=callbacks, progress_bar=True)

        mean_r, std_r = evaluate_policy(model, self.eval_env, n_eval_episodes=10, deterministic=True)
        print(f"Final evaluation on {self.env_id}: mean={mean_r:.2f} ± {std_r:.2f}")

        model.save(str(self.paths["models"] / f"last_{self.algo.lower()}"))
        print(f"Saved final model to {self.paths['models']}")

        self.train_env.close()
        self.eval_env.close()
        print("=== Training Complete ===\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="SB3 Runner (non-interactive)")
    parser.add_argument("--env", required=True, help="Environment ID (e.g. CartPole-v1)")
    parser.add_argument("--algo", required=True, help="Algorithm (e.g. PPO, A2C, SAC, TD3, DQN)")
    args = parser.parse_args()

    Runner(env_id=args.env, algo=args.algo).run()


if __name__ == "__main__":
    main()

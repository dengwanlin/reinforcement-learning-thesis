#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interactive Stable‑Baselines3 runner (single‑file, refactored).

All helper functions are now grouped into small classes:
    Util, PathManager, EnvFactory, ModelFactory, CallbackFactory.
The public entry point is still `Runner().run()`.
"""
# --------------------------------------------------------------------------- #
#  Imports
# --------------------------------------------------------------------------- #
from __future__ import annotations
import os, sys, json, cv2, numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, List, Dict

import gymnasium as gym
import gym as oldgym                     # for space conversion (Gym → Gymnasium)

from stable_baselines3 import DQN, PPO, A2C, SAC, TD3
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

#  Configuration data
from rl_config import CONFIGS, ENVS, ALGOS   # ENVS / ALGOS / CONFIGS are defined in rl_config.py

#  1. Utility class –interaction、dispatch、NumPy helpers
class Util:
    """Collect all general utility functions that are not related to business processes."""
    # ---------- File writability ----------
    @staticmethod
    def is_writable(path: Path) -> bool:
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_ok.tmp"
            probe.write_text("ok")
            probe.unlink(missing_ok=True)
            return True
        except Exception:
            return False

    # ---------- Interaction ----------
    @staticmethod
    def ask_choice(title: str, options: List[str]) -> str:
        print(f"\n== Choose {title} ==")
        for i, opt in enumerate(options, 1):
            print(f"  [{i}] {opt}")
        while True:
            raw = input(f"Enter index 1~{len(options)}: ").strip()
            if raw.isdigit() and 1 <= int(raw) <= len(options):
                return options[int(raw) - 1]
            print("Invalid input, try again.")

    @staticmethod
    def ask_int(title: str, default: int) -> int:
        raw = input(f"{title} (default {default}): ").strip()
        if raw == "":
            return default
        try:
            return int(raw.replace("_", ""))
        except Exception:
            print("Invalid input, using default.")
            return default
    @staticmethod
    def ask_yes_no(title: str, default: bool = False) -> bool:
        raw = input(f"{title} (y/n, default {'y' if default else 'n'}): ").strip().lower()
        if raw == "":
            return default
        return raw in ["y", "yes", "1"]

    @staticmethod
    def ask_str(title: str, default: str | None = None) -> str | None:
        raw = input(f"{title}{'' if default is None else f' (default {default})'}: ").strip()
        return default if raw == "" else raw

    # ---------- Linear Scheduling ----------
    @staticmethod
    def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
        def func(progress_remaining: float) -> float:
            return progress_remaining * (initial_value - final_value) + final_value
        return func

    # ---------- NumPy → uint8 ----------
    
    @staticmethod
    def to_uint8(arr: np.ndarray) -> np.ndarray:
        if isinstance(arr, np.ndarray) and arr.dtype == np.uint8:
            return arr
        a = np.array(arr)

        if a.dtype.kind == "f":          # floating point
            a = np.clip(a, 0.0, 255.0)
            if a.max() <= 1.0:           # already normalized to [0,1]
                a = a * 255.0
            a = a.astype(np.uint8)
        else:                            # integer or other numeric types
            a = np.clip(a, 0, 255).astype(np.uint8)
        return a
        
#  2. PathManager – Project root directory & run directory creation
class PathManager:
    """Responsible for finding a writable project root directory and creating the subdirectory structure for each run."""

    PREFERRED_ROOT = Path("/homes/sohawan2/reinforcement-learning-thesis/code_practice")

    def __init__(self, util: Util):
        self.util = util
        self.project_root = self._resolve_project_root()

    # ---------- Project root directory ----------
    def _resolve_project_root(self) -> Path:
        env_root = os.getenv("RL_PROJECT_ROOT")
        if env_root:
            cand = Path(env_root).expanduser().resolve()
            if self.util.is_writable(cand):
                print(f"[Path] Using RL_PROJECT_ROOT: {cand}")
                return cand
            else:
                print(f"[Path] RL_PROJECT_ROOT not writable, ignoring: {cand}")

        pref = self.PREFERRED_ROOT.expanduser().resolve()
        if self.util.is_writable(pref):
            print(f"[Path] Using preferred root: {pref}")
            return pref

        fallback = (Path(os.path.dirname(os.path.abspath(__file__))) / "rl_runs_root").resolve()
        if self.util.is_writable(fallback):
            print(f"[Path] Fallback root: {fallback}")
            return fallback

        print("[Path] ERROR: No writable project root found.", file=sys.stderr)
        sys.exit(1)

    # ---------- Create a directory for an experiment ----------
    def build_paths(self, env_id: str, algo: str, run_name: str | None) -> dict[str, Path]:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        env_tag = env_id.replace("-", "_")
        tag = ts if not run_name else f"{ts}-{run_name}"
        base = self.project_root / "runs" / env_tag / algo.upper() / tag

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

# 3. Env Factory – Environment creation, Mario adaptation, pixel packing, spatial filtering
class EnvFactory:
    """All environment-related functions are placed here."""

    def __init__(self, util: Util):
        self.util = util
        self._setup_mario_flag()

    # ---------- Check if Mario is available ----------
    def _setup_mario_flag(self) -> None:
        try:
            import gym_super_mario_bros   # noqa: F401
            from nes_py.wrappers import JoypadSpace   # noqa: F401
            self._MARIO_AVAILABLE = True
        except Exception:
            self._MARIO_AVAILABLE = False

    # ---------- Determine whether it is a pixel environment ----------
    @staticmethod
    def is_image_env(env_id: str) -> bool:
        return env_id.startswith("SuperMarioBros")

    # ---------- Gym → Gymnasium adaptation ----------
    @staticmethod
    def _to_gymnasium_space(space):
        if isinstance(space, oldgym.spaces.Box):
            return gym.spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
        if isinstance(space, oldgym.spaces.Discrete):
            return gym.spaces.Discrete(space.n)
        if isinstance(space, oldgym.spaces.MultiBinary):
            return gym.spaces.MultiBinary(space.n)
        if isinstance(space, oldgym.spaces.MultiDiscrete):
            return gym.spaces.MultiDiscrete(space.nvec)
        return space

    class GymToGymnasiumEnv(gym.Env):
        """Wrap a classic‑Gym env so that Gymnasium wrappers accept it."""
        metadata = {"render_modes": []}

        def __init__(self, old_env):
            super().__init__()
            self._env = old_env
            self.observation_space = EnvFactory._to_gymnasium_space(
                getattr(old_env, "observation_space", None)
            )
            self.action_space = EnvFactory._to_gymnasium_space(
                getattr(old_env, "action_space", None)
            )
            self.metadata = getattr(old_env, "metadata", self.metadata)
            self.reward_range = getattr(old_env, "reward_range", (-float("inf"), float("inf")))

        def reset(self, *, seed=None, options=None):
            if seed is not None:
                try:
                    self._env.reset(seed=seed)
                except TypeError:
                    try:
                        self._env.seed(seed)
                    except Exception:
                        pass
            out = self._env.reset()
            if isinstance(out, tuple) and len(out) == 2:
                return out
            return out, {}

        def step(self, action):
            out = self._env.step(action)
            if len(out) == 5:  # already gymnasium‑style
                return out
            obs, reward, done, info = out
            terminated = bool(done)
            truncated = bool(info.get("TimeLimit.truncated", False))
            return obs, reward, terminated, truncated, info

        def render(self, *a, **k):
            return self._env.render(*a, **k)

        def close(self):
            return self._env.close()

        def __getattr__(self, name):
            return getattr(self._env, name)

    # ---------- pixel‑wrapper (independent of gymnasium) ----------
    class SimpleFrameSkip(gym.Wrapper):
        """Repeat the same action `skip` times, sum rewards, stop early if done."""
        def __init__(self, env, skip: int = 4):
            super().__init__(env)
            assert skip >= 1
            self._skip = skip

        def step(self, action):
            total_reward = 0.0
            terminated = truncated = False
            info = {}
            obs = None
            for _ in range(self._skip):
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break
            return obs, total_reward, terminated, truncated, info

    class GrayScaleCompat(gym.ObservationWrapper):
        """Convert RGB frames to grayscale. keep_dim=True → (H,W,1) uint8."""
        def __init__(self, env: gym.Env, keep_dim: bool = True):
            super().__init__(env)
            self.keep_dim = keep_dim
            old = self.observation_space
            assert isinstance(old, gym.spaces.Box) and len(old.shape) == 3
            h, w, _ = old.shape
            shape = (h, w, 1) if keep_dim else (h, w)
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

        def observation(self, obs):
            arr = EnvFactory._to_uint8(np.array(obs))
            if arr.ndim != 3 or arr.shape[2] == 1:
                gray = arr
            else:
                gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            if self.keep_dim:
                gray = gray[..., None]
            return EnvFactory._to_uint8(gray)

    class ResizeCompat(gym.ObservationWrapper):
        """Resize frames to target size (H,W), keep channels."""
        def __init__(self, env: gym.Env, size=(84, 84)):
            super().__init__(env)
            self.size = tuple(size)
            old = self.observation_space
            assert isinstance(old, gym.spaces.Box)
            s = old.shape
            if len(s) == 3:
                h, w, c = s
                shape = (self.size[0], self.size[1], c)
            else:
                h, w = s
                shape = (self.size[0], self.size[1])
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8)

        def observation(self, obs):
            arr = EnvFactory._to_uint8(np.array(obs))
            resized = cv2.resize(arr, (self.size[1], self.size[0]), interpolation=cv2.INTER_AREA)
            return EnvFactory._to_uint8(resized)

    class FrameStackCompat(gym.Wrapper):
        """Stack last k frames channel‑last: (H,W,k) if single‑channel."""
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
                h, w = s
                new_shape = (h, w, k)
            else:
                h, w, c = s
                new_shape = (h, w, c * k)
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            arr = EnvFactory._to_uint8(np.array(obs))
            self.frames.clear()
            for _ in range(self.k):
                self.frames.append(arr)
            return self._get_obs(), info

        def step(self, action):
            obs, r, t, tr, info = self.env.step(action)
            self.frames.append(EnvFactory._to_uint8(np.array(obs)))
            return self._get_obs(), r, t, tr, info

        def _get_obs(self):
            frames = list(self.frames)
            x = frames[0]
            if x.ndim == 2:
                stacked = np.stack(frames, axis=-1)
            else:
                stacked = np.concatenate(frames, axis=2)
            return EnvFactory._to_uint8(stacked)

    # ---------- Helper functions ----------
    @staticmethod
    #Internal uniform uint8 conversion (same as Util.to_uint8).
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

    # ---------- Environment Factory ----------
    #"Return a callable that creates a (possibly wrapped) environment.
    def make_env(self, env_id: str, seed: int = 0) -> Callable[[], gym.Env]:
        def _init():
            if self.is_image_env(env_id):
                if not self._MARIO_AVAILABLE:
                    raise RuntimeError(
                        "SuperMarioBros requested but gym-super-mario-bros/nes-py not installed. "
                        "Run: pip install gym-super-mario-bros nes-py opencv-python"
                    )
                # 1. raw env + JoypadSpace (old gym) → 2. bridge to gymnasium
                import gym_super_mario_bros
                from nes_py.wrappers import JoypadSpace
                from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

                env = gym_super_mario_bros.make(env_id, apply_api_compatibility=True, render_mode=None)
                env = JoypadSpace(env, SIMPLE_MOVEMENT)          # old‑gym wrapper
                env = self.GymToGymnasiumEnv(env)                # bridge → gymnasium.Env

                # 3. pixel preprocessing pipeline
                env = self.SimpleFrameSkip(env, skip=4)
                env = self.GrayScaleCompat(env, keep_dim=True)   # (H,W,1)
                env = self.ResizeCompat(env, size=(84, 84))      # (84,84,1)
                env = self.FrameStackCompat(env, k=4)            # (84,84,4)
            else:
                env = gym.make(env_id)

            env.reset(seed=seed)
            return env
        return _init
#    """Create a temporary vec‑env and return its action space (for use in the filtering algorithm)."""
    def probe_action_space(self, env_id: str, seed: int = 0):
        probe = DummyVecEnv([self.make_env(env_id, seed=seed)])
        try:
            return probe.action_space
        finally:
            probe.close()

    @staticmethod
    def algo_options_for_space(action_space) -> list[str]:#    """Returns a list of available algorithm names based on the action space."""
        if isinstance(action_space, gym.spaces.Discrete):
            return ["DQN", "PPO", "A2C"]
        if isinstance(action_space, gym.spaces.Box):
            return ["PPO", "A2C", "SAC", "TD3"]
        # Multi‑Discrete / Multi‑Binary → Keep the most common set
        return ["PPO", "A2C"]

# 4. ModelFactory – Create SB3 models based on algorithms/strategies
class ModelFactory:
    """Responsible for packaging ``algo``, ``policy``, ``model_kwargs``, etc. into Stable‑Baselines3 model objects."""

    @staticmethod
    def policy_for(env_id: str) -> str:
        """Returns the policy name (CnnPolicy/MlpPolicy) required by SB3."""
        return "CnnPolicy" if EnvFactory.is_image_env(env_id) else "MlpPolicy"

    @staticmethod
    def build_model(
        algo: str,
        env,
        tb_log: Path,
        seed: int,
        model_kwargs: dict,
        policy: str,
    ):
        """Instantiate the Stable‑Baselines3 model (unified ``common`` parameter)."""
        common = dict(tensorboard_log=str(tb_log), verbose=1, seed=seed, policy=policy)

        if algo == "DQN":
            return DQN(env=env, device="auto", **model_kwargs, **common)
        if algo == "PPO":
            return PPO(env=env, device="auto", **model_kwargs, **common)
        if algo == "A2C":
            device = "auto" if policy == "CnnPolicy" else "cpu"
            return A2C(env=env, device=device, **model_kwargs, **common)
        if algo == "SAC":
            return SAC(env=env, device="auto", **model_kwargs, **common)
        if algo == "TD3":
            return TD3(env=env, device="auto", **model_kwargs, **common)

        raise ValueError(f"Unsupported algo: {algo}")

#  5.  CallbackFactory – Eval / Checkpoint / StopTraining
class CallbackFactory:
    """Encapsulates the creation of all callbacks, maintaining exactly the same behavior as the original script."""

    @staticmethod
    def make_eval_callback(
        eval_env,
        best_path,
        eval_freq: int,
        n_eval_episodes: int,
        stop_threshold: float,
    ):
        stop_cb = None
        if stop_threshold != float("inf"):
            stop_cb = StopTrainingOnRewardThreshold(reward_threshold=stop_threshold, verbose=1)

        return EvalCallback(
            eval_env,
            best_model_save_path=str(best_path),
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            callback_after_eval=stop_cb,
            verbose=1,
        )

    @staticmethod
    def make_checkpoint_callback(ckpt_path, ckpt_freq: int, algo: str, env_id: str):
        name_prefix = f"{algo.lower()}_{env_id.replace('-', '_')}"
        return CheckpointCallback(
            save_freq=ckpt_freq,
            save_path=str(ckpt_path),
            name_prefix=name_prefix,
        )

# 6. Main process class – Runner
class Runner:
    """The entry point of the business process uses the above four tool classes to complete all work."""

    def __init__(self):
        # Instantiate four tool classes (create them only once and share them globally later)
        self.util = Util()
        self.path_manager = PathManager(self.util)
        self.env_factory = EnvFactory(self.util)
        self.model_factory = ModelFactory()
        self.callback_factory = CallbackFactory()

        # Directly expose the root directory to the outside (keep the original script's property name)
        self.project_root = self.path_manager.project_root

    # ------------------------------------------------------------------- #
    #Interaction phase (all using static methods in Util)
    # ------------------------------------------------------------------- #
    def interactive_setup(self):
        print("\n===== Stable‑Baselines3 Interactive Runner =====")
        print(f"[Path] Project root: {self.project_root}")

        # 1️⃣ Environment selection
        self.env_id = self.util.ask_choice("Environment", ENVS)

        # 2️⃣ Mario availability check (early error reporting)
        if self.env_factory.is_image_env(self.env_id) and not self.env_factory._MARIO_AVAILABLE:
            print("[Error] Super Mario Bros selected but gym-super-mario-bros/nes-py is not installed.")
            print("        Try: pip install gym-super-mario-bros nes-py opencv-python")
            sys.exit(1)

        # 3️⃣ Filtering algorithm based on action space
        action_space = self.env_factory.probe_action_space(self.env_id, seed=0)
        allowed_algos = self.env_factory.algo_options_for_space(action_space)
        print(f"[Info] {self.env_id} action space = {type(action_space).__name__} → allowed algos: {allowed_algos}")
        self.algo = self.util.ask_choice("Algorithm", allowed_algos)

        # 4️⃣ Read the default configuration
        self.env_cfg = CONFIGS[self.env_id]
        self.algo_cfg = self.env_cfg[self.algo]
        self.stop_threshold = self.env_cfg["stop_threshold"]

        # 5️⃣ Interactively override some hyperparameters
        self.total_timesteps = self.util.ask_int("Total timesteps", self.algo_cfg["train"]["total_timesteps"])
        self.seed = self.util.ask_int("Random seed", 0)
        self.eval_freq = self.util.ask_int("Eval frequency", self.algo_cfg["train"]["eval_freq"])
        self.ckpt_freq = self.util.ask_int("Checkpoint frequency", self.algo_cfg["train"]["ckpt_freq"])
        self.num_envs = self.util.ask_int("Number of parallel envs", 1)
        self.use_vecnorm = self.util.ask_yes_no("Enable VecNormalize (obs/reward normalization)?", False)
        self.run_name = self.util.ask_str("Optional run name suffix", None)

        # 6️⃣ Force VecNormalize off for pixel context
        if self.env_factory.is_image_env(self.env_id) and self.use_vecnorm:
            print("[Warn] VecNormalize is not recommended for pixel‑based envs; disabling it.")
            self.use_vecnorm = False

        # 7️⃣ Create all path objects
        self.paths = self.path_manager.build_paths(self.env_id, self.algo, self.run_name)

    # ------------------------------------------------------------------- #
    # Building VecEnv (training/evaluation)
    # ------------------------------------------------------------------- #
    def _build_vec_envs(self):
        env_fns = [self.env_factory.make_env(self.env_id, seed=self.seed + i) for i in range(self.num_envs)]
        train_env = SubprocVecEnv(env_fns) if self.num_envs > 1 else DummyVecEnv(env_fns)
        train_env = VecMonitor(train_env, filename=str(self.paths["monitor_train"]))

        eval_env = DummyVecEnv([self.env_factory.make_env(self.env_id, seed=self.seed + 100)])
        eval_env = VecMonitor(eval_env, filename=str(self.paths["monitor_eval"]))

        # 对像素环境做通道转置 (C, H, W) → SB3 的 CNN 需要这种格式
        if self.env_factory.is_image_env(self.env_id):
            if not isinstance(train_env, VecTransposeImage):
                train_env = VecTransposeImage(train_env)
            if not isinstance(eval_env, VecTransposeImage):
                eval_env = VecTransposeImage(eval_env)

        # Optional VecNormalize
        if self.use_vecnorm:
            train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

        self.train_env = train_env
        self.eval_env = eval_env

    # ------------------------------------------------------------------- #
    # build model
    # ------------------------------------------------------------------- #
    def _build_model(self):
        policy = self.model_factory.policy_for(self.env_id)
        self.model = self.model_factory.build_model(
            algo=self.algo,
            env=self.train_env,
            tb_log=self.paths["tb"],
            seed=self.seed,
            model_kwargs=self.algo_cfg["model_kwargs"],
            policy=policy,
        )

    # ------------------------------------------------------------------- #
    # build callbacks
    # ------------------------------------------------------------------- #
    def _build_callbacks(self):
        n_eval_eps = 3 if self.env_factory.is_image_env(self.env_id) else 5
        self.eval_cb = self.callback_factory.make_eval_callback(
            eval_env=self.eval_env,
            best_path=self.paths["best"],
            eval_freq=self.eval_freq,
            n_eval_episodes=n_eval_eps,
            stop_threshold=self.stop_threshold,
        )
        self.ckpt_cb = self.callback_factory.make_checkpoint_callback(
            ckpt_path=self.paths["ckpt"],
            ckpt_freq=self.ckpt_freq,
            algo=self.algo,
            env_id=self.env_id,
        )

    # save the run configuration to a JSON file
    def _dump_config(self):
        run_config = {
            "env_id": self.env_id,
            "algo": self.algo,
            "policy": self.model_factory.env_id,
            "seed": self.seed,
            "total_timesteps": self.total_timesteps,
            "eval_freq": self.eval_freq,
            "ckpt_freq": self.ckpt_freq,
            "num_envs": self.num_envs,
            "use_vecnorm": self.use_vecnorm,
            "stop_threshold": self.stop_threshold,
            "model_kwargs": self.algo_cfg["model_kwargs"],
            "paths": {k: str(v) for k, v in self.paths.items()},
            "time": datetime.now().isoformat(timespec="seconds"),
            "project_root": str(self.project_root),
            "mario_available": self.env_factory._MARIO_AVAILABLE,
        }
        with open(self.paths["config"], "w", encoding="utf-8") as f:
            json.dump(run_config, f, ensure_ascii=False, indent=2, default=str)

    # ------------------------------------------------------------------- #
    # main training process
    # ------------------------------------------------------------------- #
    def run(self):
        self.interactive_setup()
        self._build_vec_envs()
        self._build_model()
        self._build_callbacks()
        self._dump_config()
        print("\n===== Training Start =====")
        print(
            json.dumps(
                {
                    "env_id": self.env_id,
                    "algo": self.algo,
                    "policy": self.model_factory.policy_for(self.env_id),
                    "seed": self.seed,
                    "total_timesteps": self.total_timesteps,
                    "num_envs": self.num_envs,
                    "use_vecnorm": self.use_vecnorm,
                },
                indent=2,
            )
        )
        # -------------------- training -------------------- #
        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=[self.eval_cb, self.ckpt_cb],
            progress_bar=True,
        )

        # -------------------- VecNormalize save -------------------- #
        if self.use_vecnorm:
            self.train_env.save(str(self.paths["vecnormalize"]))
            print(f"VecNormalize stats saved to {self.paths['vecnormalize']}")

        # -------------------- final evaluation -------------------- #
        mean_r, std_r = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=3 if self.env_factory.is_image_env(self.env_id) else 10,
            deterministic=True,
        )
        print(
            f"\n[{self.algo}/{self.model_factory.policy_for(self.env_id)}] Final Eval on {self.env_id}: "
            f"mean={mean_r:.2f} ± {std_r:.2f}"
        )

        # -------------------- save final model -------------------- #
        last_path = self.paths["models"] / f"last_{self.algo.lower()}"
        self.model.save(str(last_path))
        print(f"Saved last model to: {last_path}.zip")

        # -------------------- cleanup -------------------- #
        self.train_env.close()
        self.eval_env.close()
        print("===== Training End =====\n")

#  Entry point
def main():
    Runner().run()

if __name__ == "__main__":
    main()
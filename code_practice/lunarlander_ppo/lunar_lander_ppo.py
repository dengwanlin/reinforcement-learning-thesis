#!/usr/bin/env python3
"""
LunarLander PPO training script (Stable-Baselines3)

Location: reinforcement-learning-thesis/code_practice/lunar_lander_ppo/lunar_lander_ppo.py
Output: All model files and log files (TensorBoard, monitor, checkpoint, best/final model)
        are saved under reinforcement-learning-thesis/code_practice/lunar_lander_ppo/

Dependencies (example):
    pip install -U "stable-baselines3[extra]" "gymnasium[box2d]"

Run (example):
    python reinforcement-learning-thesis/code_practice/lunar_lander_ppo/lunar_lander_ppo.py \
        --total-timesteps 1000000 --n-envs 8

TensorBoard:
    tensorboard --logdir reinforcement-learning-thesis/code_practice/lunar_lander_ppo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.vec_env import VecMonitor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO on Gymnasium LunarLander and save all outputs in this folder."
    )
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                        help="Total training timesteps.")
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel environments.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Common PPO hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--n-steps", type=int, default=2048, help="Number of steps per update.")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size.")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor.")
    parser.add_argument("--gae-lambda", type=float, default=0.95, dest="gae_lambda",
                        help="GAE lambda.")
    parser.add_argument("--clip-range", type=float, default=0.2, help="Policy clip range.")
    parser.add_argument("--ent-coef", type=float, default=0.0, dest="ent_coef",
                        help="Entropy bonus coefficient.")
    parser.add_argument("--vf-coef", type=float, default=0.5, dest="vf_coef",
                        help="Value function loss coefficient.")
    parser.add_argument("--n-epochs", type=int, default=10, dest="n_epochs",
                        help="Number of epochs per update.")

    # Callback / evaluation settings
    parser.add_argument("--save-freq", type=int, default=50_000,
                        help="Checkpoint save frequency (in env steps).")
    parser.add_argument("--eval-freq", type=int, default=10_000,
                        help="Evaluation frequency (in env steps).")
    parser.add_argument("--stop-reward", type=float, default=230.0,
                        help="Early stop when average eval reward >= this threshold.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Base directory: script's location
    base_dir = Path(__file__).resolve().parent
    base_dir.mkdir(parents=True, exist_ok=True)

    # Training environment (parallel). monitor_dir writes monitor*.csv
    env = make_vec_env(
        "LunarLander-v3",
        n_envs=args.n_envs,
        seed=args.seed,
        monitor_dir=str(base_dir),
    )
    env = VecMonitor(env)  # Records statistics visible in TensorBoard

    # Evaluation environment (single env)
    eval_env = make_vec_env(
        "LunarLander-v3",
        n_envs=1,
        seed=args.seed + 100,
        monitor_dir=str(base_dir),
    )
    eval_env = VecMonitor(eval_env)

    # Create PPO model, TensorBoard logs go to base_dir
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=str(base_dir),
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        n_epochs=args.n_epochs,
        seed=args.seed,
    )

    # Periodic checkpoint saving (ppo_ll_ckpt_*.zip)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.save_freq // max(args.n_envs, 1), 1),
        save_path=str(base_dir),
        name_prefix="ppo_ll_ckpt",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1,
    )

    # Early stop when reaching reward threshold (used with EvalCallback)
    stop_train_cb = StopTrainingOnRewardThreshold(
        reward_threshold=args.stop_reward,
        verbose=1,
    )

    # Periodic evaluation, saves best_model.zip and evaluation logs
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(base_dir),
        log_path=str(base_dir),
        eval_freq=max(args.eval_freq // max(args.n_envs, 1), 1),
        deterministic=True,
        render=False,
        callback_on_new_best=stop_train_cb,
        verbose=1,
    )

    # Start training
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    # Save final model (ppo_lunarlander_final.zip)
    model.save(str(base_dir / "ppo_lunarlander_final"))

    # Cleanup
    env.close()
    eval_env.close()

    print(f"Training complete. All models & logs are saved under: {base_dir}")


if __name__ == "__main__":
    main()

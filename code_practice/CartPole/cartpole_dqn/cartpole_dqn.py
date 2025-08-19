# dqn_cartpole.py
import os
import argparse
import random
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    # gymnasium 的 make 时会传 seed
    os.environ["PYTHONHASHSEED"] = str(seed)

def make_env(env_id: str, seed: int = 0):
    def _init():
        env = gym.make(env_id)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init

def build_model(env, lr=1e-3, gamma=0.99, buffer_size=100_000,
                learning_starts=10_000, batch_size=64, tau=1.0,
                train_freq=4, target_update_interval=1_000, exploration_fraction=0.1,
                exploration_final_eps=0.05, policy_kwargs=None, seed=0):
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr,
        gamma=gamma,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        train_freq=train_freq,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        policy_kwargs=policy_kwargs or dict(net_arch=[256, 256]),
        verbose=1,
        seed=seed,
    )
    return model

def main():
    parser = argparse.ArgumentParser(description="DQN on CartPole-v1 (Stable-Baselines3)")
    parser.add_argument("--env_id", type=str, default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--total_timesteps", type=int, default=200_000)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--reward_threshold", type=float, default=475.0)  # CartPole-v1 满分 500
    parser.add_argument("--logdir", type=str, default="./logs_dqn_cartpole")
    parser.add_argument("--modeldir", type=str, default="./models_dqn_cartpole")
    parser.add_argument("--no_plot", action="store_true", help="Skip plotting at the end")
    args = parser.parse_args()

    os.makedirs(args.logdir, exist_ok=True)
    best_dir = os.path.join(args.modeldir, "best")
    last_dir = os.path.join(args.modeldir, "last")
    os.makedirs(best_dir, exist_ok=True)
    os.makedirs(last_dir, exist_ok=True)

    set_seed(args.seed)

    # === 训练环境（矢量化 + 监控）===
    train_env = DummyVecEnv([make_env(args.env_id, seed=args.seed)])
    train_env = VecMonitor(train_env, filename=os.path.join(args.logdir, "train.monitor.csv"))

    # === 评估环境 ===
    eval_env = DummyVecEnv([make_env(args.env_id, seed=args.seed + 10)])
    eval_env = VecMonitor(eval_env, filename=os.path.join(args.logdir, "eval.monitor.csv"))

    # === 构建模型 ===
    model = build_model(train_env, seed=args.seed)

    # === 回调：达到阈值即提前停止 + 保存最优模型 ===
    stop_cb = StopTrainingOnRewardThreshold(reward_threshold=args.reward_threshold, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=best_dir,
        log_path=args.logdir,
        eval_freq=args.eval_freq,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        callback_after_eval=stop_cb,
        verbose=1,
    )

    # === 训练 ===
    model.learn(total_timesteps=args.total_timesteps, callback=eval_cb, progress_bar=True)
    model.save(os.path.join(last_dir, "dqn_cartpole_last"))
    train_env.close()
    eval_env.close()

    # === 加载最优模型并评估 ===
    print("\n=== Evaluating best model ===")
    best_path = os.path.join(best_dir, "best_model.zip")
    if os.path.exists(best_path):
        best_model = DQN.load(best_path)
    else:
        print("Best model not found, using last model.")
        best_model = DQN.load(os.path.join(last_dir, "dqn_cartpole_last"))

    test_env = gym.make(args.env_id)
    test_env.reset(seed=args.seed + 100)
    mean_r, std_r = evaluate_policy(best_model, test_env, n_eval_episodes=20, deterministic=True)
    print(f"Test Reward over 20 eps: mean={mean_r:.2f} ± {std_r:.2f}")
    test_env.close()

    # === 可选：绘制评估分数曲线 ===
    if not args.no_plot:
        try:
            import pandas as pd
            import matplotlib.pyplot as plt

            eval_csv = os.path.join(args.logdir, "evaluations.npz")
            if os.path.exists(eval_csv):
                # Stable-Baselines3 的 EvalCallback 会保存 evaluations.npz
                data = np.load(eval_csv)
                timesteps = data["timesteps"].squeeze()
                results = data["results"]  # shape: (num_evals, n_eval_episodes)
                means = results.mean(axis=1)
                plt.figure()
                plt.plot(timesteps, means, marker="o")
                plt.xlabel("Timesteps")
                plt.ylabel("Mean Eval Reward")
                plt.title("DQN on CartPole-v1")
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            else:
                print("evaluations.npz not found, skip plotting.")
        except Exception as e:
            print("Plot failed:", e)

if __name__ == "__main__":
    main()

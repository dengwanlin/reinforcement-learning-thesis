import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from datetime import datetime

# ========= Path Setup =========
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, "lunarlander_model")
log_dir = os.path.join(script_dir, "tensorboard_logs")
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

print(f"[INFO] Script directory: {script_dir}")
print(f"[INFO] Model will be saved to: {model_dir}")
print(f"[INFO] Tensorboard logs will be saved to: {log_dir}")

# ========= Reward Logger =========
episode_rewards = []

class RewardLogger(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.episode_reward = 0

    def reset(self, **kwargs):
        if hasattr(self, "episode_reward") and self.episode_reward != 0:
            episode_rewards.append(self.episode_reward)
        self.episode_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += reward
        return obs, reward, terminated, truncated, info

# ========= Create Environment =========
env = RewardLogger(gym.make("LunarLander-v3"))

# ========= Model Configuration =========
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=5e-4,
    buffer_size=100_000,
    learning_starts=10_000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.2,
    exploration_final_eps=0.05,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1,
    tensorboard_log=log_dir
)

# ========= Train Model =========
total_timesteps = 100_000
model.learn(total_timesteps=total_timesteps)

# ========= Save Model and Rewards =========
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = os.path.join(model_dir, f"dqn_lunarlander_v3_{timestamp}")
model.save(model_path)
print(f"\n✅ Model saved to: {model_path}.zip")

reward_path = os.path.join(model_dir, "episode_rewards.npy")
np.save(reward_path, episode_rewards)
print(f"✅ Saved {len(episode_rewards)} episode rewards to: {reward_path}")

# ========= Close Environment =========
env.close()

# ========= Plot Rewards =========
if len(episode_rewards) == 0:
    print("⚠️ No rewards recorded, skipping plot.")
else:
    def smooth(y, box_pts=10):
        box = np.ones(box_pts) / box_pts
        return np.convolve(y, box, mode='same')

    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label="Raw reward")
    plt.plot(smooth(episode_rewards), label="Smoothed reward", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN Training Rewards on LunarLander-v3")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    reward_plot_path = os.path.join(model_dir, "reward_plot.png")
    plt.savefig(reward_plot_path)
    plt.show()
    print(f"✅ Reward plot saved to: {reward_plot_path}")

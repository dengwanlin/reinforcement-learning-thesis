import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from datetime import datetime

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

# ========= Create & wrap env =========
env = RewardLogger(gym.make("LunarLander-v3"))

# ========= Model config =========
model = DQN(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    tensorboard_log="./tensorboard_logs/"
)

# ========= Train =========
total_timesteps = 100_000
model.learn(total_timesteps=total_timesteps)

# ========= Save model and rewards =========
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = "code_practice/conda_test/lunarlander_model"
os.makedirs(model_dir, exist_ok=True)


model_path = os.path.join(model_dir, f"dqn_lunarlander_v3_{timestamp}")
model.save(model_path)
print(f"\n Model saved to: {model_path}.zip")


reward_path = os.path.join(model_dir, "episode_rewards.npy")
np.save(reward_path, episode_rewards)
print(f" Saved {len(episode_rewards)} episode rewards to: {reward_path}")

# ========= Close env =========
env.close()

# ========= Plot rewards =========
if len(episode_rewards) == 0:
    print(" No rewards recorded, skipping plot.")
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
    print(f" Reward plot saved to: {reward_plot_path}")

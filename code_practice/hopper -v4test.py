import gymnasium as gym
import numpy as np

env = gym.make("Hopper-v4")
obs, info = env.reset(seed=0)
for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()
print("Hopper-v4 OK")

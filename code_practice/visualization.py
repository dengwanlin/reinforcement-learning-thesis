import gymnasium as gym
from stable_baselines3 import A2C
import imageio
import os

# load the trained model
model_path = "/homes/sohawan2/reinforcement-learning-thesis/code_practice/runs/CartPole_v1/A2C/20250919-161001/models/last_a2c.zip"
model = A2C.load(model_path)

# create environment
env = gym.make("CartPole-v1", render_mode="rgb_array")
frames = []
obs, info = env.reset(seed=42)
done = False
step_count = 0
max_steps = 500

# run the model and collect frames
while not done and step_count < max_steps:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    frame = env.render()
    frames.append(frame)
    step_count += 1
env.close()

# specify target directory and save GIF
target_directory = "/homes/sohawan2/reinforcement-learning-thesis/code_practice/runs/CartPole_v1/A2C/20250919-161001"
gif_filename = "cartpole_a2c_demo.gif"
output_path = os.path.join(target_directory, gif_filename)
os.makedirs(target_directory, exist_ok=True)  # ensure directory exists
imageio.mimsave(output_path, frames, fps=30, loop=0)
print(f"GIF successfully saved to: {output_path}")
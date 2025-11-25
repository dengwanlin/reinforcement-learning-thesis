from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

eval_path = Path("/homes/sohawan2/reinforcement-learning-thesis/thesis_project/runs_seed0/CartPole-v1/a2c/20251026_035110_983255_pid796198_seed0/eval/evaluations.npz")

data = np.load(eval_path, allow_pickle=True)

if "results" in data.files:
    results = data["results"]
    if results.ndim == 2:
        mean_returns = results.mean(axis=1)
    else:
        mean_returns = results
else:
    mean_returns = data["mean_returns"]

timesteps = data["timesteps"]

# smoothing
window = 5
kernel = np.ones(window) / window
smooth = np.convolve(mean_returns, kernel, mode="same")

print("n eval =", len(mean_returns))
print("last 5 raw:", mean_returns[-5:])
print("last 5 smooth:", smooth[-5:])

r_max = smooth.max()
r_end = smooth[-10:].mean() if len(smooth) >= 10 else smooth.mean()
print("r_max =", r_max, "r_end =", r_end, "delta_post =", r_end - r_max)

plt.figure(figsize=(8,4))
plt.plot(timesteps, mean_returns, label="raw")
plt.plot(timesteps, smooth, label="smoothed", linewidth=2)
plt.axhline(r_max, color="r", linestyle="--", label="R_max")
plt.axhline(r_end, color="g", linestyle=":", label="R_end")
plt.legend()
plt.tight_layout()
plt.show()

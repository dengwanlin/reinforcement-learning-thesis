import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # Success rate data
    envs = ["CartPole-v1", "Hopper-v4", "LunarLander-v3", "LunarLanderContinuous-v3"]
    a2c = [0.9333, 0.0334, 0.1250, 0.5995]
    ppo = [1.0,    0.2531, 0.7762, 0.7367]

    # Plot
    x = np.arange(len(envs))
    width = 0.35
    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, a2c, width, label="A2C")
    plt.bar(x + width/2, ppo, width, label="PPO")
    
    plt.title("Success Rate of A2C and PPO Across Environments", fontsize=14)
    plt.xticks(x, envs, rotation=15, ha="right")
    plt.ylabel("Success rate")
    plt.ylim(0.0, 1.1)
    plt.legend()
    plt.tight_layout()

    # Save to the directory where THIS script lives
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(script_dir, "success_rate.png")

    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[INFO] Saved success rate plot to: {out_path}")

if __name__ == "__main__":
    main()

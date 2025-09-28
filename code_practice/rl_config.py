from __future__ import annotations
from typing import Any, Dict

def linear_schedule(initial_value: float, final_value: float):
    """
    Return a callable that linearly interpolates from ``initial_value`` to
    ``final_value`` given ``progress_remaining`` (the fraction of training
    that is still left, ranging from 1 → 0).
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * (initial_value - final_value) + final_value
    return func

ENVS = [
    "CartPole-v1",
    "LunarLander-v2",
    "LunarLanderContinuous-v2",
    "Taxi-v3",
    "SuperMarioBros-1-1-v3",
    "Humanoid-v4",
]

ALGOS = ["DQN", "PPO", "A2C", "SAC", "TD3"]   # full list; filtered later


CONFIGS: Dict[str, Any] = {

    "CartPole-v1": {
        "stop_threshold": float("inf"),
        "DQN": {
            "model_kwargs": dict(
                learning_rate=1e-3,
                buffer_size=50_000,
                learning_starts=1_000,
                batch_size=64,
                gamma=0.99,
                train_freq=(4, "step"),
                gradient_steps=1,
                target_update_interval=500,
                exploration_fraction=0.20,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.02,
                policy_kwargs=dict(net_arch=[256, 256]),
            ),
            "train": dict(total_timesteps=300_000, eval_freq=10_000, ckpt_freq=50_000),
        },
        "PPO": {
            "model_kwargs": dict(
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                learning_rate=3e-4,
                clip_range=0.2,
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=[64, 64]),
            ),
            "train": dict(total_timesteps=300_000, eval_freq=5_000, ckpt_freq=10_000),
        },
        "A2C": {
            "model_kwargs": dict(
                n_steps=64,
                gamma=0.99,
                learning_rate=9e-5,
                ent_coef=0.015,
                vf_coef=0.5,
                max_grad_norm=0.2,
                policy_kwargs=dict(net_arch=[64, 64]),
            ),
            "train": dict(total_timesteps=300_000, eval_freq=5_000, ckpt_freq=50_000),
        },
    },
    "LunarLander-v2": {
        "stop_threshold": float("inf"),
        "DQN": {
            "model_kwargs": dict(
                learning_rate=5e-4,
                buffer_size=100_000,
                learning_starts=10_000,
                batch_size=64,
                gamma=0.99,
                train_freq=(4, "step"),
                gradient_steps=1,
                target_update_interval=1_000,
                exploration_fraction=0.30,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                policy_kwargs=dict(net_arch=[256, 256]),
            ),
            "train": dict(total_timesteps=1_000_000, eval_freq=20_000, ckpt_freq=100_000),
        },
        "PPO": {
            "model_kwargs": dict(
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.92,
                learning_rate=1.2e-4,
                clip_range=0.1,
                ent_coef=0.015,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=[512, 512]),
            ),
            "train": dict(total_timesteps=2_000_000, eval_freq=20_000, ckpt_freq=100_000),
        },
        "A2C": {
            "model_kwargs": dict(
                n_steps=8,
                gamma=0.99,
                gae_lambda=0.95,
                learning_rate=linear_schedule(1.5e-4, 3e-5),
                ent_coef=0.05,
                vf_coef=0.5,
                max_grad_norm=0.5,
                use_rms_prop=True,
                policy_kwargs=dict(net_arch=[128, 128]),
            ),
            "train": dict(total_timesteps=2_000_000, eval_freq=50_000, ckpt_freq=250_000,normalize=True),
        },
    },
    "LunarLanderContinuous-v2": {
        "stop_threshold": float("inf"),
        "PPO": {
            "model_kwargs": dict(
                n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                learning_rate=3e-4, clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=[256, 256]),
            ),
            "train": dict(total_timesteps=1_000_000, eval_freq=20_000, ckpt_freq=100_000),
        },
        "A2C": {
            "model_kwargs": dict(
                n_steps=5, gamma=0.99, learning_rate=7e-4, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=[256, 256]),
            ),
            "train": dict(total_timesteps=800_000, eval_freq=20_000, ckpt_freq=100_000),
        },
        "SAC": {
            "model_kwargs": dict(
                learning_rate=3e-4, buffer_size=1_000_000, batch_size=256, tau=0.02, gamma=0.99,
                train_freq=(1, "step"), gradient_steps=1, ent_coef="auto",
                policy_kwargs=dict(net_arch=[256, 256]),
            ),
            "train": dict(total_timesteps=1_000_000, eval_freq=20_000, ckpt_freq=100_000),
        },
        "TD3": {
            "model_kwargs": dict(
                learning_rate=3e-4, buffer_size=1_000_000, batch_size=256, tau=0.02, gamma=0.99,
                train_freq=(1, "step"), gradient_steps=1,
                policy_kwargs=dict(net_arch=[400, 300]),
            ),
            "train": dict(total_timesteps=1_000_000, eval_freq=20_000, ckpt_freq=100_000),
        },
    },
    "Taxi-v3": {
        "stop_threshold": float("inf"),  # Perfect-policy average reward ~ 8
        "DQN": {
            "model_kwargs": dict(
                learning_rate=1e-3,
                buffer_size=50_000,
                learning_starts=1_000,
                batch_size=64,
                gamma=0.99,
                train_freq=(4, "step"),
                gradient_steps=1,
                target_update_interval=500,
                exploration_fraction=0.2,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05,
                policy_kwargs=dict(net_arch=[64, 64]),
            ),
            "train": dict(total_timesteps=50_000, eval_freq=5_000, ckpt_freq=10_000),
        },
        "PPO": {
            "model_kwargs": dict(
                n_steps=256, batch_size=64, n_epochs=4, gamma=0.99, gae_lambda=0.95,
                learning_rate=3e-4, clip_range=0.2, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=[64, 64]),
            ),
            "train": dict(total_timesteps=50_000, eval_freq=5_000, ckpt_freq=10_000),
        },
        "A2C": {
            "model_kwargs": dict(
                n_steps=5, gamma=0.99, learning_rate=7e-4, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=[64, 64]),
            ),
            "train": dict(total_timesteps=50_000, eval_freq=5_000, ckpt_freq=10_000),
        },
    },
    "SuperMarioBros-1-1-v3": {
        "stop_threshold": float("inf"),
        "DQN": {
            "model_kwargs": dict(
                learning_rate=1e-4,
                buffer_size=100_000,
                learning_starts=20_000,
                batch_size=32,
                gamma=0.99,
                train_freq=(4, "step"),
                gradient_steps=1,
                target_update_interval=10_000,
                exploration_fraction=0.10,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.01,
                policy_kwargs=dict(),  # default Nature-CNN
            ),
            "train": dict(total_timesteps=3_000_000, eval_freq=200_000, ckpt_freq=200_000),
        },
        "PPO": {
            "model_kwargs": dict(
                n_steps=2048,
                batch_size=258,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                learning_rate=2.5e-4,
                clip_range=0.1,
                ent_coef=0.02,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=dict(),    # default Nature-CNN
            ),
            "train": dict(total_timesteps=2_000_000, eval_freq=100_000, ckpt_freq=100_000),
        },
        "A2C": {
            "model_kwargs": dict(
                n_steps=16,
                gamma=0.99,
                learning_rate=3e-4,
                ent_coef=0.1,
                vf_coef=0.5,
                max_grad_norm=0.3,
                policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]  # use MLP here
        ),
            ),
            "train": dict(total_timesteps=2_000_000, eval_freq=100_000, ckpt_freq=100_000),
        },
    },
    "Humanoid-v4": {
            "stop_threshold": float("inf"),  # no early-stop; training is long
            "PPO": {
            "model_kwargs": dict(
                n_steps=8192,            # long rollouts help stability
                batch_size=1024,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                learning_rate=3e-4,
                clip_range=0.2,
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=[256, 256], ortho_init=False),
            ),
            "train": dict(total_timesteps=10_000_000, eval_freq=100_000, ckpt_freq=500_000),
        },
        "A2C": {
            "model_kwargs": dict(
                n_steps=20,              # A2C is lightweight; expect weaker performance
                gamma=0.99,
                learning_rate=7e-4,
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=[256, 256]),
            ),
            "train": dict(total_timesteps=5_000_000, eval_freq=100_000, ckpt_freq=500_000),
        },
        "SAC": {
            "model_kwargs": dict(
                learning_rate=3e-4,
                buffer_size=2_000_000,
                batch_size=512,
                tau=0.01,
                gamma=0.99,
                train_freq=(1, "step"),
                gradient_steps=1,
                ent_coef="auto",
                policy_kwargs=dict(net_arch=[256, 256]),
            ),
            "train": dict(total_timesteps=10_000_000, eval_freq=100_000, ckpt_freq=500_000),
        },
        "TD3": {
            "model_kwargs": dict(
                learning_rate=3e-4,
                buffer_size=2_000_000,
                batch_size=512,
                tau=0.01,
                gamma=0.99,
                train_freq=(1, "step"),
                gradient_steps=1,
                policy_kwargs=dict(net_arch=[400, 300]),
            ),
            "train": dict(total_timesteps=10_000_000, eval_freq=100_000, ckpt_freq=500_000),
        },
    },
}
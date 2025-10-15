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
    "Hopper-v4",
]

ALGOS = ["PPO", "A2C"]   # full list; filtered later


CONFIGS: Dict[str, Any] = {

    "CartPole-v1": {
        "stop_threshold": float("inf"),
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
        "PPO": {      #one hyperparameter resource: https://github.com/alperenunlu/ppo-lunarlander-v2
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
        "A2C": {    #https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/a2c.yml
            "model_kwargs": dict(
                n_steps=64,
                gamma=0.995,
                gae_lambda=1.0,
                #learning_rate=linear_schedule(8.3e-4, 1e-4),
                learning_rate=7e-4,
                ent_coef=0.0001,
                vf_coef=0.5,
                max_grad_norm=0.5,
                use_rms_prop=True,
                policy_kwargs=dict(net_arch=[64, 64]),
            ),
            "train": dict(total_timesteps=3_000_000, eval_freq=20_000, ckpt_freq=250_000,normalize=True),
        },
    },

    "LunarLanderContinuous-v2": {
        "stop_threshold": float("inf"),
        "PPO": {  # source:https://assets-eu.researchsquare.com/files/rs-5939959/v2_covered_b0beacb0-534e-4c72-8a6c-7954fdc6a3ed.pdf?c=1744730019
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
                #target_kl=0.02,                        # early stopping
                #clip_range_vf=0.2,                    # clip value function too (not in original PPO)
                policy_kwargs=dict(net_arch=[256, 256]),
            ),
            "train": dict(total_timesteps=1_000_000, eval_freq=20_000, ckpt_freq=100_000),
        },
        "A2C": {  # source:https://huggingface.co/qgallouedec/a2c-LunarLanderContinuous-v2-3898385124?utm_source=chatgpt.com
            "model_kwargs": dict(
                n_steps=32, 
                gamma=0.99,
                gae_lambda=0.95,
                learning_rate=7e-4,
                ent_coef=0.002, 
                vf_coef=0.5, 
                max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=[64, 64], ortho_init=True),
            ),
            "train": dict(total_timesteps=5_000_000, eval_freq=20_000, ckpt_freq=100_000),
        },
    },
    "Hopper-v4": {
        "stop_threshold": float("inf"),

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
                policy_kwargs=dict(net_arch=[256, 256]),   # MLP policy
            ),
            "train": dict(
                total_timesteps=3_000_000,
                eval_freq=100_000,
                ckpt_freq=200_000,
                normalize=True,   # use VecNormalize
            ),
        },

        "A2C": {
            "model_kwargs": dict(
                n_steps=2048,
                gamma=0.99,
                gae_lambda=0.95,
                learning_rate=7e-4,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                use_rms_prop=True,
                policy_kwargs=dict(net_arch=[128, 128]),   # smaller network
            ),
            "train": dict(
                total_timesteps=5_000_000,
                eval_freq=100_000,
                ckpt_freq=200_000,
                normalize=True,
            ),
        },
    },

}
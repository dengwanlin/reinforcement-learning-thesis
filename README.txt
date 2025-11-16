⭐️ Reinforcement Learning Hyperparameter Analysis Framework
<div align="center"> <i>A systematic framework for studying how hyperparameters influence RL learning dynamics</i> </div>
<br> <h2>📌 Overview</h2> <p style="font-size:16px; line-height:1.7;"> This project provides a flexible and automated framework to analyze the influence of <b>hyperparameters</b> on the training dynamics of popular reinforcement learning algorithms across multiple environments. It includes a custom Stable-Baselines3 runner, a full hyperparameter queue scheduler, and tools designed to help researchers evaluate convergence behavior, performance stability, and hyperparameter interactions. </p>
<h2>⚙️ Supported Algorithms & Environments</h2>
Algorithms (Stable-Baselines3)

A2C

PPO

Environments

CartPole-v1

LunarLander-v3

LunarLanderContinuous-v3

Hopper-v4

<h2>📂 Main Components</h2>
runner.py — Custom SB3 Experiment Runner
<p style="font-size:15px; line-height:1.7;"> A unified interface for launching reproducible RL experiments. Supports CLI configuration, automatic model saving, evaluation, and consistent logging across environments. </p>

Example:

python runner.py --env CartPole-v1 --algo PPO --learning-rate 0.0003 --gamma 0.99

queue-runner.py — Full Hyperparameter Search Scheduler
<p style="font-size:15px; line-height:1.7;"> Automatically expands all combinations of hyperparameter settings and runs them in sequence or in parallel. Ensures full coverage of the search space. </p>

Useful for:

Grid search

Sensitivity analysis

Interaction studies

<h2>📊 Research Questions & Analysis Goals</h2>

Below are the key analysis topics supported by the framework:

1. Which hyperparameters achieve the highest reward?

Identify optimal regions of the hyperparameter space.

2. How many episodes are required to reach a given reward?

Assess sample efficiency.

3. Does the reward curve converge or degrade?

Evaluate long-term stability and potential overfitting.

4. Variance across multiple training runs

How sensitive is the algorithm to randomness?

5. Variance between episodes within a single run

Is training smooth or oscillatory?

6. Variance across episodes of the final learned policy

Does the final policy behave consistently?

7. Individual effects of single hyperparameters

One-variable sensitivity analysis (learning rate, gamma, batch size, etc.)

8. How to visualize single hyperparameter effects

Reward curves, heatmaps, convergence plots, variance trends.

9. Interdependencies between hyperparameters

Pairwise and multi-dimensional interactions such as:

LR × gamma

entropy × batch size

n_steps × clip range

<h2>🚀 Example Workflow</h2>

Define hyperparameter ranges

Generate combinations via queue-runner.py

Train and evaluate using runner.py

Collect all reward curves, variance statistics

Run analysis & visualization scripts

Draw conclusions on best hyperparameters and stability patterns

<h2>📈 Possible Extensions (Optional)</h2>

Automatic visualization dashboard

Bayesian hyperparameter optimization

Multi-environment benchmarking

Add SB3 algorithms: SAC, TD3, DQN, DDPG

<h2>📜 License</h2> MIT License (or your chosen license)

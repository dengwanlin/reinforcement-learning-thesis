 <b>Reinforcement Learning Hyperparameter Analysis Framework </b>
<div align="left"><i>A framework designed to study how hyperparameters shape RL learning dynamics, convergence behavior, and stability</i></div> <br> <h2> Overview</h2> <p style="font-size:16px; line-height:1.7;"> This project provides a systematic, experiment-driven framework for analyzing how <b>hyperparameters</b> influence key <b>reinforcement learning (RL) training metrics</b> across multiple environments. It includes: <ul> <li>A custom Stable-Baselines3 experiment runner</li> <li>A hyperparameter queue scheduler for exhaustive grid search</li> <li>Tools to evaluate reward dynamics, convergence, and stability</li> <li>Mechanisms to study both single-parameter effects and multi-parameter interactions</li> </ul> </p>
<h2> Supported Algorithms & Environments</h2>

Algorithms (Stable-Baselines3):
A2C, PPO

Environments:
CartPole-v1, LunarLander-v3, LunarLanderContinuous-v3, Hopper-v4

<h2> Main Components</h2>
runner.py — Custom SB3 Experiment Runner
<p style="font-size:15px; line-height:1.7;"> Provides a unified interface to launch reproducible RL experiments. Supports CLI configuration, automatic logging, evaluation, and consistent training procedures across algorithms and environments. </p>

Example:

python runner.py --env CartPole-v1 --algo PPO --learning-rate 3e-4 --gamma 0.99

python queue_runner.py run --db queue.db --workdir . --workers 8 --timeout 7200
<p style="font-size:15px; line-height:1.7;"> Automatically enumerates all combinations of hyperparameters and schedules them for execution. Ensures full coverage of the search space for: </p>

Grid search

Sensitivity analysis

Interaction studies

<h2> Metrics Analyzed in This Framework</h2>

This project focuses on evaluating the following RL learning metrics:

1. Maximum Evaluation Reward

Identify which hyperparameter settings lead to the highest achievable performance.

2. Number of Episodes Needed to Reach a Target Reward

Measure training efficiency and sample complexity.

3. Convergence Behavior

Does the reward curve:

converge smoothly?

oscillate?

or collapse after rising?

4. Variance Between Different Runs (Different Seeds)

Assess robustness and sensitivity to randomness.

5. Variance Between Episodes Within a Single Training Run

Evaluate training stability (does reward change rapidly between episodes?).

6. Variance Between Episodes of the Final Policy

Analyze whether the final learned policy behaves consistently across trials.

<h2> Relationships Studied</h2>
1. Influence of Individual Hyperparameters on Each Metric

Examples:

Learning rate → convergence and variance

Gamma → long-term vs short-term reward trade-off

Entropy coefficient → exploration stability

2. Interdependencies Between Hyperparameters

How two or more hyperparameters jointly affect:

reward

stability

convergence

sensitivity

Examples of pairs studied:

Learning rate × gamma

Batch size × entropy coefficient

PPO clip range × n_steps

<h2> Research Questions (Direct Transformation of Your Analysis Goals)</h2>

Which hyperparameters can lead to the maximum reward?

How many episodes are needed to reach a given reward threshold?

Does the reward curve converge to a maximum, or does it degrade?

How can we evaluate the variance between different runs (run-to-run stability)?

How do we evaluate variance between episodes inside a run (training stability)?

How stable is the final policy (episode-to-episode variance)?

What is the influence of a single hyperparameter on each metric?

How can we visualize the effect of a single hyperparameter on all metrics?

How do we analyze the interdependencies between hyperparameters?

<h2> Example Workflow</h2>

Define hyperparameter ranges

Generate combinations using queue-runner.py

Run SB3 training using runner.py

Collect reward curves, variance data, convergence properties

Analyze hyperparameter effects using visualization tools

Summarize stability and performance profiles

Identify best hyperparameter regions

<h2> Optional Extensions</h2>

Automatic visualization dashboards

Bayesian hyperparameter optimization

Multi-environment performance comparison

Support for SAC, TD3, DQN, DDPG

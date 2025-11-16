# Topic
Analyzing the Impact of Hyperparameters on the Learning Dynamics of Popular Reinforcement Learning Algorithms Across Multiple Environments

#main code: runner.py
Custom Stable-Baselines3 runner, which supports the runninng of environments like cartpole -v1, LunarLander-v3, LunarLanderContinuous-v3, and Hopper-v4, with the algorithm of A2C and PPO. Through simple command-line interaction, users can quickly configure and launch reinforcement learning experiments.

# queue mechnism: queue-runner.py:
With a queue algorithm to make sure that all poosible Hyperparameters' combination can be arranged into the environments

# analyzing topic
1. Which hyperparameters can lead to the maximum reward

2. How many number of episodes are needed to reach certain reward

3. Does the reward curve converge to the maximum, or does the reward go down?

4. How to evaluate the variance between different runs

5. How to evaluate the variance between different episodes inside a run (does it go up/down rapitly)

6. How to evaluate the variance between different episodes of final policy (is my final result stable)

7. what is single influence of single hyperparameters on these metrics

8. How to show the influence of single hyperparameters on these metrics

9. how to find the Interdependencies of two or more hyperparameters


# We spoke about some metrics that you can measure. Here are some ideas, but keep in mind that these are not complete and you have to do your own research

Metrics:
Max (eval) reward
Num episodes needed to reach certain reward
Does it converge to the maximum, or does the reward go down?
Variance between different runs(with different seeds)
Variance between different episodes inside a run (does it go up/down rapitly)
Variance between different episodes of final policy (is my final result stable)

Relationships:
Influence of single hyperparameters on these metrics
Interdependencies of two (or more) hyperparameters

Question transformation and answer
0. the statistics of the environment
conda activate rl
cd /homes/sohawan2/reinforcement-learning-thesis/thesis_project/experiment_analysis
python statistics.py

1. Which hyperparameters can lead to the maximum reward

2. How many number of episodes are needed to reach certain reward

3. Does the reward curve converge to the maximum, or does the reward go down?

4. How to evaluate the variance between different runs

5. How to evaluate the variance between different episodes inside a run (does it go up/down rapitly)

6. How to evaluate the variance between different episodes of final policy (is my final result stable?)

7. what is single influence of single hyperparameters on these metrics

8. How to show the influence of single hyperparameters on these metrics

9. how to find the Interdependencies of two or more hyperparameters

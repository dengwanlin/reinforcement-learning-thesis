
# Experiment Analysis Pipeline


This document describes the analysis workflow used in the thesis experiments.
It explains where the data come from, how the analysis scripts are organised,
what files are produced, and how the pipeline is executed.

Project root assumed throughout this document:
/homes/sohawan2/reinforcement-learning-thesis/thesis_project



## Data Source: Training Run Directories


Each training job produces one self-contained run directory under:

runs/<ENV>/<ALGO>/<TIMESTAMP_pidXXXX_seedY>/

Example:

```

runs/Hopper-v4/a2c/20251107_171248_728087_pid2961473_seed0/

```

All post-experiment analysis operates by scanning these directories.

### 1.1 Files required for maximum-reward analysis

For each run, the following files are required:

#### config.yml

Contains run metadata and the parsed hyperparameters under the key "hyperparams_parsed".

#### eval/evaluations.npz

Generated automatically by SB3 EvalCallback during training.

### 1.2 Structure of eval/evaluations.npz

The NumPy archive contains three arrays:

timesteps
The training timestep at which each evaluation was triggered.

results
Episodic returns from the evaluation episodes.
Shape: (N_eval, N_episodes)

ep_lengths
Episode lengths corresponding to each evaluation episode.

Each row corresponds to one evaluation event.


## 2. Definition of the Maximum Achievable Reward (RQ1)

For each run, the maximum achievable reward is defined as:
```
R_max = max_t R_eval_t
```
where:

```
t indexes evaluation events (not training timesteps directly)

R_eval_t is the mean return of the evaluation episodes at evaluation t
```

Practical computation for a single run:

Load timesteps and results from eval/evaluations.npz

Compute the per-evaluation mean return:

mean_returns = results.mean(axis=1)

Compute:

R_max = max(mean_returns)
t_at_max = timesteps[argmax(mean_returns)]

This definition is used consistently in both Chapter 4 (metric definition)
and Chapter 5 (results).



## 3. Analysis Directory

All scripts related to maximum-reward analysis are located in:

experiment_analysis/max_reward/

Typical contents:

metric_rmax.py

extract_hyperparams.py

analyze_rmax_seed0.py

analyze_max_reward_configs.py

run_pipeline.py

All CSV outputs are written to this same directory.


## 4. Script Responsibilities

### 4.1 metric_rmax.py

Purpose:
Compute (R_max, t_at_max) for a single run.

Input:

eval/evaluations.npz

Output:

(R_max, t_at_max) returned in memory to the caller script

### 4.2 extract_hyperparams.py

Purpose:
Read config.yml and extract hyperparameters from
hyperparams_parsed.

Nested fields (e.g. policy_kwargs.net_arch,
policy_kwargs.ortho_init, policy_kwargs.log_std_init)
are flattened into scalar or string entries.

Input:

config.yml

Output:

Flat Python dictionary of hyperparameters (in memory)

### 4.3 analyze_rmax_seed0.py

Purpose:
Create a run-level master table for Research Question 1.

Main logic:

Recursively scan:
runs/<ENV>/<ALGO>/<RUN_ID>/

Load seed from config.yml

Keep only runs with seed == 0

Load eval/evaluations.npz

Compute (R_max, t_at_max)

Load and flatten hyperparameters from config.yml

Write one row per run

Outputs:

rmax_seed0_runs_full.csv
One row per run (seed=0), including:

env, algo, run_id, seed

R_max, t_at_max

all flattened hyperparameters

rmax_seed0_summary_by_env_algo.csv
Aggregated statistics per (env, algo):

number of runs

mean, max, min of R_max

### 4.4 analyze_max_reward_configs.py

Purpose:
Select best-performing hyperparameter configurations for RQ1.

Logic:

CartPole-v1 (saturated task):

Select all runs with R_max approximately equal to 500

Compute frequency statistics of hyperparameter values

Other environments (Hopper-v4, LunarLander-v3,
LunarLanderContinuous-v3):

For each (env, algo), find the global maximum R_max

Export all configurations achieving that maximum

Outputs:

cartpole_max500_hparam_freq.csv
Hyperparameter frequency table for CartPole-v1

other_envs_best_configs.csv
Full hyperparameter sets for best-performing configurations
in non-saturated environments

### 4.5 run_pipeline.py

Purpose:
End-to-end regeneration of all maximum-reward analysis outputs.

Steps executed:

Remove previously generated CSV files in the analysis directory

Run analyze_rmax_seed0.py

Run analyze_max_reward_configs.py

## 5. How to Run the Analysis

Recommended (full pipeline):
```
Change to analysis directory:
cd experiment_analysis/max_reward

Activate environment:
conda activate rl

Run pipeline:
python run_pipeline.py

After execution, the following files should exist:

rmax_seed0_runs_full.csv

rmax_seed0_summary_by_env_algo.csv

cartpole_max500_hparam_freq.csv

other_envs_best_configs.csv
```

## 6. Mapping to Thesis Sections

Chapter 4.3.1 (Maximum Achievable Reward):

Formal definition of R_max

Practical computation using evaluations.npz

Chapter 5.2 (Hyperparameters Leading to Maximum Reward):

CartPole-v1 frequency analysis uses cartpole_max500_hparam_freq.csv

Other environments use other_envs_best_configs.csv

## 7. Notes and Assumptions

Only seed = 0 runs are used for RQ1 extraction.

All metrics depend on EvalCallback outputs.

Runs missing eval/evaluations.npz are skipped.

Hyperparameters are read exclusively from config.yml
(hyperparams_parsed field).

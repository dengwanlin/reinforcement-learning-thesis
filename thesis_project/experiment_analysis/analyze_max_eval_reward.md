conda activate rl
(base) rtx3% conda activate rl
(rl) rtx3% cd /homes/sohawan2/reinforcement-learning-thesis/thesis_project
(rl) rtx3% python analyze_max_eval_reward.py
 Starting comprehensive reinforcement learning hyperparameter impact analysis (Simplified version)
Based on search results and best practices for tuning
================================================================================
🔍 Starting to scan the directory
 Environment: Hopper-v4
  🔧 Algorithm: a2c
     1224 experiments found
 Environment: LunarLanderContinuous-v3
  🔧 Algorithm: a2c
     3456 experiments found
  🔧 Algorithm: ppo
     3840 experiments found
 Environment: CartPole-v1
  🔧 Algorithm: a2c
     786 experiments found
  🔧 Algorithm: ppo
     1585 experiments found
 Environment: LunarLander-v3
  🔧 Algorithm: a2c
     1296 experiments found
  🔧 Algorithm: ppo
     1296 experiments found
 Successfully loaded data for 13483 experiments.

 Basic Statistical Analysis
----------------------------------------
Total experiments: 13483
Environments: ['Hopper-v4', 'LunarLanderContinuous-v3', 'CartPole-v1', 'LunarLander-v3']
Algorithms: ['a2c', 'ppo']
Overall average reward: 210.80 ± 248.20
Reward range: -3243.59 - 2784.75

🔍 Hyperparameter Impact Analysis on Maximum Reward
==================================================

 Environment: Hopper-v4 | Algorithm: a2c
Maximum reward: 2784.75
Number of configurations achieving maximum reward: 1
 Hyperparameter Commonalities in Optimal Configurations:
   learning_rate:
     Optimal value: 0.0007
     Frequency in optimal configs: 100.0% (overall: 62.7%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   n_steps:
     Optimal value: 128
     Frequency in optimal configs: 100.0% (overall: 31.4%)
     Diversity ratio: 0.25
      Strong correlation: Most optimal configurations use this value
   gamma:
     Optimal value: 0.99
     Frequency in optimal configs: 100.0% (overall: 52.9%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   gae_lambda:
     Optimal value: 0.95
     Frequency in optimal configs: 100.0% (overall: 34.0%)
     Diversity ratio: 0.33
      Strong correlation: Most optimal configurations use this value
   ent_coef:
     Optimal value: 0.0
     Frequency in optimal configs: 100.0% (overall: 50.3%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   vf_coef:
     Optimal value: 0.5
     Frequency in optimal configs: 100.0% (overall: 100.0%)
     Diversity ratio: 1.00
      Strong correlation: Most optimal configurations use this value
   max_grad_norm:
     Optimal value: 0.5
     Frequency in optimal configs: 100.0% (overall: 100.0%)
     Diversity ratio: 1.00
      Strong correlation: Most optimal configurations use this value
   use_rms_prop:
     Optimal value: True
     Frequency in optimal configs: 100.0% (overall: 50.3%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   rms_prop_eps:
     Optimal value: 0.0001
     Frequency in optimal configs: 100.0% (overall: 50.0%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   use_sde:
     Optimal value: True
     Frequency in optimal configs: 100.0% (overall: 100.0%)
     Diversity ratio: 1.00
      Strong correlation: Most optimal configurations use this value
   sde_sample_freq:
     Optimal value: 4.0
     Frequency in optimal configs: 100.0% (overall: 50.0%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   policy_kwargs:
     Optimal value: {'net_arch': [256, 256], 'ortho_init': True}
     Frequency in optimal configs: 100.0% (overall: 50.0%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value

 Hyperparameter Sensitivity Summary:
  🔥 learning_rate: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 n_steps: High sensitivity (frequency: 100.0%, diversity: 0.25)
  🔥 gamma: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 gae_lambda: High sensitivity (frequency: 100.0%, diversity: 0.33)
  🔥 ent_coef: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 vf_coef: High sensitivity (frequency: 100.0%, diversity: 1.00)
  🔥 max_grad_norm: High sensitivity (frequency: 100.0%, diversity: 1.00)
  🔥 use_rms_prop: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 rms_prop_eps: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 use_sde: High sensitivity (frequency: 100.0%, diversity: 1.00)
  🔥 sde_sample_freq: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 policy_kwargs: High sensitivity (frequency: 100.0%, diversity: 0.50)

 Configuration Diversity Analysis:
  Unique hyperparameter combinations: 1/1
  Diversity ratio: 1.00
   High diversity: Multiple different configurations achieve optimal performance

 Environment: LunarLanderContinuous-v3 | Algorithm: a2c
Maximum reward: 289.21
Number of configurations achieving maximum reward: 1
 Hyperparameter Commonalities in Optimal Configurations:
   learning_rate:
     Optimal value: 0.0003
     Frequency in optimal configs: 100.0% (overall: 33.3%)
     Diversity ratio: 0.33
      Strong correlation: Most optimal configurations use this value
   n_steps:
     Optimal value: 256
     Frequency in optimal configs: 100.0% (overall: 25.0%)
     Diversity ratio: 0.25
      Strong correlation: Most optimal configurations use this value
   gamma:
     Optimal value: 0.99
     Frequency in optimal configs: 100.0% (overall: 50.0%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   gae_lambda:
     Optimal value: 0.97
     Frequency in optimal configs: 100.0% (overall: 33.3%)
     Diversity ratio: 0.33
      Strong correlation: Most optimal configurations use this value
   ent_coef:
     Optimal value: 0.0
     Frequency in optimal configs: 100.0% (overall: 33.3%)
     Diversity ratio: 0.33
      Strong correlation: Most optimal configurations use this value
   vf_coef:
     Optimal value: 0.5
     Frequency in optimal configs: 100.0% (overall: 100.0%)
     Diversity ratio: 1.00
      Strong correlation: Most optimal configurations use this value
   max_grad_norm:
     Optimal value: 0.5
     Frequency in optimal configs: 100.0% (overall: 100.0%)
     Diversity ratio: 1.00
      Strong correlation: Most optimal configurations use this value
   use_rms_prop:
     Optimal value: True
     Frequency in optimal configs: 100.0% (overall: 50.0%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   rms_prop_eps:
     Optimal value: 0.0001
     Frequency in optimal configs: 100.0% (overall: 50.0%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   use_sde:
     Optimal value: True
     Frequency in optimal configs: 100.0% (overall: 100.0%)
     Diversity ratio: 1.00
      Strong correlation: Most optimal configurations use this value
   sde_sample_freq:
     Optimal value: 64.0
     Frequency in optimal configs: 100.0% (overall: 50.0%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   policy_kwargs:
     Optimal value: {'net_arch': [256, 256], 'ortho_init': True}
     Frequency in optimal configs: 100.0% (overall: 50.0%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value

 Hyperparameter Sensitivity Summary:
  🔥 learning_rate: High sensitivity (frequency: 100.0%, diversity: 0.33)
  🔥 n_steps: High sensitivity (frequency: 100.0%, diversity: 0.25)
  🔥 gamma: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 gae_lambda: High sensitivity (frequency: 100.0%, diversity: 0.33)
  🔥 ent_coef: High sensitivity (frequency: 100.0%, diversity: 0.33)
  🔥 vf_coef: High sensitivity (frequency: 100.0%, diversity: 1.00)
  🔥 max_grad_norm: High sensitivity (frequency: 100.0%, diversity: 1.00)
  🔥 use_rms_prop: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 rms_prop_eps: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 use_sde: High sensitivity (frequency: 100.0%, diversity: 1.00)
  🔥 sde_sample_freq: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 policy_kwargs: High sensitivity (frequency: 100.0%, diversity: 0.50)

 Configuration Diversity Analysis:
  Unique hyperparameter combinations: 1/1
  Diversity ratio: 1.00
   High diversity: Multiple different configurations achieve optimal performance

 Environment: LunarLanderContinuous-v3 | Algorithm: ppo
Maximum reward: 297.00
Number of configurations achieving maximum reward: 1
 Hyperparameter Commonalities in Optimal Configurations:
   learning_rate:
     Optimal value: 0.0001
     Frequency in optimal configs: 100.0% (overall: 50.0%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   n_steps:
     Optimal value: 1024
     Frequency in optimal configs: 100.0% (overall: 75.0%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   gamma:
     Optimal value: 0.995
     Frequency in optimal configs: 100.0% (overall: 50.0%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   gae_lambda:
     Optimal value: 0.97
     Frequency in optimal configs: 100.0% (overall: 50.0%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   ent_coef:
     Optimal value: 0.01
     Frequency in optimal configs: 100.0% (overall: 33.3%)
     Diversity ratio: 0.33
      Strong correlation: Most optimal configurations use this value
   vf_coef:
     Optimal value: 0.5
     Frequency in optimal configs: 100.0% (overall: 100.0%)
     Diversity ratio: 1.00
      Strong correlation: Most optimal configurations use this value
   max_grad_norm:
     Optimal value: 0.5
     Frequency in optimal configs: 100.0% (overall: 100.0%)
     Diversity ratio: 1.00
      Strong correlation: Most optimal configurations use this value
   use_sde:
     Optimal value: True
     Frequency in optimal configs: 100.0% (overall: 100.0%)
     Diversity ratio: 1.00
      Strong correlation: Most optimal configurations use this value
   sde_sample_freq:
     Optimal value: 64.0
     Frequency in optimal configs: 100.0% (overall: 50.0%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   policy_kwargs:
     Optimal value: {'net_arch': [128, 128], 'ortho_init': True, 'log_std_init': -0.5}
     Frequency in optimal configs: 100.0% (overall: 50.0%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   batch_size:
     Optimal value: 64.0
     Frequency in optimal configs: 100.0% (overall: 25.0%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   n_epochs:
     Optimal value: 10.0
     Frequency in optimal configs: 100.0% (overall: 75.0%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   clip_range:
     Optimal value: 0.2
     Frequency in optimal configs: 100.0% (overall: 40.0%)
     Diversity ratio: 0.33
      Strong correlation: Most optimal configurations use this value
   target_kl:
     Optimal value: 0.02
     Frequency in optimal configs: 100.0% (overall: 40.0%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value

 Hyperparameter Sensitivity Summary:
  🔥 learning_rate: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 n_steps: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 gamma: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 gae_lambda: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 ent_coef: High sensitivity (frequency: 100.0%, diversity: 0.33)
  🔥 vf_coef: High sensitivity (frequency: 100.0%, diversity: 1.00)
  🔥 max_grad_norm: High sensitivity (frequency: 100.0%, diversity: 1.00)
  🔥 use_sde: High sensitivity (frequency: 100.0%, diversity: 1.00)
  🔥 sde_sample_freq: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 policy_kwargs: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 batch_size: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 n_epochs: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 clip_range: High sensitivity (frequency: 100.0%, diversity: 0.33)
  🔥 target_kl: High sensitivity (frequency: 100.0%, diversity: 0.50)

 Configuration Diversity Analysis:
  Unique hyperparameter combinations: 1/1
  Diversity ratio: 1.00
   High diversity: Multiple different configurations achieve optimal performance

 Environment: CartPole-v1 | Algorithm: a2c
Maximum reward: 500.00
Number of configurations achieving maximum reward: 375
 Hyperparameter Commonalities in Optimal Configurations:
   learning_rate:
     Optimal value: 0.001
     Frequency in optimal configs: 40.5% (overall: 26.7%)
     Diversity ratio: 1.00
      Medium correlation: Clear preference exists
   n_steps:
     Optimal value: 64
     Frequency in optimal configs: 44.3% (overall: 34.9%)
     Diversity ratio: 1.00
      Medium correlation: Clear preference exists
   gamma:
     Optimal value: 0.99
     Frequency in optimal configs: 54.1% (overall: 51.1%)
     Diversity ratio: 1.00
      Medium correlation: Clear preference exists
   gae_lambda:
     Optimal value: 0.9
     Frequency in optimal configs: 50.9% (overall: 50.1%)
     Diversity ratio: 1.00
      Medium correlation: Clear preference exists
   ent_coef:
     Optimal value: 0.0
     Frequency in optimal configs: 52.0% (overall: 50.1%)
     Diversity ratio: 1.00
      Medium correlation: Clear preference exists
   vf_coef:
     Optimal value: 0.5
     Frequency in optimal configs: 100.0% (overall: 100.0%)
     Diversity ratio: 1.00
      Strong correlation: Most optimal configurations use this value
   max_grad_norm:
     Optimal value: 0.5
     Frequency in optimal configs: 100.0% (overall: 100.0%)
     Diversity ratio: 1.00
      Strong correlation: Most optimal configurations use this value
   use_rms_prop:
     Optimal value: False
     Frequency in optimal configs: 52.0% (overall: 50.0%)
     Diversity ratio: 1.00
      Medium correlation: Clear preference exists
   rms_prop_eps:
     Optimal value: 1e-05
     Frequency in optimal configs: 50.9% (overall: 50.0%)
     Diversity ratio: 1.00
      Medium correlation: Clear preference exists
   policy_kwargs:
     Optimal value: {'net_arch': [128, 128]}
     Frequency in optimal configs: 56.8% (overall: 51.0%)
     Diversity ratio: 1.00
      Medium correlation: Clear preference exists

 Hyperparameter Sensitivity Summary:
  🔥 vf_coef: High sensitivity (frequency: 100.0%, diversity: 1.00)
  🔥 max_grad_norm: High sensitivity (frequency: 100.0%, diversity: 1.00)
  ⭐ policy_kwargs: Medium sensitivity (frequency: 56.8%, diversity: 1.00)
  ⭐ gamma: Medium sensitivity (frequency: 54.1%, diversity: 1.00)
  ⭐ ent_coef: Medium sensitivity (frequency: 52.0%, diversity: 1.00)
  ⭐ use_rms_prop: Medium sensitivity (frequency: 52.0%, diversity: 1.00)
  ⭐ gae_lambda: Medium sensitivity (frequency: 50.9%, diversity: 1.00)
  ⭐ rms_prop_eps: Medium sensitivity (frequency: 50.9%, diversity: 1.00)
  ⭐ n_steps: Medium sensitivity (frequency: 44.3%, diversity: 1.00)
  ⭐ learning_rate: Medium sensitivity (frequency: 40.5%, diversity: 1.00)

 Configuration Diversity Analysis:
  Unique hyperparameter combinations: 364/375
  Diversity ratio: 0.97
   High diversity: Multiple different configurations achieve optimal performance

 Environment: CartPole-v1 | Algorithm: ppo
Maximum reward: 500.00
Number of configurations achieving maximum reward: 1511
 Hyperparameter Commonalities in Optimal Configurations:
   learning_rate:
     Optimal value: 0.0003
     Frequency in optimal configs: 34.0% (overall: 33.4%)
     Diversity ratio: 1.00
      Weak correlation: Some preference but high diversity
   n_steps:
     Optimal value: 256
     Frequency in optimal configs: 36.7% (overall: 36.3%)
     Diversity ratio: 1.00
      Weak correlation: Some preference but high diversity
   gamma:
     Optimal value: 0.99
     Frequency in optimal configs: 50.8% (overall: 50.0%)
     Diversity ratio: 1.00
      Medium correlation: Clear preference exists
   gae_lambda:
     Optimal value: 0.9
     Frequency in optimal configs: 50.5% (overall: 50.0%)
     Diversity ratio: 1.00
      Medium correlation: Clear preference exists
   ent_coef:
     Optimal value: 0.02
     Frequency in optimal configs: 33.8% (overall: 33.3%)
     Diversity ratio: 1.00
      Weak correlation: Some preference but high diversity
   vf_coef:
     Optimal value: 0.5
     Frequency in optimal configs: 100.0% (overall: 100.0%)
     Diversity ratio: 1.00
      Strong correlation: Most optimal configurations use this value
   max_grad_norm:
     Optimal value: 0.5
     Frequency in optimal configs: 100.0% (overall: 100.0%)
     Diversity ratio: 1.00
      Strong correlation: Most optimal configurations use this value
   policy_kwargs:
     Optimal value: {'net_arch': [128, 128]}
     Frequency in optimal configs: 50.1% (overall: 50.0%)
     Diversity ratio: 1.00
      Medium correlation: Clear preference exists
   batch_size:
     Optimal value: 64.0
     Frequency in optimal configs: 53.8% (overall: 54.5%)
     Diversity ratio: 1.00
      Medium correlation: Clear preference exists
   n_epochs:
     Optimal value: 10.0
     Frequency in optimal configs: 53.8% (overall: 54.6%)
     Diversity ratio: 1.00
      Medium correlation: Clear preference exists
   clip_range:
     Optimal value: 0.2
     Frequency in optimal configs: 50.8% (overall: 50.0%)
     Diversity ratio: 1.00
      Medium correlation: Clear preference exists
   target_kl:
     Optimal value: 0.02
     Frequency in optimal configs: 100.0% (overall: 100.0%)
     Diversity ratio: 1.00
      Strong correlation: Most optimal configurations use this value

 Hyperparameter Sensitivity Summary:
  🔥 vf_coef: High sensitivity (frequency: 100.0%, diversity: 1.00)
  🔥 max_grad_norm: High sensitivity (frequency: 100.0%, diversity: 1.00)
  🔥 target_kl: High sensitivity (frequency: 100.0%, diversity: 1.00)
  ⭐ batch_size: Medium sensitivity (frequency: 53.8%, diversity: 1.00)
  ⭐ n_epochs: Medium sensitivity (frequency: 53.8%, diversity: 1.00)
  ⭐ gamma: Medium sensitivity (frequency: 50.8%, diversity: 1.00)
  ⭐ clip_range: Medium sensitivity (frequency: 50.8%, diversity: 1.00)
  ⭐ gae_lambda: Medium sensitivity (frequency: 50.5%, diversity: 1.00)
  ⭐ policy_kwargs: Medium sensitivity (frequency: 50.1%, diversity: 1.00)
  ⚡ n_steps: Low sensitivity (frequency: 36.7%, diversity: 1.00)
  ⚡ learning_rate: Low sensitivity (frequency: 34.0%, diversity: 1.00)
  ⚡ ent_coef: Low sensitivity (frequency: 33.8%, diversity: 1.00)

 Configuration Diversity Analysis:
  Unique hyperparameter combinations: 1510/1511
  Diversity ratio: 1.00
   High diversity: Multiple different configurations achieve optimal performance

 Environment: LunarLander-v3 | Algorithm: a2c
Maximum reward: 255.44
Number of configurations achieving maximum reward: 1
 Hyperparameter Commonalities in Optimal Configurations:
   learning_rate:
     Optimal value: 0.0007
     Frequency in optimal configs: 100.0% (overall: 33.3%)
     Diversity ratio: 0.33
      Strong correlation: Most optimal configurations use this value
   n_steps:
     Optimal value: 256
     Frequency in optimal configs: 100.0% (overall: 33.3%)
     Diversity ratio: 0.33
      Strong correlation: Most optimal configurations use this value
   gamma:
     Optimal value: 0.995
     Frequency in optimal configs: 100.0% (overall: 50.0%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   gae_lambda:
     Optimal value: 0.97
     Frequency in optimal configs: 100.0% (overall: 33.3%)
     Diversity ratio: 0.33
      Strong correlation: Most optimal configurations use this value
   ent_coef:
     Optimal value: 0.01
     Frequency in optimal configs: 100.0% (overall: 33.3%)
     Diversity ratio: 0.33
      Strong correlation: Most optimal configurations use this value
   vf_coef:
     Optimal value: 0.5
     Frequency in optimal configs: 100.0% (overall: 100.0%)
     Diversity ratio: 1.00
      Strong correlation: Most optimal configurations use this value
   max_grad_norm:
     Optimal value: 0.5
     Frequency in optimal configs: 100.0% (overall: 100.0%)
     Diversity ratio: 1.00
      Strong correlation: Most optimal configurations use this value
   use_rms_prop:
     Optimal value: True
     Frequency in optimal configs: 100.0% (overall: 50.0%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   rms_prop_eps:
     Optimal value: 0.0001
     Frequency in optimal configs: 100.0% (overall: 50.0%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   policy_kwargs:
     Optimal value: {'net_arch': [256, 256], 'ortho_init': True}
     Frequency in optimal configs: 100.0% (overall: 50.0%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value

 Hyperparameter Sensitivity Summary:
  🔥 learning_rate: High sensitivity (frequency: 100.0%, diversity: 0.33)
  🔥 n_steps: High sensitivity (frequency: 100.0%, diversity: 0.33)
  🔥 gamma: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 gae_lambda: High sensitivity (frequency: 100.0%, diversity: 0.33)
  🔥 ent_coef: High sensitivity (frequency: 100.0%, diversity: 0.33)
  🔥 vf_coef: High sensitivity (frequency: 100.0%, diversity: 1.00)
  🔥 max_grad_norm: High sensitivity (frequency: 100.0%, diversity: 1.00)
  🔥 use_rms_prop: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 rms_prop_eps: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 policy_kwargs: High sensitivity (frequency: 100.0%, diversity: 0.50)

 Configuration Diversity Analysis:
  Unique hyperparameter combinations: 1/1
  Diversity ratio: 1.00
   High diversity: Multiple different configurations achieve optimal performance

 Environment: LunarLander-v3 | Algorithm: ppo
Maximum reward: 291.98
Number of configurations achieving maximum reward: 1
 Hyperparameter Commonalities in Optimal Configurations:
   learning_rate:
     Optimal value: 0.0003
     Frequency in optimal configs: 100.0% (overall: 33.3%)
     Diversity ratio: 0.33
      Strong correlation: Most optimal configurations use this value
   n_steps:
     Optimal value: 1024
     Frequency in optimal configs: 100.0% (overall: 75.0%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   gamma:
     Optimal value: 0.99
     Frequency in optimal configs: 100.0% (overall: 50.0%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   gae_lambda:
     Optimal value: 0.97
     Frequency in optimal configs: 100.0% (overall: 33.3%)
     Diversity ratio: 0.33
      Strong correlation: Most optimal configurations use this value
   ent_coef:
     Optimal value: 0.01
     Frequency in optimal configs: 100.0% (overall: 33.3%)
     Diversity ratio: 0.33
      Strong correlation: Most optimal configurations use this value
   vf_coef:
     Optimal value: 0.5
     Frequency in optimal configs: 100.0% (overall: 100.0%)
     Diversity ratio: 1.00
      Strong correlation: Most optimal configurations use this value
   max_grad_norm:
     Optimal value: 0.5
     Frequency in optimal configs: 100.0% (overall: 100.0%)
     Diversity ratio: 1.00
      Strong correlation: Most optimal configurations use this value
   policy_kwargs:
     Optimal value: {'net_arch': [256, 256], 'ortho_init': True}
     Frequency in optimal configs: 100.0% (overall: 50.0%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   batch_size:
     Optimal value: 64.0
     Frequency in optimal configs: 100.0% (overall: 25.0%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   n_epochs:
     Optimal value: 10.0
     Frequency in optimal configs: 100.0% (overall: 75.0%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   clip_range:
     Optimal value: 0.1
     Frequency in optimal configs: 100.0% (overall: 33.3%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value
   target_kl:
     Optimal value: 0.01
     Frequency in optimal configs: 100.0% (overall: 66.7%)
     Diversity ratio: 0.50
      Strong correlation: Most optimal configurations use this value

 Hyperparameter Sensitivity Summary:
  🔥 learning_rate: High sensitivity (frequency: 100.0%, diversity: 0.33)
  🔥 n_steps: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 gamma: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 gae_lambda: High sensitivity (frequency: 100.0%, diversity: 0.33)
  🔥 ent_coef: High sensitivity (frequency: 100.0%, diversity: 0.33)
  🔥 vf_coef: High sensitivity (frequency: 100.0%, diversity: 1.00)
  🔥 max_grad_norm: High sensitivity (frequency: 100.0%, diversity: 1.00)
  🔥 policy_kwargs: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 batch_size: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 n_epochs: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 clip_range: High sensitivity (frequency: 100.0%, diversity: 0.50)
  🔥 target_kl: High sensitivity (frequency: 100.0%, diversity: 0.50)

 Configuration Diversity Analysis:
  Unique hyperparameter combinations: 1/1
  Diversity ratio: 1.00
   High diversity: Multiple different configurations achieve optimal performance

 All Configurations Achieving Maximum Reward
============================================================

 Environment: Hopper-v4 | Algorithm: a2c
Maximum reward value: 2784.75
Number of experiments achieving this reward: 1
----------------------------------------

Configuration #1:
  Experiment ID: 20251107_190538_195621_pid2981425_seed0
  Reward stability: ±477.17
  Hyperparameter configuration:
    learning_rate: 0.0007
    n_steps: 128
    gamma: 0.99
    gae_lambda: 0.95
    ent_coef: 0.0
    vf_coef: 0.5
    max_grad_norm: 0.5
    use_rms_prop: True
    rms_prop_eps: 0.0001
    use_sde: True
    sde_sample_freq: 4.0
    policy_kwargs: {'net_arch': [256, 256], 'ortho_init': True}

 Environment: LunarLanderContinuous-v3 | Algorithm: a2c
Maximum reward value: 289.21
Number of experiments achieving this reward: 1
----------------------------------------

Configuration #1:
  Experiment ID: 20251105_135646_964640_pid2403106_seed0
  Reward stability: ±19.16
  Hyperparameter configuration:
    learning_rate: 0.0003
    n_steps: 256
    gamma: 0.99
    gae_lambda: 0.97
    ent_coef: 0.0
    vf_coef: 0.5
    max_grad_norm: 0.5
    use_rms_prop: True
    rms_prop_eps: 0.0001
    use_sde: True
    sde_sample_freq: 64.0
    policy_kwargs: {'net_arch': [256, 256], 'ortho_init': True}

 Environment: LunarLanderContinuous-v3 | Algorithm: ppo
Maximum reward value: 297.00
Number of experiments achieving this reward: 1
----------------------------------------

Configuration #1:
  Experiment ID: 20251101_092021_512567_pid1116286_seed0
  Reward stability: ±16.00
  Hyperparameter configuration:
    learning_rate: 0.0001
    n_steps: 1024
    gamma: 0.995
    gae_lambda: 0.97
    ent_coef: 0.01
    vf_coef: 0.5
    max_grad_norm: 0.5
    use_sde: True
    sde_sample_freq: 64.0
    policy_kwargs: {'net_arch': [128, 128], 'ortho_init': True, 'log_std_init': -0.5}
    batch_size: 64.0
    n_epochs: 10.0
    clip_range: 0.2
    target_kl: 0.02

 Environment: CartPole-v1 | Algorithm: a2c
Maximum reward value: 500.00
Number of experiments achieving this reward: 375
 Due to a large number of optimal configurations, outputting the 3 most stable ones:

Configuration #1:
  Experiment ID: 20251026_195745_033169_pid1003504_seed0
  Reward stability: ±0.00
  Hyperparameter configuration:
    learning_rate: 0.0007
    n_steps: 128
    gamma: 0.995
    gae_lambda: 0.95
    ent_coef: 0.001
    vf_coef: 0.5
    max_grad_norm: 0.5
    use_rms_prop: True
    rms_prop_eps: 0.0001
    policy_kwargs: {'net_arch': [64, 64]}

Configuration #2:
  Experiment ID: 20251026_180219_284787_pid982498_seed0
  Reward stability: ±0.00
  Hyperparameter configuration:
    learning_rate: 0.001
    n_steps: 64
    gamma: 0.99
    gae_lambda: 0.9
    ent_coef: 0.001
    vf_coef: 0.5
    max_grad_norm: 0.5
    use_rms_prop: True
    rms_prop_eps: 0.0001
    policy_kwargs: {'net_arch': [128, 128]}

Configuration #3:
  Experiment ID: 20251026_191337_247187_pid995973_seed0
  Reward stability: ±0.00
  Hyperparameter configuration:
    learning_rate: 0.0007
    n_steps: 64
    gamma: 0.99
    gae_lambda: 0.9
    ent_coef: 0.001
    vf_coef: 0.5
    max_grad_norm: 0.5
    use_rms_prop: True
    rms_prop_eps: 1e-05
    policy_kwargs: {'net_arch': [128, 128]}

 Environment: CartPole-v1 | Algorithm: ppo
Maximum reward value: 500.00
Number of experiments achieving this reward: 1511
 Due to a large number of optimal configurations, outputting the 3 most stable ones:

Configuration #1:
  Experiment ID: 20251027_043006_663272_pid1092220_seed0
  Reward stability: ±0.00
  Hyperparameter configuration:
    learning_rate: 0.0001
    n_steps: 128
    gamma: 0.995
    gae_lambda: 0.9
    ent_coef: 0.02
    vf_coef: 0.5
    max_grad_norm: 0.5
    policy_kwargs: {'net_arch': [128, 128]}
    batch_size: 32.0
    n_epochs: 20.0
    clip_range: 0.3
    target_kl: 0.02

Configuration #2:
  Experiment ID: 20251026_232743_169136_pid1041029_seed0
  Reward stability: ±0.00
  Hyperparameter configuration:
    learning_rate: 0.0003
    n_steps: 128
    gamma: 0.99
    gae_lambda: 0.95
    ent_coef: 0.0
    vf_coef: 0.5
    max_grad_norm: 0.5
    policy_kwargs: {'net_arch': [64, 64]}
    batch_size: 32.0
    n_epochs: 10.0
    clip_range: 0.3
    target_kl: 0.02

Configuration #3:
  Experiment ID: 20251027_033419_470921_pid1083976_seed0
  Reward stability: ±0.00
  Hyperparameter configuration:
    learning_rate: 0.001
    n_steps: 512
    gamma: 0.995
    gae_lambda: 0.95
    ent_coef: 0.0
    vf_coef: 0.5
    max_grad_norm: 0.5
    policy_kwargs: {'net_arch': [64, 64]}
    batch_size: 32.0
    n_epochs: 10.0
    clip_range: 0.2
    target_kl: 0.02

 Environment: LunarLander-v3 | Algorithm: a2c
Maximum reward value: 255.44
Number of experiments achieving this reward: 1
----------------------------------------

Configuration #1:
  Experiment ID: 20251027_151212_950323_pid1268875_seed0
  Reward stability: ±17.49
  Hyperparameter configuration:
    learning_rate: 0.0007
    n_steps: 256
    gamma: 0.995
    gae_lambda: 0.97
    ent_coef: 0.01
    vf_coef: 0.5
    max_grad_norm: 0.5
    use_rms_prop: True
    rms_prop_eps: 0.0001
    policy_kwargs: {'net_arch': [256, 256], 'ortho_init': True}

 Environment: LunarLander-v3 | Algorithm: ppo
Maximum reward value: 291.98
Number of experiments achieving this reward: 1
----------------------------------------

Configuration #1:
  Experiment ID: 20251029_023406_629011_pid149793_seed0
  Reward stability: ±18.75
  Hyperparameter configuration:
    learning_rate: 0.0003
    n_steps: 1024
    gamma: 0.99
    gae_lambda: 0.97
    ent_coef: 0.01
    vf_coef: 0.5
    max_grad_norm: 0.5
    policy_kwargs: {'net_arch': [256, 256], 'ortho_init': True}
    batch_size: 64.0
    n_epochs: 10.0
    clip_range: 0.1
    target_kl: 0.01

 Hyperparameter Configuration Recommendations (by Environment-Algorithm Combination)
============================================================

 Environment: Hopper-v4 | Algorithm: a2c
Number of experiments: 1224
Recommended experiment ID: 20251107_190538_195621_pid2981425_seed0
Recommended configuration (reward: 2784.75 ± 477.17):
  learning_rate: 0.0007
  n_steps: 128
  gamma: 0.99
  gae_lambda: 0.95
  ent_coef: 0.0
  vf_coef: 0.5
  max_grad_norm: 0.5
  use_rms_prop: True
  rms_prop_eps: 0.0001
  use_sde: True
  sde_sample_freq: 4.0
  policy_kwargs: {'net_arch': [256, 256], 'ortho_init': True}

 Environment: LunarLanderContinuous-v3 | Algorithm: a2c
Number of experiments: 3456
Recommended experiment ID: 20251105_135646_964640_pid2403106_seed0
Recommended configuration (reward: 289.21 ± 19.16):
  learning_rate: 0.0003
  n_steps: 256
  gamma: 0.99
  gae_lambda: 0.97
  ent_coef: 0.0
  vf_coef: 0.5
  max_grad_norm: 0.5
  use_rms_prop: True
  rms_prop_eps: 0.0001
  use_sde: True
  sde_sample_freq: 64.0
  policy_kwargs: {'net_arch': [256, 256], 'ortho_init': True}

 Environment: LunarLanderContinuous-v3 | Algorithm: ppo
Number of experiments: 3840
Recommended experiment ID: 20251101_092021_512567_pid1116286_seed0
Recommended configuration (reward: 297.00 ± 16.00):
  learning_rate: 0.0001
  n_steps: 1024
  gamma: 0.995
  gae_lambda: 0.97
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  use_sde: True
  sde_sample_freq: 64.0
  policy_kwargs: {'net_arch': [128, 128], 'ortho_init': True, 'log_std_init': -0.5}
  batch_size: 64.0
  n_epochs: 10.0
  clip_range: 0.2
  target_kl: 0.02

 Environment: CartPole-v1 | Algorithm: a2c
Number of experiments: 786
Recommended experiment ID: 20251026_195745_033169_pid1003504_seed0
Recommended configuration (reward: 500.00 ± 0.00):
  learning_rate: 0.0007
  n_steps: 128
  gamma: 0.995
  gae_lambda: 0.95
  ent_coef: 0.001
  vf_coef: 0.5
  max_grad_norm: 0.5
  use_rms_prop: True
  rms_prop_eps: 0.0001
  policy_kwargs: {'net_arch': [64, 64]}

 Environment: CartPole-v1 | Algorithm: ppo
Number of experiments: 1585
Recommended experiment ID: 20251027_043006_663272_pid1092220_seed0
Recommended configuration (reward: 500.00 ± 0.00):
  learning_rate: 0.0001
  n_steps: 128
  gamma: 0.995
  gae_lambda: 0.9
  ent_coef: 0.02
  vf_coef: 0.5
  max_grad_norm: 0.5
  policy_kwargs: {'net_arch': [128, 128]}
  batch_size: 32.0
  n_epochs: 20.0
  clip_range: 0.3
  target_kl: 0.02

 Environment: LunarLander-v3 | Algorithm: a2c
Number of experiments: 1296
Recommended experiment ID: 20251027_151212_950323_pid1268875_seed0
Recommended configuration (reward: 255.44 ± 17.49):
  learning_rate: 0.0007
  n_steps: 256
  gamma: 0.995
  gae_lambda: 0.97
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  use_rms_prop: True
  rms_prop_eps: 0.0001
  policy_kwargs: {'net_arch': [256, 256], 'ortho_init': True}

 Environment: LunarLander-v3 | Algorithm: ppo
Number of experiments: 1296
Recommended experiment ID: 20251029_023406_629011_pid149793_seed0
Recommended configuration (reward: 291.98 ± 18.75):
  learning_rate: 0.0003
  n_steps: 1024
  gamma: 0.99
  gae_lambda: 0.97
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
  policy_kwargs: {'net_arch': [256, 256], 'ortho_init': True}
  batch_size: 64.0
  n_epochs: 10.0
  clip_range: 0.1
  target_kl: 0.01

📋 Environment-Algorithm Performance Summary Table
============================================================
CartPole-v1 - ppo:
  Experiments: 1585 | Average:  490.6 ±  53.2 | Best:  500.0 | Worst:    9.2
CartPole-v1 - a2c:
  Experiments: 786 | Average:  432.5 ± 107.2 | Best:  500.0 | Worst:  104.2
Hopper-v4 - a2c:
  Experiments: 1224 | Average:  598.9 ± 380.9 | Best: 2784.7 | Worst:   50.6
LunarLander-v3 - ppo:
  Experiments: 1296 | Average:  179.3 ±  87.1 | Best:  292.0 | Worst: -124.4
LunarLander-v3 - a2c:
  Experiments: 1296 | Average:  -32.5 ± 102.6 | Best:  255.4 | Worst: -864.6
LunarLanderContinuous-v3 - ppo:
  Experiments: 3840 | Average:  123.6 ± 134.9 | Best:  297.0 | Worst: -3243.6
LunarLanderContinuous-v3 - a2c:
  Experiments: 3456 | Average:   94.6 ± 114.6 | Best:  289.2 | Worst: -703.8


 Analysis complete!
(rl) rtx3%

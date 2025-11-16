#!/usr/bin/env python3
"""
Reinforcement Learning Convergence Speed Analysis Script - Optimized Version
Analyzes the number of episodes needed to reach specific reward thresholds
Evaluates the impact of hyperparameters on convergence speed
"""
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings
from scipy import stats
import re

class RLConvergenceAnalyzer:
    """Reinforcement Learning Convergence Speed Analyzer - Optimized Version"""

    def __init__(self, runs_dir: str = "runs", target_rewards: Optional[Dict[str, float]] = None, verbose: bool = False):
        """
        Initialize the analyzer

        Args:
            runs_dir: Experiment data directory
            target_rewards: Target reward values for each environment
            verbose: Whether to show detailed read information
        """
        self.runs_dir = Path(runs_dir)
        self.target_rewards = target_rewards or {}
        self.verbose = verbose
        self.experiments_data = []

    def load_experiment_data(self) -> List[Dict]:
        """Load all experiment data"""
        print("🔍 Scanning experiment data...")

        if not self.runs_dir.exists():
            raise FileNotFoundError(f"Experiment directory does not exist: {self.runs_dir}")

        experiments = []
        total_experiments = 0

        # Iterate through environment directories
        for env_dir in self.runs_dir.iterdir():
            if not env_dir.is_dir():
                continue

            env_name = env_dir.name
            env_experiments = 0

            # Iterate through algorithm directories
            for algo_dir in env_dir.iterdir():
                if not algo_dir.is_dir():
                    continue

                algo_name = algo_dir.name
                algo_experiments = 0

                # Iterate through specific experiment directories (timestamp format)
                for run_dir in algo_dir.iterdir():
                    if run_dir.is_dir() and re.match(r'\d{8}_\d{6}_\d+_pid\d+', run_dir.name):
                        exp_data = self._parse_single_experiment(run_dir, env_name, algo_name)
                        if exp_data:
                            experiments.append(exp_data)
                            algo_experiments += 1
                            total_experiments += 1

                if algo_experiments > 0:
                    env_experiments += algo_experiments

            if env_experiments > 0:
                print(f"📁 {env_name}: {env_experiments} experiments")

        self.experiments_data = experiments
        print(f"✅ Successfully loaded {total_experiments} experiments")
        return experiments

    def _parse_single_experiment(self, run_dir: Path, env: str, algo: str) -> Optional[Dict]:
        """Parse data for a single experiment"""
        try:
            # Read config file
            config_file = run_dir / "config.yml"
            if not config_file.exists():
                return None

            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            # Read results file
            results_file = run_dir / "results.json"
            if not results_file.exists():
                return None

            with open(results_file, 'r') as f:
                results = json.load(f)

            # Extract episode rewards from progress.csv
            episode_rewards = self._extract_episode_rewards(run_dir, env, algo)

            # Extract basic experiment information
            experiment = {
                'experiment_id': run_dir.name,
                'environment': env,
                'algorithm': algo,
                'mean_reward': results.get('mean_reward', 0),
                'std_reward': results.get('std_reward', 0),
                'max_reward': results.get('max_reward', 0),
                'run_directory': str(run_dir),
                'episode_rewards': episode_rewards,
                'total_episodes': len(episode_rewards)
            }

            # Calculate episodes needed to reach target reward
            target_reward = self.target_rewards.get(env, None)
            if target_reward is not None and episode_rewards:
                episodes_to_target = self._calculate_episodes_to_target(episode_rewards, target_reward)
                experiment['episodes_to_target'] = episodes_to_target
                experiment['target_reward'] = target_reward
                experiment['reached_target'] = episodes_to_target is not None

                # Calculate convergence curve metrics
                conv_metrics = self._calculate_convergence_metrics(episode_rewards, target_reward)
                experiment.update(conv_metrics)

            # Extract hyperparameters - improved logic
            hyperparams = {}

            # Try to get hyperparameters from different locations
            if 'hyperparams' in config:
                hyperparams = config['hyperparams']
            elif 'hyperparams_parsed' in config:
                hyperparams = config['hyperparams_parsed']
            elif 'hyperparameters' in config:
                hyperparams = config['hyperparameters']
            else:
                # If none of the above exist, try to parse the entire config
                hyperparams = config

            # Flatten nested hyperparameter structure
            flattened_params = self._flatten_hyperparams(hyperparams)

            # Add flattened hyperparameters
            for key, value in flattened_params.items():
                if isinstance(value, (dict, list)):
                    experiment[f'hparam_{key}'] = str(value)
                else:
                    experiment[f'hparam_{key}'] = value

            return experiment

        except Exception as e:
            if self.verbose:
                print(f"⚠️ Error parsing experiment {run_dir}: {e}")
            return None

    def _flatten_hyperparams(self, params: Union[Dict, Any], prefix: str = '') -> Dict:
        """Flatten nested hyperparameter structure"""
        flattened = {}
        if not isinstance(params, dict):
            return {prefix: params} if prefix else {}

        for key, value in params.items():
            full_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                flattened.update(self._flatten_hyperparams(value, full_key))
            else:
                flattened[full_key] = value

        return flattened

    def _extract_episode_rewards(self, run_dir: Path, env: str, algo: str) -> List[float]:
        """Extract episode rewards from progress.csv"""
        episode_rewards = []

        # Read from progress.csv (Stable-Baselines3 standard format)
        progress_file = run_dir / "progress.csv"
        if progress_file.exists():
            try:
                df = pd.read_csv(progress_file)
                # Try common reward column names
                reward_columns = ['rollout/ep_rew_mean', 'episode_reward', 'ep_reward', 'reward', 'r']
                for col in reward_columns:
                    if col in df.columns:
                        episode_rewards = df[col].dropna().tolist()
                        if self.verbose:
                            print(f"  ✅ {env}-{algo}: Read {len(episode_rewards)} reward values")
                        break
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️ Failed to read progress.csv: {e}")

        return episode_rewards

    def _calculate_episodes_to_target(self, episode_rewards: List[float], target_reward: float) -> Optional[int]:
        """Calculate episodes needed to reach target reward"""
        if not episode_rewards:
            return None

        # Use moving average to smooth noise
        window_size = min(5, len(episode_rewards) // 20)
        if window_size > 1:
            smoothed_rewards = []
            for i in range(len(episode_rewards) - window_size + 1):
                window_avg = np.mean(episode_rewards[i:i+window_size])
                smoothed_rewards.append(window_avg)
        else:
            smoothed_rewards = episode_rewards

        # Find first point that reaches target
        for i, reward in enumerate(smoothed_rewards):
            if reward >= target_reward:
                # Check if subsequent episodes are stable
                future_check = min(3, len(smoothed_rewards) - i - 1)
                if future_check > 0:
                    future_rewards = smoothed_rewards[i+1:i+1+future_check]
                    if all(r >= target_reward * 0.8 for r in future_rewards):
                        return i + window_size
                return i + window_size
        return None

    def _calculate_convergence_metrics(self, episode_rewards: List[float], target_reward: float) -> Dict:
        """Calculate various convergence-related metrics"""
        if not episode_rewards:
            return {}

        rewards = np.array(episode_rewards)
        metrics = {
            'first_above_target': None,
            'consistent_above_target': None,
            'learning_speed': 0.0,
            'stability_after_convergence': 0.0,
        }

        # First time above target
        above_target_indices = np.where(rewards >= target_reward)[0]
        if len(above_target_indices) > 0:
            metrics['first_above_target'] = above_target_indices[0] + 1

            # Consistently above target (5 consecutive episodes)
            for i in range(len(above_target_indices) - 4):
                if above_target_indices[i+4] - above_target_indices[i] == 4:
                    metrics['consistent_above_target'] = above_target_indices[i] + 1
                    break

        # Learning speed (slope of reward curve)
        if len(rewards) > 1:
            x = np.arange(len(rewards))
            slope, _, _, _, _ = stats.linregress(x, rewards)
            metrics['learning_speed'] = slope

        # Stability after convergence (std of last 10% of episodes)
        if len(rewards) > 10:
            last_part = rewards[-len(rewards)//10:]
            metrics['stability_after_convergence'] = np.std(last_part) if len(last_part) > 1 else 0.0

        return metrics

    def convergence_analysis(self):
        """Perform comprehensive convergence speed analysis"""
        print("🚀 Starting Reinforcement Learning Convergence Speed Analysis")
        print("=" * 60)

        experiments = self.load_experiment_data()
        if not experiments:
            print("❌ No experiment data to analyze")
            return

        df = pd.DataFrame(experiments)

        # Basic convergence statistics
        self._basic_convergence_stats(df)

        # Convergence speed analysis
        self.analyze_convergence_speed(df)

        # Hyperparameter impact on convergence speed
        self.analyze_hyperparameter_impact(df)

        # Generate convergence performance report
        self.generate_convergence_report(df)

        print("\n🎉 Convergence analysis completed!")

    def _basic_convergence_stats(self, df: pd.DataFrame):
        """Basic convergence statistics"""
        print("\n📊 Basic Convergence Statistics")
        print("-" * 40)

        has_target_data = 'episodes_to_target' in df.columns and not df['episodes_to_target'].isna().all()

        if not has_target_data:
            print("⚠️ No target reward data found or no experiments reached target")
            print("💡 Current target rewards:", self.target_rewards)
            return

        print(f"Total experiments: {len(df)}")

        reached_df = df[df['reached_target'] == True]
        not_reached_df = df[df['reached_target'] == False]

        print(f"Experiments reached target: {len(reached_df)} ({len(reached_df)/len(df)*100:.1f}%)")
        print(f"Experiments not reached target: {len(not_reached_df)} ({len(not_reached_df)/len(df)*100:.1f}%)")

        if len(reached_df) > 0:
            avg_episodes = reached_df['episodes_to_target'].mean()
            std_episodes = reached_df['episodes_to_target'].std()
            print(f"Average episodes to target: {avg_episodes:.1f} ± {std_episodes:.1f}")

    def analyze_convergence_speed(self, df: pd.DataFrame):
        """Analyze convergence speed"""
        print("\n⏱️ Convergence Speed Analysis")
        print("-" * 30)

        if 'episodes_to_target' not in df.columns or df['episodes_to_target'].isna().all():
            print("⚠️ No convergence speed data to analyze")
            return

        # Analyze by environment and algorithm
        for env in df['environment'].unique():
            env_data = df[df['environment'] == env]
            target_reward = self.target_rewards.get(env, None)

            if target_reward is None:
                continue

            print(f"\n🌍 Environment: {env} | Target reward: {target_reward}")
            print("-" * 50)

            for algo in env_data['algorithm'].unique():
                algo_data = env_data[env_data['algorithm'] == algo]
                reached_data = algo_data[algo_data['reached_target'] == True]

                if len(reached_data) == 0:
                    best_reward = algo_data['mean_reward'].max() if len(algo_data) > 0 else 0
                    print(f"🔧 {algo}: Not reached target (Best reward: {best_reward:.1f})")
                    continue

                episodes = reached_data['episodes_to_target']

                print(f"🔧 {algo}: {len(reached_data)}/{len(algo_data)} experiments reached target")
                print(f"  Average episodes: {episodes.mean():.1f} ± {episodes.std():.1f}")
                print(f"  Fastest convergence: {episodes.min()} episodes")
                print(f"  Slowest convergence: {episodes.max()} episodes")

    def analyze_hyperparameter_impact(self, df: pd.DataFrame):
        """Analyze hyperparameter impact on convergence speed"""
        print("\n🔬 Hyperparameter Impact Analysis")
        print("-" * 50)

        if 'episodes_to_target' not in df.columns or df['episodes_to_target'].isna().all():
            print("⚠️ No convergence data to analyze hyperparameter impact")
            return

        reached_df = df[df['reached_target'] == True]

        if len(reached_df) == 0:
            print("⚠️ No experiments reached target reward")
            return

        significant_params = []

        for env in reached_df['environment'].unique():
            env_data = reached_df[reached_df['environment'] == env]

            for algo in env_data['algorithm'].unique():
                algo_data = env_data[env_data['algorithm'] == algo]

                if len(algo_data) < 2:
                    continue

                print(f"\n🌍 Environment: {env} | Algorithm: {algo}")

                hparam_cols = [col for col in algo_data.columns if col.startswith('hparam_')]

                for hparam in hparam_cols:
                    if hparam not in algo_data.columns:
                        continue

                    impact_info = self._analyze_parameter_convergence_impact(algo_data, hparam)
                    if impact_info:
                        significant_params.append((env, algo, hparam, impact_info))

        if not significant_params:
            print("💡 No significant hyperparameter impacts found")

    def _analyze_parameter_convergence_impact(self, data: pd.DataFrame, hparam: str) -> Optional[Dict]:
        """Analyze impact of a single hyperparameter on convergence speed"""
        try:
            temp_data = data[[hparam, 'episodes_to_target']].dropna().copy()
            temp_data['temp_param'] = temp_data[hparam].apply(
                lambda x: str(x) if not isinstance(x, (int, float, str, bool)) or x is None else x
            )

            unique_values = temp_data['temp_param'].nunique()
            if unique_values < 2:
                return None

            grouped_stats = temp_data.groupby('temp_param')['episodes_to_target'].agg([
                'mean', 'std', 'count'
            ]).round(2).sort_values('mean')

            # Check if there's significant difference (>20% improvement)
            best_episodes = grouped_stats.iloc[0]['mean']
            worst_episodes = grouped_stats.iloc[-1]['mean']
            improvement_pct = ((worst_episodes - best_episodes) / worst_episodes) * 100

            # Only consider improvements >20% as significant
            if improvement_pct < 20:
                return None

            param_name = hparam.replace('hparam_', '')
            best_value = str(grouped_stats.index[0])
            if len(best_value) > 20:
                best_value = best_value[:17] + "..."

            print(f"  📈 {param_name}:")
            print(f"    Best value: {best_value} ({best_episodes:.1f} episodes)")
            print(f"    Improvement: {improvement_pct:.1f}%")

            return {
                'parameter': param_name,
                'best_value': best_value,
                'best_episodes': best_episodes,
                'improvement_pct': improvement_pct
            }

        except Exception as e:
            return None

    def generate_convergence_report(self, df: pd.DataFrame):
        """Generate convergence performance summary report"""
        print("\n📋 Convergence Performance Summary Report")
        print("=" * 60)

        has_convergence_data = 'episodes_to_target' in df.columns and not df['episodes_to_target'].isna().all()

        if not has_convergence_data:
            print("Generating final performance report (no convergence data)")
            self.generate_final_performance_report(df)
            return

        print("\n🎯 Performance Summary by Environment and Algorithm:")

        for env in df['environment'].unique():
            env_data = df[df['environment'] == env]
            target_reward = self.target_rewards.get(env, 0)

            print(f"\n🌍 {env} (Target: {target_reward})")
            print("-" * 40)

            for algo in env_data['algorithm'].unique():
                algo_data = env_data[env_data['algorithm'] == algo]
                reached_data = algo_data[algo_data['reached_target'] == True]

                if len(algo_data) > 0:
                    success_rate = len(reached_data) / len(algo_data) * 100

                    if len(reached_data) > 0:
                        avg_episodes = reached_data['episodes_to_target'].mean()
                        fastest_episodes = reached_data['episodes_to_target'].min()
                        best_config = self._find_best_config(reached_data)
                        print(f"🔧 {algo}: Success rate {success_rate:.1f}%")
                        print(f"  Average episodes: {avg_episodes:.0f}")
                        print(f"  Fastest convergence: {fastest_episodes} episodes")
                        print(f"  Recommended config: {best_config}")
                    else:
                        best_reward = algo_data['mean_reward'].max()
                        print(f"🔧 {algo}: Not reached target, Best reward: {best_reward:.1f}")

    def generate_final_performance_report(self, df: pd.DataFrame):
        """Generate final performance report (when no convergence data)"""
        print("\nFinal Performance by Environment and Algorithm:")

        for env in df['environment'].unique():
            env_data = df[df['environment'] == env]
            target_reward = self.target_rewards.get(env, 0)

            print(f"\n🌍 {env} (Target: {target_reward})")
            print("-" * 40)

            for algo in env_data['algorithm'].unique():
                algo_data = env_data[env_data['algorithm'] == algo]

                if len(algo_data) > 0:
                    best_reward = algo_data['mean_reward'].max()
                    avg_reward = algo_data['mean_reward'].mean()

                    completion = (best_reward / target_reward) * 100 if target_reward > 0 else 0
                    print(f"🔧 {algo}: Best {best_reward:.1f}, Average {avg_reward:.1f}")
                    if target_reward > 0:
                        print(f"  Completion: {completion:.1f}%")

    def _find_best_config(self, data: pd.DataFrame) -> str:
        """Find the best configuration (fixed version)"""
        if len(data) == 0:
            return "No data"

        # Create explicit copy
        data_copy = data.copy()

        # Calculate performance score
        if 'episodes_to_target' in data_copy.columns and not data_copy['episodes_to_target'].isna().all():
            data_copy.loc[:, 'performance_score'] = data_copy['episodes_to_target'] + data_copy.get('std_reward', 0) * 0.1
        else:
            data_copy.loc[:, 'performance_score'] = -data_copy['mean_reward']

        best_idx = data_copy['performance_score'].idxmin()
        best_experiment = data_copy.loc[best_idx]

        # Get experiment identifier
        experiment_id = best_experiment.get('experiment_id', 'Unknown')

        # Extract meaningful hyperparameters
        meaningful_params = [
            'learning_rate', 'gamma', 'batch_size', 'buffer_size',
            'learning_starts', 'tau', 'train_freq', 'gradient_steps',
            'ent_coef', 'vf_coef', 'max_grad_norm', 'n_steps',
            'clip_range', 'target_update_interval', 'gae_lambda'
        ]

        hyperparam_list = []

        # Collect all hyperparameters
        for col in data_copy.columns:
            if col.startswith('hparam_'):
                param_name = col.replace('hparam_', '')

                # Filter for meaningful parameters
                if any(meaningful_param in param_name for meaningful_param in meaningful_params):
                    if pd.notna(best_experiment[col]):
                        param_value = best_experiment[col]

                        # Convert value to readable string
                        if isinstance(param_value, (int, float)):
                            value_str = f"{param_value:.6f}".rstrip('0').rstrip('.')
                        else:
                            value_str = str(param_value)

                        # Truncate long values
                        if len(value_str) > 20:
                            value_str = value_str[:17] + "..."

                        hyperparam_list.append(f"{param_name}={value_str}")

        # Format output
        if hyperparam_list:
            hyperparams_str = ", ".join(hyperparam_list[:8])  # Show first 8 key parameters
            return f"Instance: {experiment_id} | Hyperparams: [{hyperparams_str}]"
        else:
            # Try to show all available parameters
            all_params = [col.replace('hparam_', '') for col in data_copy.columns if col.startswith('hparam_')]
            return f"Instance: {experiment_id} | Available params: {', '.join(all_params[:5])}..."

def main():
    """Main function"""
    try:
        # Set target rewards for each environment
        target_rewards = {
            "CartPole-v1": 195.0,
            "LunarLander-v2": 200.0,
            "LunarLanderContinuous-v3": 200.0,
            "Hopper-v4": 3000.0,  # Add reasonable target reward
            "LunarLander-v3": 200.0  # Add reasonable target reward
        }

        # Initialize analyzer (verbose=False to turn off detailed output)
        analyzer = RLConvergenceAnalyzer("runs", target_rewards=target_rewards, verbose=False)

        # Perform convergence analysis
        analyzer.convergence_analysis()

    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

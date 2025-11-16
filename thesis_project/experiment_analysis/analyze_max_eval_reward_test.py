#!/usr/bin/env python3
"""
Comprehensive Reinforcement Learning Hyperparameter Impact Analysis
Analyzes hyperparameter effects on RL performance across multiple environments and algorithms
Provides environment-specific visualizations and recommendations
"""

import json
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
import os

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")
sns.set_style("whitegrid")

class RLExperimentLoader:
    """Loads and parses reinforcement learning experiment data from directory structure"""

    def __init__(self, runs_dir: str = "../runs"):
        self.runs_dir = Path(runs_dir)
        self.experiments = []

    def load_all_experiments(self) -> List[Dict[str, Any]]:
        """Load all experiment data from the directory structure"""
        print("Scanning experiment directory...")

        if not self.runs_dir.exists():
            raise FileNotFoundError(f"Runs directory not found: {self.runs_dir}")

        experiments = []

        # Iterate through environment directories
        for env_dir in self.runs_dir.iterdir():
            if not env_dir.is_dir():
                continue

            env_name = env_dir.name
            print(f"Environment: {env_name}")

            # Iterate through algorithm directories
            for algo_dir in env_dir.iterdir():
                if not algo_dir.is_dir():
                    continue

                algo_name = algo_dir.name
                exp_count = 0

                # Iterate through run directories
                for run_dir in algo_dir.iterdir():
                    if run_dir.is_dir():
                        experiment_data = self._parse_single_run(run_dir, env_name, algo_name)
                        if experiment_data:
                            experiments.append(experiment_data)
                            exp_count += 1

                print(f"  {algo_name}: {exp_count} experiments")

        self.experiments = experiments
        print(f"Successfully loaded {len(experiments)} experiments")
        return experiments

    def _parse_single_run(self, run_dir: Path, env: str, algo: str) -> Optional[Dict[str, Any]]:
        """Parse a single experiment run directory"""
        try:
            # Load configuration file
            config_file = run_dir / "config.yml"
            if not config_file.exists():
                return None

            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            # Load results file
            results_file = run_dir / "results.json"
            if not results_file.exists():
                return None

            with open(results_file, 'r') as f:
                results = json.load(f)

            # Extract basic experiment information
            experiment = {
                'experiment_id': run_dir.name,
                'environment': env,
                'algorithm': algo,
                'mean_reward': results.get('mean_reward', 0),
                'std_reward': results.get('std_reward', 0),
                'max_reward': results.get('max_reward', 0),
                'total_timesteps': results.get('total_timesteps', 0),
                'run_directory': str(run_dir),
            }

            # Extract hyperparameters
            hyperparams = config.get('hyperparams_parsed', {})
            for key, value in hyperparams.items():
                # Handle unhashable types
                if isinstance(value, (dict, list)):
                    experiment[f'hparam_{key}'] = str(value)
                else:
                    experiment[f'hparam_{key}'] = value

            return experiment

        except Exception as e:
            print(f"Error parsing experiment {run_dir}: {e}")
            return None

class RLResultVisualizer:
    """Creates comprehensive visualizations for RL analysis results"""

    def __init__(self, output_dir: str = "analyze_max_eval_reward"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def plot_optimal_learning_curves(self, df: pd.DataFrame, recommendations: Dict):
        """
        Plots learning curves for the optimal configurations in each environment.
        This reveals how the best-performing setups learned over time.
        """
        print("\n📈 Generating optimal configuration learning curves...")

        environments = df['environment'].unique()

        for env in environments:
            print(f"  Processing environment: {env}")
            env_data = df[df['environment'] == env]
            env_recommendations = {k: v for k, v in recommendations.items()
                                 if k.startswith(env)}

            if not env_recommendations:
                continue

            self._create_env_optimal_learning_curves(env_data, env_recommendations, env)

    def _create_env_optimal_learning_curves(self, env_data: pd.DataFrame,
                                           recommendations: Dict, env_name: str):
        """Creates learning curve visualizations for a specific environment"""

        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'{env_name} - Optimal Configurations Learning Analysis',
                        fontsize=16, fontweight='bold')

            # 1. Individual learning curves for top configurations
            self._plot_individual_learning_curves(axes[0, 0], recommendations, env_name)

            # 2. Algorithm comparison within environment
            self._plot_algorithm_comparison(axes[0, 1], recommendations, env_name)

            # 3. Performance stability analysis
            self._plot_performance_stability(axes[1, 0], recommendations, env_name)

            # 4. Convergence speed analysis
            self._plot_convergence_analysis(axes[1, 1], recommendations, env_name)

            plt.tight_layout()
            plt.savefig(self.output_dir / f'{env_name}_optimal_learning_curves.png',
                       dpi=300, bbox_inches='tight')
            plt.close()

            print(f"    ✅ Saved: {env_name}_optimal_learning_curves.png")

        except Exception as e:
            print(f"    ❌ Error creating learning curves for {env_name}: {e}")

    def _plot_individual_learning_curves(self, ax, recommendations: Dict, env_name: str):
        """Plots individual learning curves for top configurations"""

        plotted_configs = 0
        for key, config in list(recommendations.items())[:4]:  # Limit to top 4
            if plotted_configs >= 4:
                break

            try:
                # Extract experiment directory from configuration
                exp_id = config['experiment_id']
                run_directory = config.get('run_directory', '')

                # Load training progress data
                progress_file = Path(run_directory) / "progress.csv"
                if progress_file.exists():
                    progress_data = pd.read_csv(progress_file)

                    if 'rollout/episode_reward_mean' in progress_data.columns:
                        rewards = progress_data['rollout/episode_reward_mean']
                        steps = progress_data['time/total_timesteps']

                        # Smooth the curve for better visualization
                        smoothed_rewards = rewards.rolling(window=10, center=True).mean()

                        algo_name = config['algorithm']
                        label = f"{algo_name} (Final: {config['max_reward']:.1f})"

                        ax.plot(steps, smoothed_rewards, label=label, linewidth=1.5,
                               alpha=0.8)
                        plotted_configs += 1

            except Exception as e:
                print(f"      ⚠️ Could not load data for {key}: {e}")
                continue

        ax.set_xlabel('Training Timesteps')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Learning Curves of Top Configurations')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if plotted_configs == 0:
            ax.text(0.5, 0.5, 'No training progress data available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Learning Curves (Data Not Available)')

    def _plot_algorithm_comparison(self, ax, recommendations: Dict, env_name: str):
        """Compares algorithm performance within the same environment"""

        algorithm_performance = {}

        for key, config in recommendations.items():
            algo = config['algorithm']
            exp_id = config['experiment_id']
            run_directory = config.get('run_directory', '')

            try:
                progress_file = Path(run_directory) / "progress.csv"
                if progress_file.exists():
                    progress_data = pd.read_csv(progress_file)

                    if 'rollout/episode_reward_mean' in progress_data.columns:
                        rewards = progress_data['rollout/episode_reward_mean']

                        # Store the best reward achieved at each training percentage
                        if algo not in algorithm_performance:
                            algorithm_performance[algo] = []

                        # Normalize by training progress and track best reward
                        max_possible = len(rewards)
                        for i, reward in enumerate(rewards):
                            progress_pct = (i / max_possible) * 100 if max_possible > 0 else 0
                            algorithm_performance[algo].append((progress_pct, reward))

            except Exception as e:
                continue

        # Plot algorithm comparison
        for algo, performance_data in algorithm_performance.items():
            if performance_data:
                progress, rewards = zip(*performance_data)
                # Smooth for better visualization
                smoothed_rewards = pd.Series(rewards).rolling(window=5).mean()
                ax.plot(progress, smoothed_rewards, label=algo, linewidth=2, alpha=0.8)

        ax.set_xlabel('Training Progress (%)')
        ax.set_ylabel('Episode Reward')
        ax.set_title('Algorithm Performance Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_performance_stability(self, ax, recommendations: Dict, env_name: str):
        """Analyzes the stability of learning for optimal configurations"""

        stability_data = []

        for key, config in recommendations.items():
            algo = config['algorithm']
            exp_id = config['experiment_id']
            run_directory = config.get('run_directory', '')
            final_reward = config['max_reward']

            try:
                progress_file = Path(run_directory) / "progress.csv"
                if progress_file.exists():
                    progress_data = pd.read_csv(progress_file)

                    if 'rollout/episode_reward_mean' in progress_data.columns:
                        rewards = progress_data['rollout/episode_reward_mean']

                        # Calculate stability metrics
                        if len(rewards) > 10:
                            # Remove initial warm-up phase
                            stable_rewards = rewards[10:]
                            stability = 1 - (stable_rewards.std() / stable_rewards.mean())

                            stability_data.append({
                                'algorithm': algo,
                                'final_reward': final_reward,
                                'stability': stability,
                                'convergence_step': len(rewards)  # Simple proxy
                            })

            except Exception as e:
                continue

        if stability_data:
            stability_df = pd.DataFrame(stability_data)

            scatter = ax.scatter(stability_df['final_reward'],
                               stability_df['stability'],
                               c=stability_df['convergence_step'],
                               s=100, alpha=0.7, cmap='viridis')

            ax.set_xlabel('Final Reward')
            ax.set_ylabel('Training Stability (1 - CV)')
            ax.set_title('Performance vs Training Stability')
            ax.grid(True, alpha=0.3)

            # Add colorbar for convergence steps
            plt.colorbar(scatter, ax=ax, label='Training Steps to Converge')

            # Add algorithm labels
            for i, row in stability_df.iterrows():
                ax.annotate(row['algorithm'],
                          (row['final_reward'], row['stability']),
                          xytext=(5, 5), textcoords='offset points',
                          fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Insufficient data for stability analysis',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Performance vs Training Stability (Data Not Available)')

    def _plot_convergence_analysis(self, ax, recommendations: Dict, env_name: str):
        """Analyzes how quickly optimal configurations converge"""

        convergence_data = []

        for key, config in recommendations.items():
            algo = config['algorithm']
            run_directory = config.get('run_directory', '')
            final_reward = config['max_reward']

            try:
                progress_file = Path(run_directory) / "progress.csv"
                if progress_file.exists():
                    progress_data = pd.read_csv(progress_file)

                    if 'rollout/episode_reward_mean' in progress_data.columns:
                        rewards = progress_data['rollout/episode_reward_mean']

                        # Find convergence point (first point reaching 95% of final performance)
                        if len(rewards) > 10:
                            target_reward = final_reward * 0.95
                            convergence_point = None

                            for i, reward in enumerate(rewards):
                                if reward >= target_reward:
                                    convergence_point = i
                                    break

                            if convergence_point:
                                convergence_data.append({
                                    'algorithm': algo,
                                    'convergence_step': convergence_point,
                                    'final_reward': final_reward,
                                    'convergence_speed': len(rewards) - convergence_point
                                })

            except Exception as e:
                continue

        if convergence_data:
            convergence_df = pd.DataFrame(convergence_data)

            # Group by algorithm for statistical summary
            algo_groups = convergence_df.groupby('algorithm')
            algorithms = []
            convergence_speeds = []

            for algo, group in algo_groups:
                algorithms.append(algo)
                convergence_speeds.append(group['convergence_step'].mean())

            bars = ax.bar(algorithms, convergence_speeds,
                         color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
            ax.set_xlabel('Algorithm')
            ax.set_ylabel('Average Steps to Converge')
            ax.set_title('Convergence Speed Comparison')
            ax.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for bar, speed in zip(bars, convergence_speeds):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(convergence_speeds)*0.01,
                       f'{speed:.0f}', ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'Insufficient data for convergence analysis',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Convergence Speed (Data Not Available)')

# Integration with your main analysis class
class RLHyperparameterAnalyzer:
    """Main analyzer class - enhanced with optimal learning curves"""

    def comprehensive_analysis(self):
        """Enhanced analysis pipeline with optimal learning curves"""
        # ... [your existing analysis code] ...

        # After generating recommendations, add optimal learning curves
        print("\n" + "="*70)
        print("GENERATING OPTIMAL CONFIGURATION LEARNING CURVES")
        print("="*70)

        self.visualizer.plot_optimal_learning_curves(self.experiments_df, recommendations)

        print("\n✅ Optimal learning curves analysis completed!")
        return recommendations

class RLHyperparameterAnalyzer:
    """Main analyzer class for comprehensive RL hyperparameter optimization analysis"""

    def __init__(self, runs_dir: str = "runs"):
        self.loader = RLExperimentLoader(runs_dir)
        self.visualizer = RLResultVisualizer("analyze_max_eval_reward")
        self.experiments_df = None

    def comprehensive_analysis(self):
        """Execute complete analysis pipeline with environment-specific visualizations"""
        print("Starting Comprehensive RL Hyperparameter Impact Analysis")
        print("=" * 70)

        try:
            # Load data
            experiments = self.loader.load_all_experiments()
            if not experiments:
                print("No experiment data found")
                return None

            self.experiments_df = pd.DataFrame(experiments)

            # Execute analysis steps
            self._perform_basic_analysis()
            self._analyze_hyperparameter_impact()
            optimal_configs = self._identify_optimal_configurations()
            recommendations = self._generate_recommendations()

            # Generate comprehensive visualizations
            print("\nEnhanced Visualization Pipeline")
            print("-" * 40)

            # 1. Environment-specific visualizations
            self.visualizer.create_environment_specific_visualizations(
                self.experiments_df, recommendations
            )

            # 2. Cross-environment comparisons
            self.visualizer.create_cross_environment_comparison(self.experiments_df)

            print(f"\nAnalysis completed successfully!")
            print(f"Visualizations saved to: {self.visualizer.output_dir}")

            return recommendations

        except Exception as e:
            print(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _perform_basic_analysis(self):
        """Perform basic statistical analysis"""
        print("\nBASIC STATISTICAL ANALYSIS")
        print("-" * 40)

        df = self.experiments_df
        print(f"Total experiments: {len(df):,}")
        print(f"Environments: {df['environment'].unique().tolist()}")
        print(f"Algorithms: {df['algorithm'].unique().tolist()}")
        print(f"Overall mean reward: {df['mean_reward'].mean():.2f} ± {df['mean_reward'].std():.2f}")
        print(f"Reward range: {df['mean_reward'].min():.2f} - {df['mean_reward'].max():.2f}")

        # Environment-algorithm combinations summary
        env_algo_summary = df.groupby(['environment', 'algorithm']).agg({
            'mean_reward': ['count', 'mean', 'std', 'max', 'min']
        }).round(2)

        print("\nPerformance by Environment-Algorithm Combinations:")
        for (env, algo), data in env_algo_summary.iterrows():
            # Safely extract values with proper type handling
            count = data[('mean_reward', 'count')]
            mean_reward = data[('mean_reward', 'mean')]
            max_reward = data[('mean_reward', 'max')]

            # Convert count to integer safely
            try:
                count_int = int(count)
                print(f"  {env} - {algo}: {count_int:3d} runs, "
                      f"mean: {mean_reward:6.1f}, max: {max_reward:6.1f}")
            except (ValueError, TypeError):
                print(f"  {env} - {algo}: {count} runs, "
                      f"mean: {mean_reward:6.1f}, max: {max_reward:6.1f}")

    def _analyze_hyperparameter_impact(self):
        """Analyze impact of individual hyperparameters on performance"""
        print("\nHYPERPARAMETER IMPACT ANALYSIS")
        print("-" * 45)

        df = self.experiments_df
        hyperparam_cols = [col for col in df.columns if col.startswith('hparam_')]

        for env in df['environment'].unique():
            env_data = df[df['environment'] == env]

            for algo in env_data['algorithm'].unique():
                algo_data = env_data[env_data['algorithm'] == algo]

                if len(algo_data) < 5:  # Skip if insufficient data
                    continue

                print(f"\n{env} | {algo}")
                print(f"Experiments: {len(algo_data)}")

                # Analyze each hyperparameter
                for hparam in hyperparam_cols:
                    if hparam not in algo_data.columns:
                        continue

                    self._analyze_single_hyperparameter(algo_data, hparam)

    def _analyze_single_hyperparameter(self, data: pd.DataFrame, hyperparam: str):
        """Analyze the effect of a single hyperparameter"""
        try:
            # Clean and prepare data
            clean_data = data[[hyperparam, 'mean_reward']].dropna()
            if len(clean_data) < 2:
                return

            param_name = hyperparam.replace('hparam_', '')
            unique_vals = clean_data[hyperparam].nunique()

            if unique_vals < 2:
                return

            # Calculate statistics per parameter value
            stats = clean_data.groupby(hyperparam)['mean_reward'].agg(
                ['count', 'mean', 'std', 'min', 'max']
            ).round(3)

            # Sort by performance
            stats = stats.sort_values('mean', ascending=False)

            best_val = stats.index[0]
            best_perf = stats.iloc[0]['mean']
            worst_perf = stats.iloc[-1]['mean']
            improvement = best_perf - worst_perf

            print(f"  {param_name}:")
            print(f"     Values tested: {unique_vals}")
            print(f"     Best: {best_val} -> {best_perf:.1f}")
            print(f"     Worst: {stats.index[-1]} -> {worst_perf:.1f}")

            if improvement > 0:
                improvement_pct = (improvement / abs(worst_perf)) * 100
                print(f"     Improvement: {improvement:.1f} ({improvement_pct:.1f}%)")

        except Exception as e:
            print(f"    Error analyzing {hyperparam}: {e}")

    def _identify_optimal_configurations(self) -> Dict[str, pd.DataFrame]:
        """Identify optimal configurations for each environment-algorithm combination"""
        print("\nOPTIMAL CONFIGURATION ANALYSIS")
        print("-" * 45)

        df = self.experiments_df
        optimal_configs = {}

        for env in df['environment'].unique():
            env_data = df[df['environment'] == env]

            for algo in env_data['algorithm'].unique():
                algo_data = env_data[env_data['algorithm'] == algo]

                if len(algo_data) == 0:
                    continue

                # Find maximum reward
                max_reward = algo_data['mean_reward'].max()
                optimal_experiments = algo_data[
                    np.abs(algo_data['mean_reward'] - max_reward) < 1e-5
                ]

                key = f"{env}_{algo}"
                optimal_configs[key] = optimal_experiments

                print(f"\n{env} | {algo}")
                print(f"Maximum reward: {max_reward:.2f}")
                print(f"Optimal configurations: {len(optimal_experiments)}")

                if len(optimal_experiments) > 0:
                    # Show most stable configuration
                    most_stable = optimal_experiments.loc[
                        optimal_experiments['std_reward'].idxmin()
                    ]
                    print(f"Most stable config: ±{most_stable['std_reward']:.2f}")

        return optimal_configs

    def _generate_recommendations(self) -> Dict[str, Dict]:
        """Generate hyperparameter recommendations for each environment-algorithm combination"""
        print("\nHYPERPARAMETER RECOMMENDATIONS")
        print("-" * 40)

        df = self.experiments_df
        recommendations = {}

        for env in df['environment'].unique():
            env_data = df[df['environment'] == env]

            for algo in env_data['algorithm'].unique():
                algo_data = env_data[env_data['algorithm'] == algo]

                if len(algo_data) == 0:
                    continue

                max_reward = algo_data['mean_reward'].max()
                optimal_configs = algo_data[
                    np.abs(algo_data['mean_reward'] - max_reward) < 1e-5
                ]

                if len(optimal_configs) > 0:
                    # Select most stable optimal configuration
                    best_config = optimal_configs.loc[
                        optimal_configs['std_reward'].idxmin()
                    ]

                    key = f"{env}_{algo}"
                    recommendations[key] = {
                        'environment': env,
                        'algorithm': algo,
                        'max_reward': max_reward,
                        'stability': best_config['std_reward'],
                        'experiment_id': best_config['experiment_id'],
                        'hyperparameters': {}
                    }

                    # Extract hyperparameters
                    hparams = {k: v for k, v in best_config.items()
                              if k.startswith('hparam_') and pd.notna(v)}

                    for param, value in hparams.items():
                        param_name = param.replace('hparam_', '')
                        recommendations[key]['hyperparameters'][param_name] = value

                    print(f"\n{env} | {algo}")
                    print(f"Recommended Experiment: {best_config['experiment_id']}")
                    print(f"Performance: {max_reward:.2f} ± {best_config['std_reward']:.2f}")
                    print("Optimal hyperparameters:")

                    for param_name, value in recommendations[key]['hyperparameters'].items():
                        value_str = str(value)
                        if len(value_str) > 50:
                            value_str = value_str[:47] + "..."
                        print(f"  {param_name}: {value_str}")

        return recommendations

def main():
    """Main execution function"""
    try:
        # Initialize analyzer
        analyzer = RLHyperparameterAnalyzer("runs")

        # Perform comprehensive analysis
        recommendations = analyzer.comprehensive_analysis()

        if recommendations:
            print("\n" + "=" * 70)
            print("FINAL RECOMMENDATIONS SUMMARY")
            print("=" * 70)

            for key, rec in recommendations.items():
                print(f"\n{rec['environment']} - {rec['algorithm']}")
                print(f"   Maximum Reward: {rec['max_reward']:.2f}")
                print(f"   Stability: ±{rec['stability']:.2f}")
                print(f"   Experiment ID: {rec['experiment_id']}")
                print("   Key Hyperparameters:")

                for param, value in list(rec['hyperparameters'].items())[:5]:
                    value_str = str(value)
                    if len(value_str) > 50:
                        value_str = value_str[:47] + "..."
                    print(f"     {param}: {value_str}")

        print(f"\nAnalysis complete! Check the 'analyze_max_eval_reward' directory for visualizations.")

    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

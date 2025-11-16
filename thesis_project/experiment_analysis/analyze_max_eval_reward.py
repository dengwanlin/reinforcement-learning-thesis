#!/usr/bin/env python3
"""
Reinforcement Learning Hyperparameter Impact Analysis Script
Outputs recommended configurations based on environment-algorithm combinations, intelligently handling numerous optimal configuration scenarios.
Focuses on analyzing the impact of hyperparameters on maximum reward.
"""
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

class RLHyperparameterAnalyzer:
    # Reinforcement Learning Hyperparameter Analyzer - Simplified Version

    def __init__(self, runs_dir: str = "../runs"):
        self.runs_dir = Path(runs_dir)
        self.experiments_data = []
        self.visualizer = RLVisualizer("hyperparameter_analysis_plots")

    def load_experiment_data(self) -> List[Dict]:
        # Load all experimental data.
        print("🔍 Starting to scan the directory")

        if not self.runs_dir.exists():
            raise FileNotFoundError(f"Directory does not exist: {self.runs_dir}")

        experiments = []

        # Traverse environment directories
        for env_dir in self.runs_dir.iterdir():
            if not env_dir.is_dir():
                continue

            env_name = env_dir.name
            print(f" Environment: {env_name}")

            # Traverse algorithm directories
            for algo_dir in env_dir.iterdir():
                if not algo_dir.is_dir():
                    continue

                algo_name = algo_dir.name
                print(f"  🔧 Algorithm: {algo_name}")

                exp_count = 0
                # Traverse specific experiment directories
                for run_dir in algo_dir.iterdir():
                    if run_dir.is_dir():
                        exp_data = self._parse_single_experiment(run_dir, env_name, algo_name)
                        if exp_data:
                            experiments.append(exp_data)
                            exp_count += 1

                print(f"     {exp_count} experiments found")

        self.experiments_data = experiments
        print(f" Successfully loaded data for {len(experiments)} experiments.")
        return experiments

    def _parse_single_experiment(self, run_dir: Path, env: str, algo: str) -> Optional[Dict]:
        """Analyze data from a single experiment"""
        try:
            # Read configuration file
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

            # Extract basic experiment information
            experiment = {
                'experiment_id': run_dir.name,
                'environment': env,
                'algorithm': algo,
                'mean_reward': results.get('mean_reward', 0),
                'std_reward': results.get('std_reward', 0),
                'max_reward': results.get('max_reward', 0),
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
            print(f" Error parsing experiment {run_dir}: {e}")
            return None

    def comprehensive_analysis(self):
        """Comprehensive analysis main function - Simplified version"""
        print(" Starting comprehensive reinforcement learning hyperparameter impact analysis (Simplified version)")
        print("Based on search results and best practices for tuning")
        print("="*80)

        # Load data
        experiments = self.load_experiment_data()

        if not experiments:
            print(" No experimental data available for analysis")
            return

        # Convert to DataFrame
        df = pd.DataFrame(experiments)

        # Basic statistics
        self._basic_statistical_analysis(df)

        # Simplified hyperparameter impact analysis (focused on maximum reward)
        self.optimized_hyperparameter_analysis(df)

        # Find all configurations achieving maximum reward
        self.find_all_max_reward_configs(df)

        # Generate configuration recommendations (by environment-algorithm combination)
        self.generate_recommendations_by_env_algo(df)

        # Create performance summary
        self.create_performance_summary(df)

        print("\n Analysis complete!")

        experiments = self.load_experiment_data()
        df = pd.DataFrame(experiments)

        # 现有分析流程
        self._basic_statistical_analysis(df)
        self.optimized_hyperparameter_analysis(df)
        self.find_all_max_reward_configs(df)
        recommendations = self.generate_recommendations_by_env_algo(df)
        self.create_performance_summary(df)

        # 新增可视化生成环节
        print("\n📊 开始生成分析可视化图表...")
        self.generate_comprehensive_visualizations(df, recommendations)

        print("\n🎉 分析和可视化完成！")

    def _basic_statistical_analysis(self, df: pd.DataFrame):
        """Basic statistical analysis"""
        print("\n Basic Statistical Analysis")
        print("-" * 40)

        print(f"Total experiments: {len(df)}")
        print(f"Environments: {df['environment'].unique().tolist()}")
        print(f"Algorithms: {df['algorithm'].unique().tolist()}")
        print(f"Overall average reward: {df['mean_reward'].mean():.2f} ± {df['mean_reward'].std():.2f}")
        print(f"Reward range: {df['mean_reward'].min():.2f} - {df['mean_reward'].max():.2f}")

    def optimized_hyperparameter_analysis(self, df: pd.DataFrame):
        """Optimized hyperparameter analysis - focused on insights related to maximum reward"""
        print("\n🔍 Hyperparameter Impact Analysis on Maximum Reward")
        print("=" * 50)

        # Analyze hyperparameter patterns only in configurations achieving maximum reward
        for env in df['environment'].unique():
            env_data = df[df['environment'] == env]

            for algo in env_data['algorithm'].unique():
                algo_data = env_data[env_data['algorithm'] == algo]

                if len(algo_data) < 2:
                    continue

                # Find maximum reward value for this combination
                max_reward = algo_data['mean_reward'].max()

                # Find all experiments achieving maximum reward
                max_reward_experiments = algo_data[
                    np.abs(algo_data['mean_reward'] - max_reward) < 1e-5
                ]

                if len(max_reward_experiments) == 0:
                    continue

                print(f"\n Environment: {env} | Algorithm: {algo}")
                print(f"Maximum reward: {max_reward:.2f}")
                print(f"Number of configurations achieving maximum reward: {len(max_reward_experiments)}")

                # Analyze hyperparameter patterns in optimal configurations
                self._analyze_optimal_config_patterns(max_reward_experiments, algo_data)

    def _analyze_optimal_config_patterns(self, optimal_configs: pd.DataFrame, all_configs: pd.DataFrame):
        """Analyze hyperparameter patterns in optimal configurations"""
        hparam_cols = [col for col in optimal_configs.columns if col.startswith('hparam_')]

        print(" Hyperparameter Commonalities in Optimal Configurations:")

        significant_params = []
        sensitivity_analysis = []

        for hparam in hparam_cols:
            if hparam not in optimal_configs.columns:
                continue

            # Analyze distribution of this parameter in optimal configurations
            optimal_values = optimal_configs[hparam].value_counts()
            all_values = all_configs[hparam].value_counts()

            # Only analyze parameters with significant patterns (one value appears > 30% of the time)
            if len(optimal_values) > 0:
                most_common_value = optimal_values.index[0]
                most_common_count = optimal_values.iloc[0]
                frequency = (most_common_count / len(optimal_configs)) * 100

                # Calculate diversity of this parameter across all configurations
                all_diversity = all_configs[hparam].nunique()
                optimal_diversity = optimal_configs[hparam].nunique()
                diversity_ratio = optimal_diversity / all_diversity if all_diversity > 0 else 0

                param_name = hparam.replace('hparam_', '')

                # Record parameter sensitivity
                sensitivity = {
                    'parameter': param_name,
                    'frequency': frequency,
                    'diversity_ratio': diversity_ratio,
                    'optimal_value': most_common_value,
                    'significance': 'High' if frequency > 60 else ('Medium' if frequency > 40 else 'Low')
                }
                sensitivity_analysis.append(sensitivity)

                if frequency > 30:  # Significant pattern detected
                    # Calculate how common this value is across all configurations
                    overall_frequency = (all_values.get(most_common_value, 0) / len(all_configs)) * 100

                    print(f"   {param_name}:")
                    print(f"     Optimal value: {most_common_value}")
                    print(f"     Frequency in optimal configs: {frequency:.1f}% (overall: {overall_frequency:.1f}%)")
                    print(f"     Diversity ratio: {diversity_ratio:.2f}")

                    if frequency > 60:
                        print(f"      Strong correlation: Most optimal configurations use this value")
                    elif frequency > 40:
                        print(f"      Medium correlation: Clear preference exists")
                    else:
                        print(f"      Weak correlation: Some preference but high diversity")

                    significant_params.append(param_name)

        if not significant_params:
            print("   No significant commonalities: Multiple hyperparameter configurations achieve optimal performance")

        # Output sensitivity summary
        if sensitivity_analysis:
            print(f"\n Hyperparameter Sensitivity Summary:")
            sensitivity_analysis.sort(key=lambda x: x['frequency'], reverse=True)

            for sensitivity in sensitivity_analysis:
                icon = "🔥" if sensitivity['significance'] == 'High' else ("⭐" if sensitivity['significance'] == 'Medium' else "⚡")
                print(f"  {icon} {sensitivity['parameter']}: {sensitivity['significance']} sensitivity "
                      f"(frequency: {sensitivity['frequency']:.1f}%, diversity: {sensitivity['diversity_ratio']:.2f})")

        # Calculate configuration diversity
        unique_configs = len(optimal_configs.drop_duplicates(subset=hparam_cols))
        diversity_ratio = unique_configs / len(optimal_configs)

        print(f"\n Configuration Diversity Analysis:")
        print(f"  Unique hyperparameter combinations: {unique_configs}/{len(optimal_configs)}")
        print(f"  Diversity ratio: {diversity_ratio:.2f}")

        if diversity_ratio > 0.7:
            print("   High diversity: Multiple different configurations achieve optimal performance")
        elif diversity_ratio > 0.3:
            print("   Medium diversity: Some common configuration patterns exist")
        else:
            print("   Low diversity: Most configurations use similar hyperparameter settings")

    def find_all_max_reward_configs(self, df: pd.DataFrame):
        """Find all configurations achieving maximum reward - Intelligent output version"""
        print("\n All Configurations Achieving Maximum Reward")
        print("=" * 60)

        for env in df['environment'].unique():
            env_data = df[df['environment'] == env]

            for algo in env_data['algorithm'].unique():
                algo_data = env_data[env_data['algorithm'] == algo]

                if len(algo_data) == 0:
                    continue

                # Find maximum reward value
                max_reward = algo_data['mean_reward'].max()

                # Find all experiments achieving maximum reward
                max_reward_experiments = algo_data[
                    np.abs(algo_data['mean_reward'] - max_reward) < 1e-5
                ]

                if len(max_reward_experiments) > 0:
                    print(f"\n Environment: {env} | Algorithm: {algo}")
                    print(f"Maximum reward value: {max_reward:.2f}")
                    print(f"Number of experiments achieving this reward: {len(max_reward_experiments)}")

                    # Intelligent output logic: Output summary if more than 5 configurations
                    if len(max_reward_experiments) > 5:
                        print(" Due to a large number of optimal configurations, outputting the 3 most stable ones:")
                        # Sort by stability and take top 3
                        most_stable = max_reward_experiments.nsmallest(3, 'std_reward')
                        self._print_detailed_configs(most_stable, max_reward)
                    else:
                        print("-" * 40)
                        self._print_detailed_configs(max_reward_experiments, max_reward)

    def _print_detailed_configs(self, experiments: pd.DataFrame, max_reward: float):
        """Output detailed configuration information"""
        # Sort by stability
        experiments = experiments.sort_values('std_reward')

        for i, (_, exp) in enumerate(experiments.iterrows(), 1):
            print(f"\nConfiguration #{i}:")
            print(f"  Experiment ID: {exp['experiment_id']}")
            print(f"  Reward stability: ±{exp['std_reward']:.2f}")

            # Display hyperparameter configuration
            hparams = {k: v for k, v in exp.items()
                      if k.startswith('hparam_') and pd.notna(v)}

            if hparams:
                print("  Hyperparameter configuration:")
                for param, value in hparams.items():
                    param_name = param.replace('hparam_', '')
                    value_str = str(value)
                    if len(value_str) > 100:
                        value_str = value_str[:100] + "..."
                    print(f"    {param_name}: {value_str}")

    def generate_recommendations_by_env_algo(self, df: pd.DataFrame):
        """Generate configuration recommendations by environment-algorithm combination"""
        print("\n Hyperparameter Configuration Recommendations (by Environment-Algorithm Combination)")
        print("=" * 60)

        recommendations = {}

        for env in df['environment'].unique():
            env_data = df[df['environment'] == env]

            for algo in env_data['algorithm'].unique():
                algo_data = env_data[env_data['algorithm'] == algo]

                if len(algo_data) == 0:
                    continue

                # Find maximum reward value for this environment-algorithm combination
                max_reward = algo_data['mean_reward'].max()

                # Find all experiments achieving maximum reward
                max_experiments = algo_data[
                    np.abs(algo_data['mean_reward'] - max_reward) < 1e-5
                ]

                if len(max_experiments) > 0:
                    # Select the most stable configuration (lowest standard deviation)
                    most_stable = max_experiments.loc[max_experiments['std_reward'].idxmin()]

                    key = f"{env}_{algo}"
                    recommendations[key] = {
                        'max_reward': max_reward,
                        'recommended_config': most_stable,
                        'stability': most_stable['std_reward']
                    }

                    print(f"\n Environment: {env} | Algorithm: {algo}")
                    print(f"Number of experiments: {len(algo_data)}")
                    print(f"Recommended experiment ID: {most_stable['experiment_id']}")
                    print(f"Recommended configuration (reward: {max_reward:.2f} ± {most_stable['std_reward']:.2f}):")

                    hparams = {k: v for k, v in most_stable.items()
                              if k.startswith('hparam_') and pd.notna(v)}

                    for param, value in hparams.items():
                        param_name = param.replace('hparam_', '')
                        value_str = str(value)
                        if len(value_str) > 100:
                            value_str = value_str[:100] + "..."
                        print(f"  {param_name}: {value_str}")

        return recommendations

    def create_performance_summary(self, df: pd.DataFrame):
        """Create performance summary table"""
        print("\n📋 Environment-Algorithm Performance Summary Table")
        print("=" * 60)

        summary_data = []

        for env in df['environment'].unique():
            env_data = df[df['environment'] == env]

            for algo in env_data['algorithm'].unique():
                algo_data = env_data[env_data['algorithm'] == algo]

                if len(algo_data) > 0:
                    rewards = algo_data['mean_reward']
                    summary_data.append({
                        'Environment': env,
                        'Algorithm': algo,
                        'Number of Experiments': len(algo_data),
                        'Average Reward': rewards.mean(),
                        'Reward Std Dev': rewards.std(),
                        'Best Reward': rewards.max(),
                        'Worst Reward': rewards.min(),
                    })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values(['Environment', 'Average Reward'], ascending=[True, False])

            for _, row in summary_df.iterrows():
                print(f"{row['Environment']} - {row['Algorithm']}:")
                print(f"  Experiments: {row['Number of Experiments']:3d} | " +
                      f"Average: {row['Average Reward']:6.1f} ± {row['Reward Std Dev']:5.1f} | " +
                      f"Best: {row['Best Reward']:6.1f} | " +
                      f"Worst: {row['Worst Reward']:6.1f}")
            print()
    def generate_comprehensive_visualizations(self, df, recommendations):
        """生成综合可视化报告"""
        try:
            # 1. 性能总览仪表板
            self.visualizer.plot_performance_summary_dashboard(df)
            print("✅ 生成性能总览仪表板")

            # 2. 超参数敏感性分析
            self.visualizer.plot_hyperparameter_sensitivity_heatmap(df)
            print("✅ 生成超参数敏感性热力图")

            # 3. 为每个推荐配置生成学习曲线对比
            best_configs = self._extract_best_config_paths(recommendations)
            if best_configs:
                self.visualizer.plot_learning_curves_comparison(best_configs)
                print("✅ 生成最优配置学习曲线对比")

            # 4. 环境算法对比分析
            self.visualizer.plot_environment_algorithm_comparison(df)
            print("✅ 生成环境算法对比分析")

            print(f"📁 所有可视化图表已保存至: {self.visualizer.output_dir}")

        except Exception as e:
            print(f"⚠️ 生成可视化图表时出错: {e}")

    def _extract_best_config_paths(self, recommendations):
        """从推荐配置中提取实验路径"""
        best_configs = {}
        for key, config in recommendations.items():
            exp_id = config['recommended_config']['experiment_id']
            run_path = config['recommended_config']['run_directory']
            best_configs[run_path] = f"{key}_{exp_id[:8]}"
        return best_configs
def main():
    """Main function"""
    try:
        # Initialize analyzer
        analyzer = RLHyperparameterAnalyzer("runs")

        # Perform comprehensive analysis
        analyzer.comprehensive_analysis()

    except Exception as e:
        print(f" Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

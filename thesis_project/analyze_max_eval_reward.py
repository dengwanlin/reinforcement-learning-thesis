#!/usr/bin/env python3
"""
修复版强化学习超参数影响分析脚本
按环境算法组合输出推荐配置，智能处理大量最优配置情况
"""
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

class RLHyperparameterAnalyzer:
    """修复版强化学习超参数分析器"""

    def __init__(self, runs_dir: str = "runs"):
        self.runs_dir = Path(runs_dir)
        self.experiments_data = []

    def load_experiment_data(self) -> List[Dict]:
        """加载所有实验数据"""
        print("🔍 开始扫描实验数据...")

        if not self.runs_dir.exists():
            raise FileNotFoundError(f"实验目录不存在: {self.runs_dir}")

        experiments = []

        # 遍历环境目录
        for env_dir in self.runs_dir.iterdir():
            if not env_dir.is_dir():
                continue

            env_name = env_dir.name
            print(f"📁 环境: {env_name}")

            # 遍历算法目录
            for algo_dir in env_dir.iterdir():
                if not algo_dir.is_dir():
                    continue

                algo_name = algo_dir.name
                print(f"  🔧 算法: {algo_name}")

                exp_count = 0
                # 遍历具体实验目录
                for run_dir in algo_dir.iterdir():
                    if run_dir.is_dir():
                        exp_data = self._parse_single_experiment(run_dir, env_name, algo_name)
                        if exp_data:
                            experiments.append(exp_data)
                            exp_count += 1

                print(f"    找到 {exp_count} 个实验")

        self.experiments_data = experiments
        print(f"✅ 成功加载 {len(experiments)} 个实验的数据")
        return experiments

    def _parse_single_experiment(self, run_dir: Path, env: str, algo: str) -> Optional[Dict]:
        """解析单个实验的数据"""
        try:
            # 读取配置文件
            config_file = run_dir / "config.yml"
            if not config_file.exists():
                return None

            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)

            # 读取结果文件
            results_file = run_dir / "results.json"
            if not results_file.exists():
                return None

            with open(results_file, 'r') as f:
                results = json.load(f)

            # 提取基本实验信息
            experiment = {
                'experiment_id': run_dir.name,
                'environment': env,
                'algorithm': algo,
                'mean_reward': results.get('mean_reward', 0),
                'std_reward': results.get('std_reward', 0),
                'max_reward': results.get('max_reward', 0),
                'run_directory': str(run_dir),
            }

            # 提取超参数
            hyperparams = config.get('hyperparams_parsed', {})

            for key, value in hyperparams.items():
                # 处理不可哈希类型
                if isinstance(value, (dict, list)):
                    experiment[f'hparam_{key}'] = str(value)
                else:
                    experiment[f'hparam_{key}'] = value

            return experiment

        except Exception as e:
            print(f"⚠️ 解析实验 {run_dir} 时出错: {e}")
            return None

    def comprehensive_analysis(self):
        """综合分析主函数"""
        print("🚀 开始强化学习超参数影响综合分析")
        print("基于搜索结果的最佳实践和调优策略")
        print("="*80)

        # 加载数据
        experiments = self.load_experiment_data()

        if not experiments:
            print("❌ 没有可分析的实验数据")
            return

        # 转换为DataFrame
        df = pd.DataFrame(experiments)

        # 基础统计
        self._basic_statistical_analysis(df)

        # 超参数影响分析
        self.safe_hyperparameter_analysis(df)

        # 找出所有达到最大奖励的配置
        self.find_all_max_reward_configs(df)

        # 生成配置推荐（按环境算法组合）
        self.generate_recommendations_by_env_algo(df)

        # 创建性能总结
        self.create_performance_summary(df)

        print("\n🎉 分析完成！")

    def _basic_statistical_analysis(self, df: pd.DataFrame):
        """基础统计分析"""
        print("\n📊 基础统计分析")
        print("-" * 40)

        print(f"总实验数量: {len(df)}")
        print(f"涉及环境: {df['environment'].unique().tolist()}")
        print(f"涉及算法: {df['algorithm'].unique().tolist()}")
        print(f"整体平均奖励: {df['mean_reward'].mean():.2f} ± {df['mean_reward'].std():.2f}")
        print(f"奖励范围: {df['mean_reward'].min():.2f} - {df['mean_reward'].max():.2f}")

    def safe_hyperparameter_analysis(self, df: pd.DataFrame):
        """安全的超参数影响分析"""
        print("\n🔬 超参数影响分析")
        print("-" * 50)

        # 按环境和算法分组分析
        for env in df['environment'].unique():
            env_data = df[df['environment'] == env]

            for algo in env_data['algorithm'].unique():
                algo_data = env_data[env_data['algorithm'] == algo]

                if len(algo_data) < 2:
                    continue

                print(f"\n🌍 环境: {env} | 算法: {algo}")
                print(f"实验数量: {len(algo_data)}")
                print(f"平均奖励: {algo_data['mean_reward'].mean():.2f} ± {algo_data['mean_reward'].std():.2f}")

                # 分析每个超参数
                hparam_cols = [col for col in algo_data.columns if col.startswith('hparam_')]

                for hparam in hparam_cols:
                    if hparam not in algo_data.columns:
                        continue
                    self._analyze_single_parameter(algo_data, hparam)

    def _analyze_single_parameter(self, data: pd.DataFrame, hparam: str):
        """分析单个超参数"""
        try:
            # 创建临时可哈希的列
            temp_data = data[[hparam, 'mean_reward']].dropna().copy()
            temp_data['temp_param'] = temp_data[hparam].apply(
                lambda x: str(x) if not isinstance(x, (int, float, str, bool)) or x is None else x
            )

            unique_values = temp_data['temp_param'].nunique()
            if unique_values < 2:
                return

            # 分组统计
            grouped_stats = temp_data.groupby('temp_param')['mean_reward'].agg([
                'mean', 'std', 'count', 'min', 'max'
            ]).round(3).sort_values('mean', ascending=False)

            param_name = hparam.replace('hparam_', '')
            print(f"\n  📈 {param_name}:")
            print(f"    测试值数量: {unique_values}")

            # 显示每个值的性能
            for value, stats in grouped_stats.iterrows():
                value_str = str(value)
                if len(value_str) > 50:
                    value_str = value_str[:47] + "..."
                print(f"    {value_str}: {stats['mean']:.2f} ± {stats['std']:.2f} (n={stats['count']})")

            # 找出最佳值
            best_value = grouped_stats.index[0]
            best_perf = grouped_stats.iloc[0]['mean']
            worst_perf = grouped_stats.iloc[-1]['mean']
            improvement = best_perf - worst_perf

            best_value_str = str(best_value)
            if len(best_value_str) > 50:
                best_value_str = best_value_str[:47] + "..."

            print(f"    🏆 最佳值: {best_value_str} (平均奖励: {best_perf:.2f})")

            if improvement > 0:
                improvement_pct = (improvement / abs(worst_perf)) * 100 if worst_perf != 0 else 0
                print(f"    📊 最佳比最差提升: {improvement:.2f} (+{improvement_pct:.1f}%)")

        except Exception as e:
            print(f"    分析超参数 {hparam} 时出错: {e}")

    def find_all_max_reward_configs(self, df: pd.DataFrame):
        """找出所有达到最大奖励的配置 - 智能输出版"""
        print("\n🏆 所有达到最大奖励的超参数配置")
        print("=" * 60)

        for env in df['environment'].unique():
            env_data = df[df['environment'] == env]

            for algo in env_data['algorithm'].unique():
                algo_data = env_data[env_data['algorithm'] == algo]

                if len(algo_data) == 0:
                    continue

                # 找出最大奖励值
                max_reward = algo_data['mean_reward'].max()

                # 找出所有达到最大奖励的实验
                max_reward_experiments = algo_data[
                    np.abs(algo_data['mean_reward'] - max_reward) < 1e-5
                ]

                if len(max_reward_experiments) > 0:
                    print(f"\n🎯 环境: {env} | 算法: {algo}")
                    print(f"最大奖励值: {max_reward:.2f}")
                    print(f"达到该奖励的实验数量: {len(max_reward_experiments)}")

                    # 智能输出逻辑：超过5个配置时输出统计摘要
                    if len(max_reward_experiments) > 5:
                        print("📊 由于达到最优的配置过多，输出统计摘要:")
                        self._print_configs_statistics(max_reward_experiments)
                    else:
                        print("-" * 40)
                        # 按稳定性排序
                        max_reward_experiments = max_reward_experiments.sort_values('std_reward')

                        for i, (_, exp) in enumerate(max_reward_experiments.iterrows(), 1):
                            print(f"\n配置 #{i}:")
                            print(f"  实验ID: {exp['experiment_id']}")
                            print(f"  奖励稳定性: ±{exp['std_reward']:.2f}")

                            # 显示超参数配置
                            hparams = {k: v for k, v in exp.items()
                                      if k.startswith('hparam_') and pd.notna(v)}

                            if hparams:
                                print("  超参数配置:")
                                for param, value in hparams.items():
                                    param_name = param.replace('hparam_', '')
                                    value_str = str(value)
                                    if len(value_str) > 100:
                                        value_str = value_str[:100] + "..."
                                    print(f"    {param_name}: {value_str}")

    def _print_configs_statistics(self, max_experiments: pd.DataFrame):
        """输出达到最优配置的统计摘要"""
        hparam_cols = [col for col in max_experiments.columns if col.startswith('hparam_')]

        print("📈 达到最优配置的超参数分布统计:")

        for hparam in hparam_cols:
            if hparam not in max_experiments.columns:
                continue

            # 统计该超参数的值分布
            value_counts = max_experiments[hparam].value_counts()
            total_configs = len(max_experiments)

            param_name = hparam.replace('hparam_', '')

            # 只显示出现频率超过10%的值
            common_values = []
            for value, count in value_counts.items():
                percentage = (count / total_configs) * 100
                if percentage > 10:
                    common_values.append((value, count, percentage))

            if common_values:
                common_values.sort(key=lambda x: x[2], reverse=True)

                print(f"\n  📊 {param_name}:")
                for value, count, percentage in common_values:
                    value_str = str(value)
                    if len(value_str) > 50:
                        value_str = value_str[:47] + "..."

                    print(f"    {value_str}: {count}次 ({percentage:.1f}%)")

        # 计算配置多样性
        unique_configs = len(max_experiments.drop_duplicates(subset=hparam_cols))
        diversity_ratio = unique_configs / len(max_experiments)

        print(f"\n🎯 配置多样性分析:")
        print(f"  独特超参数组合数: {unique_configs}")
        print(f"  多样性比率: {diversity_ratio:.2f}")

        if diversity_ratio > 0.7:
            print("  💡 高多样性: 多种不同配置都能达到最优性能")
        elif diversity_ratio > 0.3:
            print("  💡 中等多样性: 存在一些常见配置模式")
        else:
            print("  💡 低多样性: 大多数配置使用相似的超参数设置")

    def generate_recommendations_by_env_algo(self, df: pd.DataFrame):
        """按环境算法组合生成配置推荐"""
        print("\n💡 超参数配置推荐（按环境算法组合）")
        print("=" * 60)

        recommendations = {}

        for env in df['environment'].unique():
            env_data = df[df['environment'] == env]

            for algo in env_data['algorithm'].unique():
                algo_data = env_data[env_data['algorithm'] == algo]

                if len(algo_data) == 0:
                    continue

                # 找出该环境算法组合的最大奖励值
                max_reward = algo_data['mean_reward'].max()

                # 找出所有达到最大奖励的实验
                max_experiments = algo_data[
                    np.abs(algo_data['mean_reward'] - max_reward) < 1e-5
                ]

                if len(max_experiments) > 0:
                    # 选择最稳定的配置（奖励标准差最小）
                    most_stable = max_experiments.loc[max_experiments['std_reward'].idxmin()]

                    key = f"{env}_{algo}"
                    recommendations[key] = {
                        'max_reward': max_reward,
                        'recommended_config': most_stable,
                        'stability': most_stable['std_reward']
                    }

                    print(f"\n🌍 环境: {env} | 算法: {algo}")
                    print(f"实验数量: {len(algo_data)}")
                    print(f"推荐配置 (奖励: {max_reward:.2f} ± {most_stable['std_reward']:.2f}):")

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
        """创建性能总结表"""
        print("\n📋 环境算法性能总结表")
        print("=" * 60)

        summary_data = []

        for env in df['environment'].unique():
            env_data = df[df['environment'] == env]

            for algo in env_data['algorithm'].unique():
                algo_data = env_data[env_data['algorithm'] == algo]

                if len(algo_data) > 0:
                    rewards = algo_data['mean_reward']
                    summary_data.append({
                        '环境': env,
                        '算法': algo,
                        '实验数量': len(algo_data),
                        '平均奖励': rewards.mean(),
                        '奖励标准差': rewards.std(),
                        '最佳奖励': rewards.max(),
                        '最差奖励': rewards.min(),
                    })

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df = summary_df.sort_values(['环境', '平均奖励'], ascending=[True, False])

            for _, row in summary_df.iterrows():
                print(f"{row['环境']} - {row['算法']}:")
                print(f"  实验数: {row['实验数量']:3d} | " +
                      f"平均奖励: {row['平均奖励']:6.1f} ± {row['奖励标准差']:5.1f} | " +
                      f"最佳: {row['最佳奖励']:6.1f} | " +
                      f"最差: {row['最差奖励']:6.1f}")
            print()

def main():
    """主函数"""
    try:
        # 初始化分析器
        analyzer = RLHyperparameterAnalyzer("runs")

        # 执行综合分析
        analyzer.comprehensive_analysis()

    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析不同运行之间的方差
分析现有实验数据中的方差，包括：
1. 不同算法在相同环境下的方差比较
2. 不同超参数配置的方差分析
3. 训练过程的稳定性分析
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re

# 设置中文字体和图表样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_all_runs_data(runs_root="runs"):
    """加载所有运行的配置和结果数据"""
    runs_data = []
    runs_root = Path(runs_root)

    print("正在扫描运行目录...")

    # 遍历所有环境目录
    for env_dir in runs_root.iterdir():
        if not env_dir.is_dir():
            continue

        env_name = env_dir.name
        print(f"处理环境: {env_name}")

        # 遍历所有算法目录
        for algo_dir in env_dir.iterdir():
            if not algo_dir.is_dir():
                continue

            algo_name = algo_dir.name
            print(f"  算法: {algo_name}")

            # 遍历所有运行目录
            for run_dir in algo_dir.iterdir():
                if not run_dir.is_dir():
                    continue

                # 检查是否有结果文件
                results_file = run_dir / "results.json"
                config_file = run_dir / "config.yml"

                if results_file.exists() and config_file.exists():
                    try:
                        # 读取配置和结果
                        with open(config_file, 'r') as f:
                            config = yaml.safe_load(f)

                        with open(results_file, 'r') as f:
                            results = json.load(f)

                        # 提取训练日志中的奖励曲线（如果存在）
                        training_rewards = extract_training_rewards(run_dir)

                        run_data = {
                            'environment': env_name,
                            'algorithm': algo_name,
                            'run_id': run_dir.name,
                            'seed': config.get('seed', 0),
                            'hyperparams': config.get('hyperparams_parsed', {}),
                            'final_reward': results.get('mean_reward', 0),
                            'final_std': results.get('std_reward', 0),
                            'training_rewards': training_rewards,
                            'config_path': str(config_file),
                            'run_directory': str(run_dir)
                        }

                        runs_data.append(run_data)

                    except Exception as e:
                        print(f"    警告: 无法处理 {run_dir}: {e}")

    print(f"总共加载了 {len(runs_data)} 个运行的数据")
    return runs_data

def extract_training_rewards(run_dir):
    """从训练日志中提取奖励曲线"""
    rewards = []

    # 尝试从不同的日志文件中提取数据
    log_files = [
        run_dir / "log" / "progress.log",
        run_dir / "monitor.csv",
        run_dir / "vecmonitor.csv"
    ]

    for log_file in log_files:
        if log_file.exists():
            try:
                if log_file.name.endswith('.csv'):
                    # 处理CSV格式的监控文件
                    df = pd.read_csv(log_file)
                    if 'r' in df.columns:  # 奖励列
                        rewards.extend(df['r'].dropna().tolist())
                else:
                    # 处理文本日志文件
                    with open(log_file, 'r') as f:
                        for line in f:
                            # 匹配奖励数字（适应不同格式）
                            reward_matches = re.findall(r'reward[=:]\s*([-\d.]+)', line.lower())
                            if reward_matches:
                                rewards.append(float(reward_matches[0]))
                            # 匹配ep_rew_mean
                            ep_rew_matches = re.findall(r'ep_rew_mean[=:]\s*([-\d.]+)', line.lower())
                            if ep_rew_matches:
                                rewards.append(float(ep_rew_matches[0]))

                if rewards:  # 如果找到了奖励数据就停止
                    break

            except Exception as e:
                continue

    return rewards if len(rewards) > 10 else []  # 只返回有足够数据的曲线

def calculate_variance_stats(rewards):
    """计算奖励数据的方差统计量"""
    if len(rewards) == 0:
        return {}

    rewards_array = np.array(rewards)

    return {
        'mean_reward': np.mean(rewards_array),
        'std_reward': np.std(rewards_array),
        'cv_reward': np.std(rewards_array) / np.mean(rewards_array) if np.mean(rewards_array) != 0 else float('inf'),
        'min_reward': np.min(rewards_array),
        'max_reward': np.max(rewards_array),
        'reward_range': np.max(rewards_array) - np.min(rewards_array),
        'q1_reward': np.percentile(rewards_array, 25),
        'q3_reward': np.percentile(rewards_array, 75),
        'num_episodes': len(rewards_array)
    }

def analyze_variance_by_environment_algorithm(runs_data):
    """按环境和算法分组分析方差"""
    analysis_results = []

    # 按环境和算法分组
    grouped_data = defaultdict(list)
    for run in runs_data:
        key = (run['environment'], run['algorithm'])
        grouped_data[key].append(run)

    for (env, algo), runs in grouped_data.items():
        if len(runs) >= 2:  # 至少需要2次运行来计算方差
            final_rewards = [r['final_reward'] for r in runs]
            variance_stats = calculate_variance_stats(final_rewards)

            # 分析训练稳定性（如果有多条训练曲线）
            training_stability = analyze_training_stability(runs)

            analysis_results.append({
                'environment': env,
                'algorithm': algo,
                'num_runs': len(runs),
                **variance_stats,
                **training_stability,
                'run_ids': [r['run_id'] for r in runs],
                'example_final_rewards': final_rewards[:5]  # 显示前5个奖励值作为示例
            })

    return pd.DataFrame(analysis_results)

def analyze_training_stability(runs):
    """分析训练过程的稳定性"""
    stability_metrics = {
        'avg_training_std': 0,
        'convergence_consistency': 0,
        'num_runs_with_training_data': 0
    }

    runs_with_training_data = [r for r in runs if r['training_rewards']]
    stability_metrics['num_runs_with_training_data'] = len(runs_with_training_data)

    if len(runs_with_training_data) < 2:
        return stability_metrics

    # 计算每条训练曲线的标准差平均值
    training_stds = []
    for run in runs_with_training_data:
        if run['training_rewards']:
            training_stds.append(np.std(run['training_rewards']))

    if training_stds:
        stability_metrics['avg_training_std'] = np.mean(training_stds)

        # 计算收敛一致性（最后10%训练步数的方差）
        convergence_rewards = []
        for run in runs_with_training_data:
            if run['training_rewards']:
                last_10_percent = run['training_rewards'][-len(run['training_rewards'])//10:]
                if last_10_percent:
                    convergence_rewards.append(np.mean(last_10_percent))

        if len(convergence_rewards) >= 2:
            stability_metrics['convergence_consistency'] = np.std(convergence_rewards)

    return stability_metrics

def analyze_hyperparameter_sensitivity(runs_data):
    """分析超参数对方差的影响"""
    sensitivity_results = []

    # 按环境和算法分组
    env_algo_groups = defaultdict(list)
    for run in runs_data:
        key = (run['environment'], run['algorithm'])
        env_algo_groups[key].append(run)

    for (env, algo), runs in env_algo_groups.items():
        if len(runs) < 3:  # 需要足够的数据点
            continue

        # 按超参数配置分组
        hyperparam_groups = defaultdict(list)
        for run in runs:
            # 将超参数字典转换为可哈希的键
            param_key = str(sorted(run['hyperparams'].items()))
            hyperparam_groups[param_key].append(run)

        for param_key, param_runs in hyperparam_groups.items():
            if len(param_runs) >= 2:  # 至少需要2次运行
                final_rewards = [r['final_reward'] for r in param_runs]
                variance_stats = calculate_variance_stats(final_rewards)

                sensitivity_results.append({
                    'environment': env,
                    'algorithm': algo,
                    'hyperparams': param_key,
                    'num_runs': len(param_runs),
                    **variance_stats,
                    'example_rewards': final_rewards[:3]  # 示例奖励值
                })

    return pd.DataFrame(sensitivity_results)

def generate_variance_plots(variance_df, sensitivity_df, output_dir="variance_analysis"):
    """生成方差分析图表"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    if not variance_df.empty:
        # 1. 环境和算法组合的方差热图
        plt.figure(figsize=(12, 8))

        # 创建热图数据
        heatmap_data = variance_df.pivot_table(
            index='environment',
            columns='algorithm',
            values='cv_reward',  # 使用变异系数
            fill_value=0
        )

        sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlOrRd")
        plt.title("不同环境和算法的奖励变异系数 (CV)")
        plt.tight_layout()
        plt.savefig(output_path / "environment_algorithm_variance_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. 最终奖励的箱线图
        plt.figure(figsize=(14, 8))

        # 准备箱线图数据
        plot_data = []
        for _, row in variance_df.iterrows():
            for run_id, final_reward in zip(row['run_ids'], [row['mean_reward']] * row['num_runs']):
                plot_data.append({
                    'environment': row['environment'],
                    'algorithm': row['algorithm'],
                    'final_reward': final_reward,
                    'group': f"{row['environment']}_{row['algorithm']}"
                })

        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            sns.boxplot(data=plot_df, x='environment', y='final_reward', hue='algorithm')
            plt.title("不同配置的最终奖励分布")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_path / "final_reward_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()

    if not sensitivity_df.empty and len(sensitivity_df) > 1:
        # 3. 超参数敏感性分析
        plt.figure(figsize=(15, 10))

        # 选择变异系数最大的几个配置进行展示
        top_configs = sensitivity_df.nlargest(10, 'cv_reward')

        if not top_configs.empty:
            # 创建条形图显示不同超参数配置的方差
            plt.barh(
                range(len(top_configs)),
                top_configs['cv_reward'],
                tick_label=[f"{row['environment']}_{row['algorithm']}" for _, row in top_configs.iterrows()]
            )
            plt.xlabel("变异系数 (CV)")
            plt.title("超参数配置敏感性分析 (CV最大的10个配置)")
            plt.tight_layout()
            plt.savefig(output_path / "hyperparameter_sensitivity.png", dpi=300, bbox_inches='tight')
            plt.close()

def generate_report(variance_df, sensitivity_df, output_dir="variance_analysis"):
    """生成详细的方差分析报告"""
    output_path = Path(output_dir)

    with open(output_path / "variance_analysis_report.md", "w", encoding="utf-8") as f:
        f.write("# 强化学习算法运行方差分析报告\n\n")

        f.write("## 1. 执行摘要\n\n")
        f.write(f"- 分析时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- 总共分析运行数: {variance_df['num_runs'].sum() if not variance_df.empty else 0}\n")
        f.write(f"- 分析的环境-算法组合数: {len(variance_df) if not variance_df.empty else 0}\n\n")

        f.write("## 2. 不同算法环境组合的方差分析\n\n")
        if not variance_df.empty:
            # 按变异系数排序
            sorted_df = variance_df.sort_values('cv_reward', ascending=False)
            f.write(sorted_df.to_markdown(index=False))
        else:
            f.write("无有效数据\n")

        f.write("\n\n## 3. 关键发现\n\n")
        if not variance_df.empty:
            # 找出最稳定和最不稳定的配置
            most_stable = variance_df.loc[variance_df['cv_reward'].idxmin()]
            least_stable = variance_df.loc[variance_df['cv_reward'].idxmax()]

            f.write(f"- **最稳定的配置**: {most_stable['environment']} + {most_stable['algorithm']} (CV: {most_stable['cv_reward']:.4f})\n")
            f.write(f"- **最不稳定的配置**: {least_stable['environment']} + {least_stable['algorithm']} (CV: {least_stable['cv_reward']:.4f})\n")
            f.write(f"- **平均变异系数**: {variance_df['cv_reward'].mean():.4f}\n")

        f.write("\n\n## 4. 超参数敏感性分析\n\n")
        if not sensitivity_df.empty:
            f.write(sensitivity_df.to_markdown(index=False))
        else:
            f.write("无超参数敏感性数据\n")

        f.write("\n\n## 5. 建议\n\n")
        f.write("- 对于需要稳定性的应用，选择变异系数低的算法环境组合\n")
        f.write("- 对于超参数调优，注意高方差的配置可能需要更多次运行来获得可靠结果\n")
        f.write("- 考虑使用集成方法或多次运行平均来降低方差影响\n")

def main():
    """主函数"""
    print("开始分析运行方差...")

    # 1. 加载所有运行数据
    runs_data = load_all_runs_data("runs")

    if not runs_data:
        print("未找到有效的运行数据！")
        return

    # 2. 按环境和算法分析方差
    print("\n正在分析环境和算法方差...")
    variance_df = analyze_variance_by_environment_algorithm(runs_data)

    if not variance_df.empty:
        print("环境和算法方差分析完成！")
        print(f"分析了 {len(variance_df)} 个环境-算法组合")

        # 显示摘要
        print("\n方差分析摘要:")
        for _, row in variance_df.iterrows():
            print(f"{row['environment']} + {row['algorithm']}: "
                  f"{row['num_runs']}次运行, CV={row['cv_reward']:.4f}")
    else:
        print("没有足够的数据进行方差分析（需要至少2次相同配置的运行）")
        return

    # 3. 分析超参数敏感性
    print("\n正在分析超参数敏感性...")
    sensitivity_df = analyze_hyperparameter_sensitivity(runs_data)

    if not sensitivity_df.empty:
        print(f"超参数敏感性分析完成！分析了 {len(sensitivity_df)} 个超参数配置")

    # 4. 生成图表和报告
    #print("\n正在生成图表和报告...")
    #generate_variance_plots(variance_df, sensitivity_df)
    #generate_report(variance_df, sensitivity_df)

    # 5. 保存数据
    output_dir = Path("variance_analysis")
    output_dir.mkdir(exist_ok=True)

    variance_df.to_csv(output_dir / "variance_analysis.csv", index=False, encoding='utf-8')
    if not sensitivity_df.empty:
        sensitivity_df.to_csv(output_dir / "hyperparameter_sensitivity.csv", index=False, encoding='utf-8')

    print(f"\n分析完成！")
    print(f"- 报告保存至: {output_dir / 'variance_analysis_report.md'}")
    print(f"- 数据保存至: {output_dir / 'variance_analysis.csv'}")
    print(f"- 图表保存至: {output_dir}/")

    # 显示最重要的发现
    if not variance_df.empty:
        most_stable = variance_df.loc[variance_df['cv_reward'].idxmin()]
        least_stable = variance_df.loc[variance_df['cv_reward'].idxmax()]

        print(f"\n最重要的发现:")
        print(f"最稳定的配置: {most_stable['environment']} + {most_stable['algorithm']} (CV: {most_stable['cv_reward']:.4f})")
        print(f"最不稳定的配置: {least_stable['environment']} + {least_stable['algorithm']} (CV: {least_stable['cv_reward']:.4f})")

if __name__ == "__main__":
    main()

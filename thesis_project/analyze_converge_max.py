#!/usr/bin/env python3
"""
强化学习超参数收敛性分析系统 - 完整修复版
适配您的目录结构，支持从progress.csv读取训练历史
"""
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class RLConvergenceAnalyzer:
    """强化学习超参数收敛性分析器 - 完整版本"""

    def __init__(self, runs_dir: str = "runs"):
        self.runs_dir = Path(runs_dir)
        self.experiments_data = []

    def load_experiment_data(self) -> List[Dict]:
        """加载实验数据，支持从CSV文件读取训练历史"""
        print("🔍 扫描实验数据...")

        if not self.runs_dir.exists():
            raise FileNotFoundError(f"实验目录不存在: {self.runs_dir}")

        experiments = []

        for env_dir in self.runs_dir.iterdir():
            if not env_dir.is_dir():
                continue

            env_name = env_dir.name
            print(f"📁 环境: {env_name}")

            for algo_dir in env_dir.iterdir():
                if not algo_dir.is_dir():
                    continue

                algo_name = algo_dir.name
                print(f"  🔧 算法: {algo_name}")

                exp_count = 0
                valid_history_count = 0

                for run_dir in algo_dir.iterdir():
                    if run_dir.is_dir():
                        exp_data = self._parse_single_experiment(run_dir, env_name, algo_name)
                        if exp_data:
                            experiments.append(exp_data)
                            exp_count += 1
                            if exp_data.get('training_history'):
                                valid_history_count += 1

                print(f"    找到 {exp_count} 个实验，其中 {valid_history_count} 个有训练历史")

        self.experiments_data = experiments
        print(f"✅ 成功加载 {len(experiments)} 个实验的数据")
        return experiments

    def _parse_single_experiment(self, run_dir: Path, env: str, algo: str) -> Optional[Dict]:
        """解析单个实验的完整数据 - 支持CSV格式"""
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

            # 从progress.csv加载训练历史
            training_history = None
            progress_file = run_dir / "progress.csv"

            if progress_file.exists():
                training_history = self._load_training_history_from_csv(progress_file)

            # 构建实验数据
            experiment = {
                'experiment_id': run_dir.name,
                'environment': env,
                'algorithm': algo,
                'mean_reward': results.get('mean_reward', 0),
                'std_reward': results.get('std_reward', 0),
                'max_reward': results.get('max_reward', 0),
                'min_reward': results.get('min_reward', 0),
                'training_history': training_history,
                'run_directory': str(run_dir),
            }

            # 提取超参数
            hyperparams = config.get('hyperparams_parsed', {})
            for key, value in hyperparams.items():
                if isinstance(value, (dict, list)):
                    experiment[f'hparam_{key}'] = str(value)
                else:
                    experiment[f'hparam_{key}'] = value

            return experiment

        except Exception as e:
            print(f"⚠️ 解析实验 {run_dir} 时出错: {e}")
            return None

    def _load_training_history_from_csv(self, csv_file: Path) -> Optional[Dict]:
        """从CSV文件加载训练历史数据"""
        try:
            df = pd.read_csv(csv_file)

            # 尝试识别奖励列
            reward_data = None
            reward_column = None

            # 常见的奖励列名模式
            common_reward_columns = [
                'rollout/ep_rew_mean',      # Stable-Baselines3 标准列名
                'episode_reward',
                'mean_reward',
                'reward',
                'ep_rew_mean',
                'eval/mean_reward'
            ]

            for col in common_reward_columns:
                if col in df.columns:
                    reward_data = df[col].dropna().tolist()
                    reward_column = col
                    break

            # 如果标准列名没找到，尝试模糊匹配
            if reward_data is None:
                reward_cols = [col for col in df.columns
                             if any(keyword in col.lower() for keyword in ['reward', 'rew'])]
                if reward_cols:
                    reward_data = df[reward_cols[0]].dropna().tolist()
                    reward_column = reward_cols[0]

            if reward_data and len(reward_data) > 5:  # 确保有足够的数据点
                return {
                    'mean_reward': reward_data,
                    'source_file': str(csv_file),
                    'reward_column': reward_column,
                    'data_points': len(reward_data)
                }
            else:
                return None

        except Exception as e:
            print(f"⚠️ 读取CSV文件失败: {csv_file} - {e}")
            return None

    def analyze_convergence_dynamics(self, training_rewards: List[float],
                                   window_size: int = 100) -> Dict[str, Any]:
        """
        分析奖励曲线的收敛动态特性
        返回收敛类型、置信度和详细诊断信息
        """
        if len(training_rewards) < window_size * 3:
            return {
                'convergence_type': 'insufficient_data',
                'confidence': 0.0,
                'error': f'数据点不足: {len(training_rewards)} < {window_size * 3}'
            }

        # 将训练过程分为早期、中期、晚期三个阶段
        segment_size = len(training_rewards) // 3
        early_stage = training_rewards[:segment_size]
        mid_stage = training_rewards[segment_size:segment_size*2]
        late_stage = training_rewards[segment_size*2:]

        # 计算各阶段的统计特性
        early_mean, early_std = np.mean(early_stage), np.std(early_stage)
        mid_mean, mid_std = np.mean(mid_stage), np.std(mid_stage)
        late_mean, late_std = np.mean(late_stage), np.std(late_stage)

        # 趋势分析
        early_to_mid_trend = mid_mean - early_mean
        mid_to_late_trend = late_mean - mid_mean
        overall_trend = late_mean - early_mean

        # 稳定性分析（变异系数）
        early_cv = early_std / (abs(early_mean) + 1e-8)
        late_cv = late_std / (abs(late_mean) + 1e-8)
        stability_improvement = early_cv - late_cv

        # 收敛类型判断
        convergence_type = "uncertain"
        confidence = 0.5
        diagnosis = {}

        if (overall_trend > 0 and late_cv < early_cv * 0.8 and
            mid_to_late_trend >= -early_std * 0.5):
            convergence_type = "converging_to_max"
            confidence = min(0.95, 0.5 + abs(overall_trend) / (early_std + 1e-8))
            diagnosis["reason"] = "奖励稳步上升且后期波动减小，呈现良好收敛"

        elif overall_trend < -early_std and late_cv > early_cv:
            convergence_type = "diverging"
            confidence = 0.7 + min(0.2, abs(overall_trend) / (early_std + 1e-8))
            diagnosis["reason"] = "奖励显著下降且波动增加，可能发散"

        elif (abs(overall_trend) < early_std * 0.5 and late_cv < 0.3 and
              late_mean < np.percentile(training_rewards, 75)):
            convergence_type = "premature_convergence"
            confidence = 0.6
            diagnosis["reason"] = "奖励早期停滞，可能陷入局部最优"

        elif late_cv > early_cv * 1.5 and abs(overall_trend) < early_std:
            convergence_type = "oscillating"
            confidence = 0.8
            diagnosis["reason"] = "奖励波动加剧，可能学习率过高"

        elif early_to_mid_trend > 0 and mid_to_late_trend < -early_std * 0.5:
            convergence_type = "peak_and_decline"
            confidence = 0.75
            diagnosis["reason"] = "奖励达到峰值后下降，可能过拟合"

        else:
            convergence_type = "uncertain"
            confidence = 0.4
            diagnosis["reason"] = "收敛模式不明确"

        diagnosis.update({
            'early_stage_mean': early_mean,
            'mid_stage_mean': mid_mean,
            'late_stage_mean': late_mean,
            'trend_strength': overall_trend,
            'stability_improvement': stability_improvement,
            'final_stability': late_cv
        })

        return {
            'convergence_type': convergence_type,
            'confidence': round(confidence, 3),
            'diagnosis': diagnosis
        }

    def hyperparameter_sensitivity_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        超参数敏感性分析
        计算每个超参数对收敛性的影响程度
        """
        print("\n🔬 超参数敏感性分析")
        print("-" * 50)

        # 检查必要的列是否存在
        if 'mean_reward' not in df.columns:
            print("❌ 缺少mean_reward列，跳过敏感性分析")
            return pd.DataFrame()

        sensitivity_results = []

        # 获取所有超参数列
        hparam_cols = [col for col in df.columns if col.startswith('hparam_')]

        for hparam in hparam_cols:
            if hparam not in df.columns:
                continue

            # 清理数据
            param_data = df[[hparam, 'mean_reward']].dropna()

            if len(param_data) < 2 or len(param_data[hparam].unique()) < 2:
                continue

            param_name = hparam.replace('hparam_', '')

            # 连续参数分析
            if pd.api.types.is_numeric_dtype(param_data[hparam]):
                try:
                    correlation = param_data[hparam].corr(param_data['mean_reward'])
                    sensitivity_score = abs(correlation) if not pd.isna(correlation) else 0
                    analysis_type = "数值相关性"
                except:
                    sensitivity_score = 0
                    analysis_type = "数值相关性(计算失败)"
            else:
                # 分类参数分析：使用方差分析
                try:
                    grouped_means = param_data.groupby(hparam)['mean_reward'].mean()
                    overall_mean = param_data['mean_reward'].mean()
                    between_var = ((grouped_means - overall_mean)**2).sum()
                    total_var = ((param_data['mean_reward'] - overall_mean)**2).sum()
                    sensitivity_score = between_var / total_var if total_var > 0 else 0
                    analysis_type = "方差分析"
                except:
                    sensitivity_score = 0
                    analysis_type = "方差分析(计算失败)"

            # 评估敏感性等级
            if sensitivity_score > 0.3:
                importance = "高"
            elif sensitivity_score > 0.1:
                importance = "中"
            else:
                importance = "低"

            sensitivity_results.append({
                'parameter': param_name,
                'sensitivity_score': round(sensitivity_score, 3),
                'importance': importance,
                'analysis_type': analysis_type,
                'unique_values': len(param_data[hparam].unique()),
                'optimal_range': self._find_optimal_range(param_data, hparam, 'mean_reward')
            })

            print(f"  📊 {param_name}: 敏感性 {sensitivity_score:.3f} ({importance}影响)")

        # 按敏感性排序
        if sensitivity_results:
            sensitivity_results.sort(key=lambda x: x['sensitivity_score'], reverse=True)
            return pd.DataFrame(sensitivity_results)
        else:
            return pd.DataFrame()

    def _find_optimal_range(self, data: pd.DataFrame, param_col: str, target_col: str) -> str:
        """找出超参数的最优范围"""
        try:
            if pd.api.types.is_numeric_dtype(data[param_col]):
                # 数值参数：找到性能最好的区间
                corr = data[param_col].corr(data[target_col])
                if abs(corr) > 0.1:
                    # 找到性能最好的30%数据点对应的参数范围
                    best_data = data.nlargest(max(1, int(len(data)*0.3)), target_col)
                    min_val = best_data[param_col].min()
                    max_val = best_data[param_col].max()
                    return f"{min_val:.3f} - {max_val:.3f}"
            else:
                # 分类参数：找到最佳值
                best_value = data.groupby(param_col)[target_col].mean().idxmax()
                return str(best_value)

        except Exception:
            return "分析失败"

        return "无明显规律"

    def comprehensive_convergence_analysis(self):
        """执行全面的收敛性分析"""
        print("🚀 开始强化学习超参数收敛性综合分析")
        print("基于训练过程动态分析收敛特性")
        print("=" * 80)

        # 加载数据
        experiments = self.load_experiment_data()

        if not experiments:
            print("❌ 没有可分析的实验数据")
            return

        # 转换为DataFrame
        df = pd.DataFrame(experiments)

        # 初始化收敛类型和置信度列
        df['convergence_type'] = 'no_history_data'
        df['convergence_confidence'] = 0.0

        # 为每个实验分析收敛性
        convergence_results = []
        has_valid_history = False

        print("\n📈 分析收敛动态...")
        for i, exp in enumerate(experiments):
           # if i % 1000 == 0:  # 进度显示
          #      print(f"  已分析 {i}/{len(experiments)} 个实验...")

            if exp.get('training_history') and 'mean_reward' in exp['training_history']:
                rewards = exp['training_history']['mean_reward']

                # 确保奖励数据是有效的数字列表
                if isinstance(rewards, list) and len(rewards) > 50 and all(isinstance(x, (int, float)) for x in rewards):
                    convergence_analysis = self.analyze_convergence_dynamics(rewards)
                    convergence_results.append({
                        'experiment_id': exp['experiment_id'],
                        **convergence_analysis
                    })

                    # 更新DataFrame
                    mask = df['experiment_id'] == exp['experiment_id']
                    df.loc[mask, 'convergence_type'] = convergence_analysis['convergence_type']
                    df.loc[mask, 'convergence_confidence'] = convergence_analysis['confidence']
                    has_valid_history = True

        if has_valid_history:
            valid_count = len([r for r in convergence_results if r['convergence_type'] != 'insufficient_data'])
            print(f"✅ 成功分析 {valid_count} 个实验的收敛动态")
        else:
            print("⚠️ 警告: 没有有效的训练历史数据可用于收敛性分析")

        # 基础统计分析
        self._basic_statistical_analysis(df)

        # 收敛类型分布分析
        self._convergence_distribution_analysis(df)

        # 超参数敏感性分析
        sensitivity_df = self.hyperparameter_sensitivity_analysis(df)

        # 生成详细报告
        self._generate_detailed_report(df, sensitivity_df)

        print("\n🎉 收敛性分析完成！")
        # 在 comprehensive_convergence_analysis 方法末尾添加：
        self.experiments_data = df.to_dict('records')  # 更新数据以包含收敛分析结果

    def _basic_statistical_analysis(self, df: pd.DataFrame):
        """基础统计分析"""
        print("\n📊 基础统计分析")
        print("-" * 40)

        print(f"总实验数量: {len(df)}")
        print(f"涉及环境: {df['environment'].unique().tolist()}")
        print(f"涉及算法: {df['algorithm'].unique().tolist()}")

        if 'convergence_type' in df.columns:
            convergence_stats = df['convergence_type'].value_counts()
            print("\n📈 收敛类型分布:")
            for conv_type, count in convergence_stats.items():
                percentage = (count / len(df)) * 100
                print(f"  {conv_type}: {count}次 ({percentage:.1f}%)")

    def _convergence_distribution_analysis(self, df: pd.DataFrame):
        """收敛类型分布分析"""
        if 'convergence_type' not in df.columns:
            print("❌ 无收敛性分析数据")
            return

        print("\n📊 收敛类型详细分析")
        print("-" * 50)

        # 按环境算法分组分析
        convergence_summary = []

        for env in df['environment'].unique():
            env_data = df[df['environment'] == env]

            for algo in env_data['algorithm'].unique():
                algo_data = env_data[env_data['algorithm'] == algo]

                if len(algo_data) == 0:
                    continue

                convergence_stats = algo_data['convergence_type'].value_counts()
                total_experiments = len(algo_data)

                # 计算成功收敛比例
                successful_convergence = convergence_stats.get('converging_to_max', 0)
                success_rate = (successful_convergence / total_experiments) * 100

                print(f"\n🌍 环境: {env} | 算法: {algo}")
                print(f"  实验数量: {total_experiments}")
                print(f"  成功收敛比例: {success_rate:.1f}%")

                for conv_type, count in convergence_stats.items():
                    percentage = (count / total_experiments) * 100
                    print(f"  {conv_type}: {count}次 ({percentage:.1f}%)")

    def _generate_detailed_report(self, df: pd.DataFrame, sensitivity_df: pd.DataFrame):
        """生成详细分析报告"""
        print("\n📋 详细分析报告")
        print("=" * 60)

        # 关键发现总结
        print("🎯 关键发现总结:")

        # 最重要的超参数
        if not sensitivity_df.empty:
            top_param = sensitivity_df.iloc[0]
            print(f"  最重要的超参数: {top_param['parameter']} "
                  f"(敏感性: {top_param['sensitivity_score']})")

        # 收敛成功率
        if 'convergence_type' in df.columns:
            success_count = (df['convergence_type'] == 'converging_to_max').sum()
            success_rate = (success_count / len(df)) * 100
            print(f"  成功收敛实验: {success_count}/{len(df)} ({success_rate:.1f}%)")

        # 问题诊断
        problem_patterns = df[df['convergence_type'].isin(['diverging', 'oscillating', 'premature_convergence'])]
        if not problem_patterns.empty:
            print(f"\n🔍 问题实验诊断: 发现 {len(problem_patterns)} 个有问题的实验")

        # 调优建议
        print("\n💡 调优建议:")
        self._generate_tuning_recommendations(df, sensitivity_df)

    def _generate_tuning_recommendations(self, df: pd.DataFrame, sensitivity_df: pd.DataFrame):
        """生成超参数调优建议"""
        print("  基于收敛性分析的调优建议:")

        # 学习率建议
        if 'hparam_learning_rate' in df.columns:
            lr_data = df[df['convergence_type'] == 'converging_to_max']
            if not lr_data.empty and pd.api.types.is_numeric_dtype(lr_data['hparam_learning_rate']):
                optimal_lr = lr_data['hparam_learning_rate'].dropna()
                if len(optimal_lr) > 0:
                    print(f"  • 学习率建议范围: {optimal_lr.quantile(0.25):.2e} - {optimal_lr.quantile(0.75):.2e}")

        # 显示最重要的参数建议
        if not sensitivity_df.empty:
            top_params = sensitivity_df.head(3)
            print(f"  • 重点调优参数: {', '.join(top_params['parameter'].tolist())}")

    def answer_teachers_question(self):
        """直接回答老师关于收敛性的问题"""
        print("\n🎓 针对老师问题的专门分析")
        print("=" * 60)
        print("问题: '奖励是收敛到最大值，还是下降？'")
        print("-" * 60)

        if not self.experiments_data:
            print("❌ 没有数据可供分析")
            return

        df = pd.DataFrame(self.experiments_data)

        # 确保有收敛类型数据
        if 'convergence_type' not in df.columns:
            print("❌ 请先运行收敛性分析")
            return

        # 统计收敛类型
        convergence_stats = df['convergence_type'].value_counts()
        total_experiments = len(df)

        print(f"\n📊 基于 {total_experiments} 个实验的分析结果:")
        print("-" * 40)

        for conv_type, count in convergence_stats.items():
            percentage = (count / total_experiments) * 100

            if conv_type == 'converging_to_max':
                print(f"✅ {count}个实验({percentage:.1f}%) - 成功收敛到最大值")
                print("   特征: 奖励稳步增长，后期波动小，性能稳定")

            elif conv_type == 'diverging':
                print(f"❌ {count}个实验({percentage:.1f}%) - 奖励下降或发散")
                print("   原因: 学习率过高、超参数设置不当导致不稳定")

            elif conv_type == 'premature_convergence':
                print(f"⚠️ {count}个实验({percentage:.1f}%) - 早熟收敛(陷入局部最优)")
                print("   特征: 早期即停滞，未能达到潜在最佳性能")

            elif conv_type == 'oscillating':
                print(f"🔄 {count}个实验({percentage:.1f}%) - 奖励震荡")
                print("   原因: 探索率或学习率设置不当")

            elif conv_type == 'no_history_data':
                print(f"📊 {count}个实验({percentage:.1f}%) - 无训练历史数据")
                print("   状态: 无法分析收敛动态，仅有最终结果")
            else:
                print(f"❓ {count}个实验({percentage:.1f}%) - {conv_type}")

        # 关键结论
        success_count = convergence_stats.get('converging_to_max', 0)
        success_rate = (success_count / total_experiments) * 100

        no_data_count = convergence_stats.get('no_history_data', 0)
        no_data_rate = (no_data_count / total_experiments) * 100

        analyzable_count = total_experiments - no_data_count
        if analyzable_count > 0:
            effective_success_rate = (success_count / analyzable_count) * 100
        else:
            effective_success_rate = 0

        print(f"\n🎯 关键结论:")
        print(f"  • 总体实验: {total_experiments} 个")
        print(f"  • 可分析实验: {analyzable_count} 个 ({100-no_data_rate:.1f}%)")
        print(f"  • 成功收敛率: {success_rate:.1f}% (总体) / {effective_success_rate:.1f}% (可分析实验)")

        if effective_success_rate > 70:
            print("💡 超参数设置总体良好，大部分实验能收敛")
        elif effective_success_rate > 40:
            print("💡 超参数设置有待优化，约半数实验能收敛")
        else:
            print("💡 超参数设置需要重大调整，收敛成功率较低")

def main():
    """主函数"""
    try:
        # 初始化分析器
        analyzer = RLConvergenceAnalyzer("runs")

        # 执行全面分析
        analyzer.comprehensive_convergence_analysis()

        # 专门回答老师的问题
        analyzer.answer_teachers_question()

    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

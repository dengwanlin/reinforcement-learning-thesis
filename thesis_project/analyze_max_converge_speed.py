#!/usr/bin/env python3
"""
强化学习收敛速度分析脚本 - 简化输出版本
分析达到特定奖励所需回合数，评估超参数对收敛速度的影响
"""
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import warnings
from scipy import stats
import re

class RLConvergenceAnalyzer:
    """强化学习收敛速度分析器 - 简化输出版本"""

    def __init__(self, runs_dir: str = "runs", target_rewards: Optional[Dict[str, float]] = None, verbose: bool = False):
        """
        初始化分析器

        Args:
            runs_dir: 实验数据目录
            target_rewards: 各环境的目标奖励值字典
            verbose: 是否显示详细读取信息
        """
        self.runs_dir = Path(runs_dir)
        self.target_rewards = target_rewards or {}
        self.verbose = verbose  # 控制是否显示详细读取信息
        self.experiments_data = []

    def load_experiment_data(self) -> List[Dict]:
        """加载所有实验数据，简化输出"""
        print("🔍 开始扫描实验数据...")

        if not self.runs_dir.exists():
            raise FileNotFoundError(f"实验目录不存在: {self.runs_dir}")

        experiments = []
        total_experiments = 0

        # 遍历环境目录
        for env_dir in self.runs_dir.iterdir():
            if not env_dir.is_dir():
                continue

            env_name = env_dir.name
            env_experiments = 0

            # 遍历算法目录
            for algo_dir in env_dir.iterdir():
                if not algo_dir.is_dir():
                    continue

                algo_name = algo_dir.name
                algo_experiments = 0

                # 遍历具体实验目录（时间戳格式的目录）
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
                print(f"📁 {env_name}: {env_experiments} 个实验")

        self.experiments_data = experiments
        print(f"✅ 成功加载 {total_experiments} 个实验的数据")
        return experiments

    def _parse_single_experiment(self, run_dir: Path, env: str, algo: str) -> Optional[Dict]:
        """解析单个实验的数据，简化输出"""
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

            # 从progress.csv读取回合奖励历史
            episode_rewards = self._extract_episode_rewards(run_dir, env, algo)

            # 提取基本实验信息
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

            # 计算达到目标奖励所需的回合数
            target_reward = self.target_rewards.get(env, None)
            if target_reward is not None and episode_rewards:
                episodes_to_target = self._calculate_episodes_to_target(episode_rewards, target_reward)
                experiment['episodes_to_target'] = episodes_to_target
                experiment['target_reward'] = target_reward
                experiment['reached_target'] = episodes_to_target is not None

                # 计算收敛曲线指标
                conv_metrics = self._calculate_convergence_metrics(episode_rewards, target_reward)
                experiment.update(conv_metrics)

            # 提取超参数
            hyperparams = config.get('hyperparams', {})
            if not hyperparams:
                hyperparams = config

            for key, value in hyperparams.items():
                if isinstance(value, (dict, list)):
                    experiment[f'hparam_{key}'] = str(value)
                else:
                    experiment[f'hparam_{key}'] = value

            return experiment

        except Exception as e:
            if self.verbose:
                print(f"⚠️ 解析实验 {run_dir} 时出错: {e}")
            return None

    def _extract_episode_rewards(self, run_dir: Path, env: str, algo: str) -> List[float]:
        """从progress.csv提取回合奖励历史，简化输出"""
        episode_rewards = []

        # 从progress.csv读取（Stable-Baselines3标准格式）
        progress_file = run_dir / "progress.csv"
        if progress_file.exists():
            try:
                df = pd.read_csv(progress_file)
                # 使用rollout/ep_rew_mean列（根据您的图片信息）
                if 'rollout/ep_rew_mean' in df.columns:
                    episode_rewards = df['rollout/ep_rew_mean'].dropna().tolist()
                    if self.verbose:  # 只有verbose模式才显示详细读取信息
                        print(f"  ✅ {env}-{algo}: 读取{len(episode_rewards)}个奖励值")
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️ 读取progress.csv失败: {e}")

        return episode_rewards

    def _calculate_episodes_to_target(self, episode_rewards: List[float], target_reward: float) -> Optional[int]:
        """计算达到目标奖励所需的回合数"""
        if not episode_rewards:
            return None

        # 使用滑动窗口平均来平滑噪声
        window_size = min(5, len(episode_rewards) // 20)
        if window_size > 1:
            smoothed_rewards = []
            for i in range(len(episode_rewards) - window_size + 1):
                window_avg = np.mean(episode_rewards[i:i+window_size])
                smoothed_rewards.append(window_avg)
        else:
            smoothed_rewards = episode_rewards

        # 寻找第一个达到目标的点
        for i, reward in enumerate(smoothed_rewards):
            if reward >= target_reward:
                # 检查后续回合是否稳定
                future_check = min(3, len(smoothed_rewards) - i - 1)
                if future_check > 0:
                    future_rewards = smoothed_rewards[i+1:i+1+future_check]
                    if all(r >= target_reward * 0.8 for r in future_rewards):
                        return i + window_size
                return i + window_size
        return None

    def _calculate_convergence_metrics(self, episode_rewards: List[float], target_reward: float) -> Dict:
        """计算收敛相关的各种指标"""
        if not episode_rewards:
            return {}

        rewards = np.array(episode_rewards)
        metrics = {
            'first_above_target': None,
            'consistent_above_target': None,
            'learning_speed': 0.0,
            'stability_after_convergence': 0.0,
        }

        # 首次达到目标
        above_target_indices = np.where(rewards >= target_reward)[0]
        if len(above_target_indices) > 0:
            metrics['first_above_target'] = above_target_indices[0] + 1

            # 持续达到目标（连续5个回合）
            for i in range(len(above_target_indices) - 4):
                if above_target_indices[i+4] - above_target_indices[i] == 4:
                    metrics['consistent_above_target'] = above_target_indices[i] + 1
                    break

        # 学习速度（奖励曲线的斜率）
        if len(rewards) > 1:
            x = np.arange(len(rewards))
            slope, _, _, _, _ = stats.linregress(x, rewards)
            metrics['learning_speed'] = slope

        # 收敛后的稳定性（最后10%回合的奖励标准差）
        if len(rewards) > 10:
            last_part = rewards[-len(rewards)//10:]
            metrics['stability_after_convergence'] = np.std(last_part) if len(last_part) > 1 else 0.0

        return metrics

    def convergence_analysis(self):
        """执行收敛速度综合分析"""
        print("🚀 开始强化学习收敛速度分析")
        print("=" * 60)

        experiments = self.load_experiment_data()
        if not experiments:
            print("❌ 没有可分析的实验数据")
            return

        df = pd.DataFrame(experiments)

        # 基础收敛统计
        self._basic_convergence_stats(df)

        # 收敛速度分析
        self.analyze_convergence_speed(df)

        # 超参数对收敛速度的影响
        self.analyze_hyperparameter_impact(df)

        # 生成收敛性能报告
        self.generate_convergence_report(df)

        print("\n🎉 收敛分析完成！")

    def _basic_convergence_stats(self, df: pd.DataFrame):
        """基础收敛统计"""
        print("\n📊 基础收敛统计分析")
        print("-" * 40)

        has_target_data = 'episodes_to_target' in df.columns and not df['episodes_to_target'].isna().all()

        if not has_target_data:
            print("⚠️ 未找到目标奖励数据或没有实验达到目标")
            print("💡 当前设置的目标奖励:", self.target_rewards)
            return

        print(f"总实验数量: {len(df)}")

        reached_df = df[df['reached_target'] == True]
        not_reached_df = df[df['reached_target'] == False]

        print(f"达到目标奖励的实验: {len(reached_df)} ({len(reached_df)/len(df)*100:.1f}%)")
        print(f"未达到目标奖励的实验: {len(not_reached_df)} ({len(not_reached_df)/len(df)*100:.1f}%)")

        if len(reached_df) > 0:
            avg_episodes = reached_df['episodes_to_target'].mean()
            std_episodes = reached_df['episodes_to_target'].std()
            print(f"平均达到目标所需回合数: {avg_episodes:.1f} ± {std_episodes:.1f}")

    def analyze_convergence_speed(self, df: pd.DataFrame):
        """分析收敛速度"""
        print("\n⏱️ 收敛速度分析")
        print("-" * 30)

        if 'episodes_to_target' not in df.columns or df['episodes_to_target'].isna().all():
            print("⚠️ 无收敛速度数据可分析")
            return

        # 按环境和算法分组分析
        for env in df['environment'].unique():
            env_data = df[df['environment'] == env]
            target_reward = self.target_rewards.get(env, None)

            if target_reward is None:
                continue

            print(f"\n🌍 环境: {env} | 目标奖励: {target_reward}")
            print("-" * 50)

            for algo in env_data['algorithm'].unique():
                algo_data = env_data[env_data['algorithm'] == algo]
                reached_data = algo_data[algo_data['reached_target'] == True]

                if len(reached_data) == 0:
                    best_reward = algo_data['mean_reward'].max() if len(algo_data) > 0 else 0
                    print(f"🔧 {algo}: 未达到目标 (最佳: {best_reward:.1f})")
                    continue

                episodes = reached_data['episodes_to_target']

                print(f"🔧 {algo}: {len(reached_data)}/{len(algo_data)} 实验达到目标")
                print(f"  平均回合数: {episodes.mean():.1f} ± {episodes.std():.1f}")

                # 性能评级
                avg_episodes = episodes.mean()
                if avg_episodes < 100:
                    rating = "⭐️⭐️⭐️⭐️⭐️ (极快)"
                elif avg_episodes < 300:
                    rating = "⭐️⭐️⭐️⭐️ (快速)"
                elif avg_episodes < 500:
                    rating = "⭐️⭐️⭐️ (中等)"
                elif avg_episodes < 1000:
                    rating = "⭐️⭐️ (较慢)"
                else:
                    rating = "⭐️ (很慢)"

                print(f"  收敛速度: {rating}")

    def analyze_hyperparameter_impact(self, df: pd.DataFrame):
        """分析超参数对收敛速度的影响"""
        print("\n🔬 超参数对收敛速度的影响分析")
        print("-" * 50)

        if 'episodes_to_target' not in df.columns or df['episodes_to_target'].isna().all():
            print("⚠️ 无收敛数据可分析超参数影响")
            return

        reached_df = df[df['reached_target'] == True]

        if len(reached_df) == 0:
            print("⚠️ 没有实验达到目标奖励")
            return

        significant_params = []

        for env in reached_df['environment'].unique():
            env_data = reached_df[reached_df['environment'] == env]

            for algo in env_data['algorithm'].unique():
                algo_data = env_data[env_data['algorithm'] == algo]

                if len(algo_data) < 2:
                    continue

                print(f"\n🌍 环境: {env} | 算法: {algo}")

                hparam_cols = [col for col in algo_data.columns if col.startswith('hparam_')]

                for hparam in hparam_cols:
                    if hparam not in algo_data.columns:
                        continue

                    impact_info = self._analyze_parameter_convergence_impact(algo_data, hparam)
                    if impact_info:  # 只显示有显著影响的参数
                        significant_params.append((env, algo, hparam, impact_info))

        # 如果没有显著影响的参数，显示提示信息
        if not significant_params:
            print("💡 未发现超参数对收敛速度有显著影响")
            # 添加有意义的超参数过滤
        meaningful_params = [
        'learning_rate', 'gamma', 'batch_size', 'buffer_size',
        'learning_starts', 'tau', 'train_freq', 'gradient_steps',
        'ent_coef', 'vf_coef', 'max_grad_norm', 'n_steps'
        ]

        hparam_cols = [col for col in hparam_cols if any(param in col for param in meaningful_params)]


    def _analyze_parameter_convergence_impact(self, data: pd.DataFrame, hparam: str) -> Optional[Dict]:
        """分析单个超参数对收敛速度的影响，返回有显著影响的信息"""
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

            # 检查是否有显著差异（最佳和最差相差超过20%）
            best_episodes = grouped_stats.iloc[0]['mean']
            worst_episodes = grouped_stats.iloc[-1]['mean']
            improvement_pct = ((worst_episodes - best_episodes) / worst_episodes) * 100

            # 只有改进超过20%才认为是显著影响
            if improvement_pct < 20:
                return None

            param_name = hparam.replace('hparam_', '')
            best_value = str(grouped_stats.index[0])
            if len(best_value) > 20:
                best_value = best_value[:17] + "..."

            print(f"  📈 {param_name}:")
            print(f"    最佳值: {best_value} ({best_episodes:.1f} 回合)")
            print(f"    改进: {improvement_pct:.1f}%")

            return {
                'parameter': param_name,
                'best_value': best_value,
                'best_episodes': best_episodes,
                'improvement_pct': improvement_pct
            }

        except Exception as e:
            return None

    def generate_convergence_report(self, df: pd.DataFrame):
        """生成收敛性能总结报告"""
        print("\n📋 收敛性能总结报告")
        print("=" * 60)

        has_convergence_data = 'episodes_to_target' in df.columns and not df['episodes_to_target'].isna().all()

        if not has_convergence_data:
            print("生成最终性能报告（无收敛数据）")
            self.generate_final_performance_report(df)
            return

        print("\n🎯 各环境算法性能总结:")

        for env in df['environment'].unique():
            env_data = df[df['environment'] == env]
            target_reward = self.target_rewards.get(env, 0)

            print(f"\n🌍 {env} (目标: {target_reward})")
            print("-" * 40)

            for algo in env_data['algorithm'].unique():
                algo_data = env_data[env_data['algorithm'] == algo]
                reached_data = algo_data[algo_data['reached_target'] == True]

                if len(algo_data) > 0:
                    success_rate = len(reached_data) / len(algo_data) * 100

                    if len(reached_data) > 0:
                        avg_episodes = reached_data['episodes_to_target'].mean()
                        best_config = self._find_best_config(reached_data)
                        print(f"🔧 {algo}: 成功率 {success_rate:.1f}%, 平均 {avg_episodes:.0f} 回合")
                        print(f"   推荐: {best_config}")
                    else:
                        best_reward = algo_data['mean_reward'].max()
                        print(f"🔧 {algo}: 未达到目标, 最佳奖励: {best_reward:.1f}")

    def generate_final_performance_report(self, df: pd.DataFrame):
        """生成最终性能报告（当无收敛数据时）"""
        print("\n各环境算法最终性能:")

        for env in df['environment'].unique():
            env_data = df[df['environment'] == env]
            target_reward = self.target_rewards.get(env, 0)

            print(f"\n🌍 {env} (目标: {target_reward})")
            print("-" * 40)

            for algo in env_data['algorithm'].unique():
                algo_data = env_data[env_data['algorithm'] == algo]

                if len(algo_data) > 0:
                    best_reward = algo_data['mean_reward'].max()
                    avg_reward = algo_data['mean_reward'].mean()

                    completion = (best_reward / target_reward) * 100 if target_reward > 0 else 0
                    print(f"🔧 {algo}: 最佳 {best_reward:.1f}, 平均 {avg_reward:.1f}")
                    if target_reward > 0:
                        print(f"   完成度: {completion:.1f}%")

    def _find_best_config(self, data: pd.DataFrame) -> str:
    #找出最佳配置（修复SettingWithCopyWarning）
        if len(data) == 0:
            return "无数据"

    # 创建数据的显式副本，避免操作视图（View）的警告
        data_copy = data.copy()

    # 在副本上进行计算操作
        if 'episodes_to_target' in data_copy.columns and not data_copy          ['episodes_to_target'].isna().all():
            data_copy.loc[:, 'performance_score'] = data_copy['episodes_to_target'] + data_copy.get('std_reward', 0) * 0.1
        else:
            data_copy.loc[:, 'performance_score'] = -data_copy['mean_reward']

        best_idx = data_copy['performance_score'].idxmin()

        best_config = []
        for col in data_copy.columns:
            if col.startswith('hparam_') and pd.notna(data_copy.loc[best_idx, col]):
                param_name = col.replace('hparam_', '')
                value = data_copy.loc[best_idx, col]
                value_str = str(value)
                if len(value_str) > 15:
                    value_str = value_str[:12] + "..."
                best_config.append(f"{param_name}={value_str}")

        return ", ".join(best_config[:2]) if best_config else "默认配置"

def main():
    """主函数"""
    try:
        # 设置各环境的目标奖励值（请根据您的环境设置）
        target_rewards = {
            "CartPole-v1": 195.0,
            "LunarLander-v3": 200.0,
            "LunarLanderContinuous-v3":200.0,
        }

        # 初始化分析器（verbose=False 关闭详细输出）
        analyzer = RLConvergenceAnalyzer("runs", target_rewards=target_rewards, verbose=False)

        # 执行收敛分析
        analyzer.convergence_analysis()

    except Exception as e:
        print(f"❌ 分析过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

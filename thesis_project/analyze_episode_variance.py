#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from collections import defaultdict

def analyze_single_run_variance(run_dir, window_size=50):
    """分析单个运行的episode间方差"""
    monitor_file = Path(run_dir) / "monitor.csv"
    
    # 读取monitor.csv
    df = pd.read_csv(monitor_file, skiprows=1)  # 跳过第一行注释
    if 'r' not in df.columns:  # 确保有奖励列
        if len(df.columns) >= 2:
            df.columns = ['l', 't', 'r']  # length, timesteps, reward
    
    rewards = df['r'].values if 'r' in df.columns else df.iloc[:, 2].values
    
    # 计算滑动窗口方差
    variances = []
    mean_rewards = []
    timesteps = []
    current_timestep = 0
    
    for i in range(window_size, len(rewards)):
        window_rewards = rewards[i-window_size:i]
        variance = np.var(window_rewards)
        mean_reward = np.mean(window_rewards)
        
        variances.append(variance)
        mean_rewards.append(mean_reward)
        
        # 估算时间步（基于episode长度）
        if 'l' in df.columns:
            current_timestep += np.sum(df['l'].values[i-window_size:i])
        else:
            current_timestep += window_size * 200  # CartPole的近似长度
        
        timesteps.append(current_timestep)
    
    return {
        'timesteps': timesteps,
        'variances': variances,
        'mean_rewards': mean_rewards,
        'episode_rewards': rewards.tolist(),
        'window_size': window_size
    }

def analyze_multiple_runs(run_dirs, output_dir="variance_results"):
    """分析多个运行的方差模式"""
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    for run_dir in run_dirs:
        print(f"分析: {run_dir}")
        try:
            # 提取运行信息
            path_parts = Path(run_dir).parts
            env_name = path_parts[-3]  # runs/ENV/ALGO/RUN_ID
            algo_name = path_parts[-2]
            run_id = path_parts[-1]
            
            # 分析方差
            result = analyze_single_run_variance(run_dir)
            result['env'] = env_name
            result['algo'] = algo_name
            result['run_id'] = run_id
            
            key = f"{env_name}_{algo_name}_{run_id}"
            all_results[key] = result
            
            # 保存单个运行结果
            output_file = Path(output_dir) / f"{key}_variance.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
                
        except Exception as e:
            print(f"分析失败 {run_dir}: {e}")
    
    return all_results

def plot_variance_analysis(all_results, output_dir="variance_results"):
    """绘制方差分析图表"""
    
    # 按环境和算法分组
    grouped_results = defaultdict(list)
    for key, result in all_results.items():
        group_key = f"{result['env']}_{result['algo']}"
        grouped_results[group_key].append(result)
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 方差随时间变化
    ax1 = axes[0, 0]
    for group_key, results in grouped_results.items():
        for i, result in enumerate(results):
            if len(result['timesteps']) > 0:
                ax1.plot(result['timesteps'], result['variances'], 
                        alpha=0.7, label=f"{group_key}_{i}" if i < 3 else "")
    
    ax1.set_xlabel('Timesteps')
    ax1.set_ylabel('Variance (Sliding Window)')
    ax1.set_title('Episode Variance Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 平均奖励与方差关系
    ax2 = axes[0, 1]
    for group_key, results in grouped_results.items():
        for result in results:
            if len(result['variances']) > 0 and len(result['mean_rewards']) > 0:
                ax2.scatter(result['mean_rewards'], result['variances'], 
                           alpha=0.6, label=group_key)
    
    ax2.set_xlabel('Mean Reward')
    ax2.set_ylabel('Variance')
    ax2.set_title('Variance vs Mean Reward')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 方差收敛速度分析
    ax3 = axes[1, 0]
    convergence_data = []
    
    for group_key, results in grouped_results.items():
        for result in results:
            if len(result['variances']) > 10:
                # 计算方差降到初始值10%所需的时间
                initial_var = result['variances'][0]
                target_var = initial_var * 0.1
                
                convergence_step = None
                for i, var in enumerate(result['variances']):
                    if var <= target_var:
                        convergence_step = result['timesteps'][i] if i < len(result['timesteps']) else i
                        break
                
                if convergence_step:
                    convergence_data.append({
                        'group': group_key,
                        'convergence_step': convergence_step,
                        'final_variance': result['variances'][-1]
                    })
                    ax3.scatter(convergence_step, result['variances'][-1], 
                               label=group_key, alpha=0.7)
    
    ax3.set_xlabel('Convergence Timestep')
    ax3.set_ylabel('Final Variance')
    ax3.set_title('Variance Convergence Speed')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Episode奖励分布变化
    ax4 = axes[1, 1]
    for group_key, results in grouped_results.items():
        if results:
            result = results[0]  # 取第一个示例
            rewards = result['episode_rewards']
            
            # 分成几个阶段看分布变化
            quartile = len(rewards) // 4
            phases = {
                'First 25%': rewards[:quartile],
                '25%-50%': rewards[quartile:2*quartile],
                '50%-75%': rewards[2*quartile:3*quartile],
                'Last 25%': rewards[3*quartile:]
            }
            
            for phase_name, phase_rewards in phases.items():
                if phase_rewards:
                    ax4.hist(phase_rewards, alpha=0.5, label=f"{group_key} - {phase_name}")
    
    ax4.set_xlabel('Episode Reward')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Reward Distribution Evolution')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/variance_analysis_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return convergence_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", required=True, 
                       help="运行目录路径列表")
    parser.add_argument("--output", default="variance_results")
    args = parser.parse_args()
    
    # 分析所有运行
    results = analyze_multiple_runs(args.runs, args.output)
    
    # 绘制分析图表
    convergence_data = plot_variance_analysis(results, args.output)
    
    # 保存汇总统计
    summary = {
        'total_runs_analyzed': len(results),
        'convergence_analysis': convergence_data
    }
    
    with open(f'{args.output}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"分析完成！结果保存在: {args.output}")

#!/usr/bin/env python3
"""
Experimental Data Statistical Analysis Tool - Focused on Scanning and Analysis Functions
"""

import os
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional

class ExperimentAnalyzer:
    """Experimental Data Analyzer - Focused on Scanning and Analysis"""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.experiments = {}

    def scan_experiments(self) -> Dict:
        """Scan all experimental data"""
        if not self.base_path.exists():
            raise FileNotFoundError(f"Base path does not exist: {self.base_path}")

        print(f"Scanning experiment directory: {self.base_path}")

        # reset experiments dictionary so that multiple calls to scan_experiments work correctly
        self.experiments = {}

        env_count = 0
        algo_count = 0

        for env_dir in self.base_path.iterdir():
            if not env_dir.is_dir() or env_dir.name.startswith('.'):
                continue

            env_name = env_dir.name
            self.experiments[env_name] = {}
            env_count += 1

            for algo_dir in env_dir.iterdir():
                if not algo_dir.is_dir() or algo_dir.name.startswith('.'):
                    continue

                algo_name = algo_dir.name
                experiment_dirs = []
                algo_count += 1

                exp_count = 0
                for exp_dir in algo_dir.iterdir():
                    if exp_dir.is_dir() and not exp_dir.name.startswith('.'):
                        exp_info = self._parse_experiment_id(exp_dir.name)
                        if exp_info:
                            experiment_dirs.append({
                                'path': exp_dir,
                                'id': exp_dir.name,
                                **exp_info
                            })
                            exp_count += 1

                print(f"  {env_name}/{algo_name}: Found {exp_count} experiments")
                self.experiments[env_name][algo_name] = experiment_dirs

        total_experiments = self._get_total_experiments()
        print(f"\nScanning complete:")
        print(f"  Environments: {env_count}")
        print(f"  Algorithms: {algo_count}")
        print(f"  Total experiments: {total_experiments}")
        return self.experiments

    def _parse_experiment_id(self, exp_id: str) -> Optional[Dict]:
        """Parse experiment ID"""
        # Format: 20251026_035110_983255_pid796198_seed0
        pattern = r'(\d{8})_(\d{6})_(\d+)_pid(\d+)_seed(\d+)'
        match = re.match(pattern, exp_id)

        if match:
            date, time, timestamp, pid, seed = match.groups()
            return {
                'date': f"{date[:4]}-{date[4:6]}-{date[6:8]}",
                'time': f"{time[:2]}:{time[2:4]}:{time[4:6]}",
                'timestamp': timestamp,
                'pid': int(pid),
                'seed': int(seed)
            }
        return None

    def _get_total_experiments(self) -> int:
        """Get total number of experiments"""
        total = 0
        for env in self.experiments.values():
            for algo in env.values():
                total += len(algo)
        return total

    def get_statistics(self) -> Dict:
        """Get statistics"""
        if not self.experiments:
            self.scan_experiments()

        stats = {}
        for env, algos in self.experiments.items():
            stats[env] = {}
            for algo, experiments in algos.items():
                if not experiments:  # deal with empty algorithms
                    continue

                seeds = list(set(exp['seed'] for exp in experiments))
                latest_exp = max(experiments, key=lambda x: x['timestamp'])

                stats[env][algo] = {
                    'count': len(experiments),
                    'seeds': sorted(seeds),
                    'seed_count': len(seeds),
                    'latest_date': latest_exp['date'] if latest_exp else 'N/A'
                }
        return stats

    def get_experiments_by_seed(self, seed: int) -> List[Dict]:
        """Get experiments filtered by seed"""
        experiments = []
        for env, algos in self.experiments.items():
            for algo, exp_list in algos.items():
                for exp in exp_list:
                    if exp['seed'] == seed:
                        experiments.append({
                            'env': env,
                            'algo': algo,
                            **exp
                        })
        return experiments

    def compare_algorithms(self, env_name: str) -> Dict:
        """Compare experiments of different algorithms in the same environment"""
        if env_name not in self.experiments:
            return {}

        comparison = {}
        for algo, experiments in self.experiments[env_name].items():
            if not experiments:  # skip empty algorithms
                continue

            seeds = list(set(exp['seed'] for exp in experiments))
            comparison[algo] = {
                'experiment_count': len(experiments),
                'unique_seeds': len(seeds),
                'seed_list': sorted(seeds),
                'avg_experiments_per_seed': len(experiments) / len(seeds) if seeds else 0
            }

        return comparison


def print_statistics(stats: Dict):
    """Print statistics - concise table format"""
    if not stats:
        print("No experiment data found")
        return

    print("=" * 80)
    print("Experiment Statistics Summary")
    print("=" * 80)

    for env, algos in stats.items():
        print(f"\nEnvironment: {env}")
        print("-" * 60)
        print(f"{'Algorithm':<15} {'Experiment Count':<10} {'Seed Count':<10} {'Latest Experiment Date'}")
        print("-" * 60)

        if not algos:  # deal with empty algorithms
            print("No algorithms found")
            continue

        for algo, info in algos.items():
            print(f"{algo:<15} {info['count']:<10} {info['seed_count']:<10} {info['latest_date']}")


def print_comparison(comparison: Dict, env_name: str):
    """Print algorithm comparison results"""
    if not comparison:
        print(f"No algorithm data found for environment: {env_name}")
        return

    print(f"\nEnvironment {env_name} Algorithm Comparison")
    print("=" * 60)
    print(f"{'Algorithm':<15} {'Experiment Count':<8} {'Unique Seeds':<8} {'Avg. Experiments per Seed'}")
    print("-" * 60)

    for algo, info in comparison.items():
        avg_exp = info['avg_experiments_per_seed']
        print(f"{algo:<15} {info['experiment_count']:<8} {info['unique_seeds']:<8} {avg_exp:.1f}")


def main():
    """Main function - concise command line interface"""
    DEFAULT_BASE_PATH = "/homes/sohawan2/reinforcement-learning-thesis/thesis_project/runs"

    parser = argparse.ArgumentParser(
        description="Experiment data statistics analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "base_path",
        nargs='?',  # 0 or 1 argument
        default=DEFAULT_BASE_PATH,
        help=f"Experiment data root directory path (default: {DEFAULT_BASE_PATH})"
    )
    parser.add_argument("--env", "-e", help="Specify environment name")
    parser.add_argument("--algo", "-a", help="Specify algorithm name")
    parser.add_argument("--seed", "-s", type=int, help="Filter experiments by seed")
    parser.add_argument("--compare", "-c", action="store_true",
                       help="Compare different algorithms in the same environment")

    args = parser.parse_args()

    # More detailed path information
    print(f"Using path: {args.base_path}")
    if args.base_path == DEFAULT_BASE_PATH:
        print("(Using default path)")

    # Validate base path
    if not os.path.exists(args.base_path):
        print(f"Error: Path does not exist: {args.base_path}")
        print("Please provide a valid path or check the default path configuration")
        return

    if not os.path.isdir(args.base_path):
        print(f"Error: Path is not a directory: {args.base_path}")
        return

    try:
        analyzer = ExperimentAnalyzer(args.base_path)

        if args.seed is not None:
            # Filter experiments by seed
            experiments = analyzer.get_experiments_by_seed(args.seed)
            if not experiments:
                print(f"No experiments found with seed {args.seed}")
                return

            print(f"\nExperiments with seed {args.seed}:")
            print("-" * 50)
            for exp in experiments:
                print(f"{exp['env']}/{exp['algo']} - {exp['date']} {exp['time']}")

        elif args.compare:
            if not args.env:
                print("Error: --compare requires --env to be specified")
                return

            # Algorithm comparison mode
            comparison = analyzer.compare_algorithms(args.env)
            if comparison:
                print_comparison(comparison, args.env)
            else:
                print(f"No experiment data found for environment: {args.env}")

        elif args.env and args.algo:
            # Specific environment + algorithm statistics
            stats = analyzer.get_statistics()
            if args.env in stats and args.algo in stats[args.env]:
                env_stats = {args.env: {args.algo: stats[args.env][args.algo]}}
                print_statistics(env_stats)
            else:
                print(f"No data found for {args.env}/{args.algo}")

        elif args.env:
            # Specific environment statistics
            stats = analyzer.get_statistics()
            if args.env in stats:
                env_stats = {args.env: stats[args.env]}
                print_statistics(env_stats)
            else:
                print(f"No data found for environment: {args.env}")

        else:
            # Full statistics
            stats = analyzer.get_statistics()
            print_statistics(stats)

    except Exception as e:
        print(f"Execution error: {e}")
        import traceback
        traceback.print_exc()  # Show detailed error information for debugging


if __name__ == "__main__":
    main()

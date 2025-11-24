#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_pipeline.py

一键重建 max_reward 分析的所有结果：

1) 删除当前目录下的所有 .csv（只删结果，不删代码）
2) 运行 analyze_rmax_seed0.py
3) 运行 analyze_max_reward_configs.py
"""

from pathlib import Path
import subprocess


def run(cmd):
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    script_path = Path(__file__).resolve()
    workdir = script_path.parent
    print(f"[INFO] workdir = {workdir}")

    # 1) 清理旧结果（只删 csv，不动 py）
    for f in workdir.glob("*.csv"):
        print("[CLEAN]", f)
        f.unlink()

    # 2) 重跑分析
    run(["python", "analyze_rmax_seed0.py"])
    run(["python", "analyze_max_reward_configs.py"])


if __name__ == "__main__":
    main()

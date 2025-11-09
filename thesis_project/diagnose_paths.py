# thesis_project/diagnose_paths.py
#!/usr/bin/env python3
"""
diagnostic script - view current directory structure
"""
import os
from pathlib import Path

def diagnose_directory_structure():
    print("=== Directory Structure Diagnosis ===")

    # to get current working directory
    current_dir = Path.cwd()
    print(f"Current Working Directory: {current_dir}")

    # to get script directory
    script_dir = Path(__file__).parent
    print(f"Script Directory: {script_dir}")

    # to check if runs directory exists
    runs_path = Path("runs")
    print(f"\ncheck 'runs' directory:")
    print(f"absolute path: {runs_path} - exists: {runs_path.exists()}")

    # to check absolute path
    abs_runs_path = current_dir / "runs"
    print(f"absolute path: {abs_runs_path} - exists: {abs_runs_path.exists()}")

    # to list current directory contents
    print(f"\ncurrent directory contents:")
    for item in current_dir.iterdir():
        print(f"  {item.name}/'folder' if item.is_dir() else 'file'")

    # to check possible runs directory locations
    possible_locations = [
        current_dir / "runs",
        script_dir / "runs",
        script_dir.parent / "runs",  # upper level directory
        Path.home() / "reinforcement-learning-thesis" / "runs"
    ]

    print(f"\ncheck possible runs directory locations:")
    for location in possible_locations:
        if location.exists():
            print(f"✓ found: {location}")
            # display contents
            try:
                subdirs = [d.name for d in location.iterdir() if d.is_dir()]
                print(f"  include subdirectories: {subdirs[:5]}...")  # only show first 5
            except:
                print(f"  cannot read contents")

if __name__ == "__main__":
    diagnose_directory_structure()

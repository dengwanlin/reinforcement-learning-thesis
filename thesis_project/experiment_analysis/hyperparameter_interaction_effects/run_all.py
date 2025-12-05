# hyperparameter_interaction_effects/run_all.py

from .collect_runs import build_consolidated_table
from .interaction_analysis import run_interaction_pipeline


def main():
    print("=== Step 1: Building consolidated metrics table ===")
    build_consolidated_table()

    print("\n=== Step 2: Running interaction analysis pipeline ===")
    run_interaction_pipeline()

    print("\nAll done.")


if __name__ == "__main__":
    main()

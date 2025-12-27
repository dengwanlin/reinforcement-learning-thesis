#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract a per-run hyperparameter table from runs_seed0.

Traverses:
  ROOT_RUNS/<ENV>/<ALGO>/<RUN_ID>/config.yml

Writes:
  ANALYSIS_DIR/hp_table.csv

Notes:
- We try to be robust to nested YAML structures.
- We only store simple scalar/list values (numbers/strings/bools/lists).
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple
import json

import pandas as pd
import yaml

ROOT_RUNS = Path("/homes/sohawan2/reinforcement-learning-thesis/thesis_project/runs_seed0")
ANALYSIS_DIR = Path("/homes/sohawan2/reinforcement-learning-thesis/thesis_project/experiment_analysis/convergence_analysis")


def flatten_dict(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Flatten nested dict into key paths: e.g. {"a":{"b":1}} -> {"a.b":1}
    Lists are kept as JSON strings if they are not scalars.
    """
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key))
        elif isinstance(v, (list, tuple)):
            # store lists as JSON string for stable CSV
            out[key] = json.dumps(v)
        else:
            out[key] = v
    return out


def safe_load_yaml(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, dict):
        return {"_raw": str(obj)}
    return obj


def iter_runs(root: Path) -> Iterable[Tuple[str, str, Path]]:
    """
    Yield (env, algo, run_dir) for all runs.
    """
    for env_dir in sorted(root.iterdir()):
        if not env_dir.is_dir():
            continue
        env = env_dir.name
        for algo_dir in sorted(env_dir.iterdir()):
            if not algo_dir.is_dir():
                continue
            algo = algo_dir.name
            for run_dir in sorted(algo_dir.iterdir()):
                if run_dir.is_dir():
                    yield env, algo, run_dir


def parse_seed(run_id: str) -> int | None:
    if "seed" not in run_id:
        return None
    try:
        return int(run_id.split("seed")[-1])
    except Exception:
        return None


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    n_total = 0
    n_missing = 0

    for env, algo, run_dir in iter_runs(ROOT_RUNS):
        n_total += 1
        run_id = run_dir.name
        seed = parse_seed(run_id)

        cfg_path = run_dir / "config.yml"
        if not cfg_path.is_file():
            n_missing += 1
            continue

        cfg = safe_load_yaml(cfg_path)
        flat = flatten_dict(cfg)

        # add required identifiers
        flat["env"] = env
        flat["algo"] = algo
        flat["run_id"] = run_id
        flat["seed"] = seed

        rows.append(flat)

    df = pd.DataFrame(rows)

    out_path = ANALYSIS_DIR / "hp_table.csv"
    df.to_csv(out_path, index=False)

    print(f"Scanned runs: {n_total}")
    print(f"Missing config.yml: {n_missing}")
    print(f"Saved hp table to: {out_path}")
    print("Columns (sample):", df.columns[:30].tolist())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import csv
import itertools
import random
import argparse
from pathlib import Path

# -----------------------------
# 1) Base grid + env-specific
# -----------------------------
GRID_BASE = [
    "learning_rate:2.5e-4|learning_rate:3e-4|learning_rate:1e-4",
    "n_steps:1024|n_steps:2048|n_steps:4096",
    "batch_size:64|batch_size:128|batch_size:256",
    "n_epochs:10|n_epochs:20",
    "gamma:0.99|gamma:0.995",
    "clip_range:0.1|clip_range:0.2|clip_range:0.3",
    "gae_lambda:0.90|gae_lambda:0.95|gae_lambda:0.97",
    "ent_coef:0.0|ent_coef:0.01|ent_coef:0.02",
    "vf_coef:0.5",
    "max_grad_norm:0.5",
    "target_kl:0.01|target_kl:0.02",
    # keep last for readability
    "policy_kwargs:dict(net_arch=[64,64])|policy_kwargs:dict(net_arch=[128,128])|policy_kwargs:dict(net_arch=[256,256],ortho_init=True)",
]

def get_grid_for_env(env_id: str) -> list[str]:
    g = GRID_BASE.copy()
    if "CartPole" in env_id:
        g = replace_opt(g, "n_steps", "n_steps:128|n_steps:256|n_steps:512")
        g = replace_opt(g, "batch_size", "batch_size:32|batch_size:64")
        g = replace_opt(g, "learning_rate", "learning_rate:3e-4|learning_rate:1e-3|learning_rate:1e-4")
        g = replace_opt(g, "clip_range", "clip_range:0.2|clip_range:0.3")
        g = replace_opt(g, "gae_lambda", "gae_lambda:0.90|gae_lambda:0.95")
        g = replace_opt(g, "policy_kwargs", "policy_kwargs:dict(net_arch=[64,64])|policy_kwargs:dict(net_arch=[128,128])")
        g = replace_opt(g, "target_kl", "target_kl:0.02")
    elif "LunarLanderContinuous" in env_id:
        # continuous version of Lander
        g = replace_opt(g, "learning_rate", "learning_rate:3e-4|learning_rate:1e-4")
        g = replace_opt(g, "n_steps", "n_steps:1024|n_steps:2048")
        g = replace_opt(g, "batch_size", "batch_size:64|batch_size:128")
        g = replace_opt(g, "gae_lambda", "gae_lambda:0.95|gae_lambda:0.97")
        g = replace_opt(g, "ent_coef", "ent_coef:0.0|ent_coef:0.005|ent_coef:0.01")
        g.append("clip_range_vf:None|clip_range_vf:0.2")
        g.append("use_sde:True")
        g.append("sde_sample_freq:4|sde_sample_freq:64")
        g = replace_opt(g, "policy_kwargs",
                        "policy_kwargs:dict(net_arch=[256,256],ortho_init=True,log_std_init=-1.0)|"
                        "policy_kwargs:dict(net_arch=[128,128],ortho_init=True,log_std_init=-0.5)")
    elif "Hopper" in env_id:
        # longer continuous control
        g = replace_opt(g, "learning_rate", "learning_rate:3e-4|learning_rate:1e-4|learning_rate:5e-5")
        g = replace_opt(g, "n_steps", "n_steps:2048|n_steps:4096")
        g = replace_opt(g, "batch_size", "batch_size:128|batch_size:256")
        g.append("clip_range_vf:None|clip_range_vf:0.2")
        g.append("use_sde:True")
        g.append("sde_sample_freq:4|sde_sample_freq:64")
        g = replace_opt(g, "ent_coef", "ent_coef:0.0|ent_coef:0.005")
        g = replace_opt(g, "policy_kwargs",
                        "policy_kwargs:dict(net_arch=[256,256],ortho_init=True,log_std_init=-1.0)|"
                        "policy_kwargs:dict(net_arch=[128,128],ortho_init=True,log_std_init=-0.5)")
    elif "LunarLander" in env_id:
        # discrete Lander
        g = replace_opt(g, "learning_rate", "learning_rate:2.5e-4|learning_rate:3e-4|learning_rate:1e-4")
        g = replace_opt(g, "n_steps", "n_steps:1024|n_steps:2048")
        g = replace_opt(g, "batch_size", "batch_size:64|batch_size:128")
        g = replace_opt(g, "clip_range", "clip_range:0.1|clip_range:0.2")
        g = replace_opt(g, "gae_lambda", "gae_lambda:0.90|gae_lambda:0.95|gae_lambda:0.97")
        g = replace_opt(g, "ent_coef", "ent_coef:0.0|ent_coef:0.01|ent_coef:0.02")
        g = replace_opt(g, "policy_kwargs",
                        "policy_kwargs:dict(net_arch=[128,128])|policy_kwargs:dict(net_arch=[256,256],ortho_init=True)")
    return g

def replace_opt(grid: list[str], key: str, new_line: str) -> list[str]:
    out, found = [], False
    for line in grid:
        k = line.split(":", 1)[0]
        if k == key:
            out.append(new_line); found = True
        else:
            out.append(line)
    if not found:
        out.append(new_line)
    return out

# --------------------------------------
# 2) parse & expand grid
# --------------------------------------
def parse_grid_lines(lines: list[str]) -> dict[str, list[str]]:
    space: dict[str, list[str]] = {}
    for line in lines:
        for p in line.split("|"):
            k, v = p.split(":", 1)
            space.setdefault(k, []).append(v)
    return space

def expand(space: dict[str, list[str]]):
    keys = list(space.keys())
    vals = [space[k] for k in keys]
    for values in itertools.product(*vals):
        yield dict(zip(keys, values))

def ordered_keys(space: dict[str, list[str]]) -> list[str]:
    keys = list(space.keys())  # preserves insertion order
    if "policy_kwargs" in keys:
        keys = [k for k in keys if k != "policy_kwargs"] + ["policy_kwargs"]
    return keys

# --------------------------------------
# 3) filtering invalid combinations
# --------------------------------------
def is_valid(combo: dict[str, str], n_envs: int = 1, continuous: bool = False) -> bool:
    try:
        n_steps = int(combo.get("n_steps", "2048"))
        batch = int(combo.get("batch_size", "64"))
        n_epochs = int(combo.get("n_epochs", "10"))
        clip = float(combo.get("clip_range", "0.2"))
        lr = float(str(combo.get("learning_rate", "3e-4")))
    except ValueError:
        return False

    total_batch = n_steps * n_envs

    # a) rollout divisible by batch
    if batch <= 0 or total_batch % batch != 0:
        return False

    # b) SB3: batch_size divisible by n_envs
    if batch % n_envs != 0:
        return False

    # c) reasonable #minibatches
    n_minibatches = total_batch // batch
    if not (2 <= n_minibatches <= 64):
        return False

    # d) clip bounds
    if clip > 0.3 or clip <= 0.0:
        return False

    # e) n_epochs bounds
    if not (3 <= n_epochs <= 30):
        return False

    # f) control compute & overfitting
    if n_epochs * n_minibatches > 256:
        return False

    # g) continuous envs → require gSDE (soft policy; change if you want)
    if continuous and combo.get("use_sde", "False") != "True":
        return False

    # h) ortho init + too large lr tends to be unstable
    pk = combo.get("policy_kwargs", "")
    if "ortho_init=True" in pk and lr > 3e-4:
        return False

    # i) batch should not exceed total
    if batch > total_batch:
        return False

    # j) (optional) tight clip → small target_kl
    try:
        tkl = float(combo.get("target_kl", "0.02"))
        if clip <= 0.1 and tkl > 0.015:
            return False
    except ValueError:
        return False

    return True

# --------------------------------------
# 4) emit CSV and runner commands
# --------------------------------------
def to_hparam_list(combo: dict[str, str], keys_order: list[str]) -> list[str]:
    """['k1:v1', 'k2:v2', ...] in the declared order (policy_kwargs last)."""
    return [f"{k}:{combo[k]}" for k in keys_order if k in combo]

def _quote_if_needed(s: str) -> str:
    """
    If the argument contains characters that zsh/bash might misinterpret, it will be automatically wrapped in single quotes.
If single quotes contain single quotes within them (very rare), they will be escaped: ' → '"'"'
    """
    special = "[](){}=,|&;<>*$`\\\" "  # Common characters that need to be protected (including spaces)
    if any(ch in s for ch in special):
        return "'" + s.replace("'", "'\"'\"'") + "'"
    return s


def write_csv(path: Path, rows: list[dict[str, str]], keys_order: list[str]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys_order)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys_order})

def write_cmds(path: Path, rows: list[dict[str, str]], keys_order: list[str], runner: str, algo: str, env_id: str):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            parts = []
            for p in to_hparam_list(r, keys_order):
                # Automatically quote parameters containing brackets/square brackets/spaces/commas/equal signs, etc.
                parts.append(_quote_if_needed(p))
            hp = " ".join(parts)
            cmd = f"python {runner} --algo {algo} --env {env_id} --hyperparams {hp}"
            f.write(cmd + "\n")
def maybe_sample(valid: list[dict[str, str]], sample_k: int | None = None, seed: int = 0):
    if sample_k is None or sample_k >= len(valid):
        return valid
    rnd = random.Random(seed)
    rnd.shuffle(valid)
    return valid[:sample_k]

def make_for_env(env_id: str, algo: str, runner_py: str, out_dir: Path, n_envs: int = 1, sample_k: int | None = None, seed: int = 0):
    grid_lines = get_grid_for_env(env_id)
    space = parse_grid_lines(grid_lines)
    keys_order = ordered_keys(space)
    continuous = ("Continuous" in env_id) or ("Hopper" in env_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    combos = list(expand(space))
    valid = [c for c in combos if is_valid(c, n_envs=n_envs, continuous=continuous)]
    picked = maybe_sample(valid, sample_k=sample_k, seed=seed)

    csv_path = out_dir / f"{env_id}_{algo}_grid.csv"
    txt_path = out_dir / f"{env_id}_{algo}_cmds.txt"

    write_csv(csv_path, picked if picked else valid, keys_order)
    write_cmds(txt_path, picked if picked else valid, keys_order, runner_py, algo, env_id)

    print(f"[{env_id}] total: {len(combos):,} | valid: {len(valid):,} | written: {len(picked if picked else valid):,} | csv: {csv_path} | cmds: {txt_path}")

# --------------------------------------
# 5) CLI
# --------------------------------------
def main():
    ap = argparse.ArgumentParser("Make PPO grid and runner commands")
    ap.add_argument("--runner", type=str, default="runner.py", help="Path to your training runner (python script).")
    ap.add_argument("--algo", type=str, default="ppo", choices=["ppo"], help="Algorithm to generate grid for.")
    ap.add_argument("--envs", nargs="+", default=["LunarLander-v3","CartPole-v1","Hopper-v4","LunarLanderContinuous-v3"], help="List of env ids.")
    ap.add_argument("--n-envs", type=int, default=1, help="VecEnv count for divisibility checks.")
    ap.add_argument("--sample", type=int, default=None, help="If set, randomly sample K valid combos per env.")
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    ap.add_argument("--out", type=str, default=None, help="Output folder (default: ./grids next to this script).")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out) if args.out else (script_dir / "grids")

    for env_id in args.envs:
        make_for_env(env_id=env_id,
                     algo=args.algo,
                     runner_py=args.runner,
                     out_dir=out_dir,
                     n_envs=args.n_envs,
                     sample_k=args.sample,
                     seed=args.seed)

if __name__ == "__main__":
    main()



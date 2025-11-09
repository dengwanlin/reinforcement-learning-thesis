#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import csv
import itertools
import random
import argparse
from pathlib import Path

# =========================
# 1) A2C Base Grid + Environment Clipping
# =========================
GRID_BASE_A2C = [
    # Common range for A2C (SB3 experience)
    "learning_rate:7e-4|learning_rate:3e-4|learning_rate:1e-3|learning_rate:1e-4",
    "n_steps:64|n_steps:128|n_steps:256|n_steps:512",
    "gamma:0.99|gamma:0.995",
    "gae_lambda:0.90|gae_lambda:0.95|gae_lambda:0.97",
    "ent_coef:0.0|ent_coef:0.001|ent_coef:0.01",
    "vf_coef:0.5",
    "max_grad_norm:0.5",
    # A2C defaults to RMSprop in SB3 (can also switch to Adam)
    "use_rms_prop:True|use_rms_prop:False",
    "rms_prop_eps:1e-5|rms_prop_eps:1e-4",
    # policy kwargs (keep last)
    "policy_kwargs:dict(net_arch=[64,64])|policy_kwargs:dict(net_arch=[128,128])|policy_kwargs:dict(net_arch=[256,256],ortho_init=True)",
]

def get_grid_for_env_a2c(env_id: str) -> list[str]:
    """Trim/append A2C search space based on environment."""
    g = GRID_BASE_A2C.copy()

    if "CartPole" in env_id:
        # Short round + small network, small step size, lr can be more aggressive
        g = replace_opt(g, "n_steps", "n_steps:64|n_steps:128|n_steps:256")
        g = replace_opt(g, "learning_rate", "learning_rate:1e-3|learning_rate:7e-4|learning_rate:3e-4|learning_rate:1e-4")
        g = replace_opt(g, "gae_lambda", "gae_lambda:0.90|gae_lambda:0.95")
        g = replace_opt(g, "ent_coef", "ent_coef:0.0|ent_coef:0.001")
        g = replace_opt(g, "policy_kwargs", "policy_kwargs:dict(net_arch=[64,64])|policy_kwargs:dict(net_arch=[128,128])")

    elif "LunarLanderContinuous" in env_id:
        # Continuous: gSDE is recommended; lr is slightly conservative
        g = replace_opt(g, "learning_rate", "learning_rate:7e-4|learning_rate:3e-4|learning_rate:1e-4")
        g = replace_opt(g, "n_steps", "n_steps:64|n_steps:128|n_steps:256|n_steps:512")
        g = replace_opt(g, "ent_coef", "ent_coef:0.0|ent_coef:0.001|ent_coef:0.01")
        g.append("use_sde:True")
        g.append("sde_sample_freq:4|sde_sample_freq:64")
        g = replace_opt(g, "policy_kwargs",
                        "policy_kwargs:dict(net_arch=[256,256],ortho_init=True)|"
                        "policy_kwargs:dict(net_arch=[128,128],ortho_init=True)")

    elif "Hopper" in env_id:
        # MuJoCo continuous + long episode: rollout can be slightly longer
        g = replace_opt(g, "learning_rate", "learning_rate:7e-4|learning_rate:3e-4|learning_rate:1e-4")
        g = replace_opt(g, "n_steps", "n_steps:128|n_steps:256|n_steps:512|n_steps:1024")
        g.append("use_sde:True")
        g.append("sde_sample_freq:4|sde_sample_freq:64")
        g = replace_opt(g, "ent_coef", "ent_coef:0.0|ent_coef:0.001")
        g = replace_opt(g, "policy_kwargs",
                        "policy_kwargs:dict(net_arch=[256,256],ortho_init=True)|"
                        "policy_kwargs:dict(net_arch=[128,128],ortho_init=True)")

    elif "LunarLander" in env_id:
        # Discrete Lander: medium rollout, slightly higher lambda
        g = replace_opt(g, "learning_rate", "learning_rate:7e-4|learning_rate:3e-4|learning_rate:1e-4")
        g = replace_opt(g, "n_steps", "n_steps:64|n_steps:128|n_steps:256")
        g = replace_opt(g, "gae_lambda", "gae_lambda:0.90|gae_lambda:0.95|gae_lambda:0.97")
        g = replace_opt(g, "ent_coef", "ent_coef:0.0|ent_coef:0.001|ent_coef:0.01")
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

# ======================
# 2) Parsing & Cartesian Expansion
# ======================
def parse_grid_lines(lines: list[str]) -> dict[str, list[str]]:
    space: dict[str, list[str]] = {}
    for line in lines:
        for p in line.split("|"):
            k, v = p.split(":", 1)
            space.setdefault(k, []).append(v)
    return space

def expand(space: dict[str, list[str]]):
    keys = list(space.keys())
    for values in itertools.product(*[space[k] for k in keys]):
        yield dict(zip(keys, values))

def ordered_keys(space: dict[str, list[str]]) -> list[str]:
    keys = list(space.keys())  # Preserve parsing order
    if "policy_kwargs" in keys:
        keys = [k for k in keys if k != "policy_kwargs"] + ["policy_kwargs"]
    return keys

# ======================
# 3) A2C filtering rules (close to SB3 usage)
# ======================
def is_valid_a2c(combo: dict[str, str], n_envs: int = 1, continuous: bool = False) -> bool:
    """
A2C sanity checks.
Unlike PPO, A2C does not have a batch_size; instead, the update scale is controlled by n_steps*n_envs.
    """
    try:
        n_steps = int(combo.get("n_steps", "128"))
        lr = float(str(combo.get("learning_rate", "7e-4")))
        gamma = float(combo.get("gamma", "0.99"))
        lam = float(combo.get("gae_lambda", "0.95"))
    except ValueError:
        return False

    # a) Rollout scale: too small will result in high gradient noise, too large will result in slow training (can be adjusted based on hardware)
    total_batch = n_steps * n_envs
    if not (64 <= total_batch <= 8192):
        return False

    # b) gamma reasonable range
    if not (0.95 <= gamma <= 0.999):
        return False

    # c) lambda reasonable range
    if not (0.85 <= lam <= 0.99):
        return False

    # d) Don’t use extreme learning rates
    if not (1e-5 <= lr <= 5e-3):
        return False

    # e) Continuous control: requires gSDE (relax if you want)
    if continuous and combo.get("use_sde", "False") != "True":
        return False

    # f) Orthogonal initialization + excessive learning rate instability (similar to PPO)
    pk = combo.get("policy_kwargs", "")
    if "ortho_init=True" in pk and lr > 7e-4:
        return False

    return True

# ======================
# 4) output: CSV & CMDs
# ======================
def to_hparam_list(combo: dict[str, str], keys_order: list[str]) -> list[str]:
    return [f"{k}:{combo[k]}" for k in keys_order if k in combo]
def _quote_if_needed(s: str) -> str:
    special = "[](){}=,|&;<>*$`\\\" "  # include normal space and special chars
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

def make_for_env_a2c(env_id: str, runner_py: str, out_dir: Path, n_envs: int = 1, sample_k: int | None = None, seed: int = 0):
    grid_lines = get_grid_for_env_a2c(env_id)
    space = parse_grid_lines(grid_lines)
    keys_order = ordered_keys(space)
    continuous = ("Continuous" in env_id) or ("Hopper" in env_id)

    out_dir.mkdir(parents=True, exist_ok=True)
    combos = list(expand(space))
    valid = [c for c in combos if is_valid_a2c(c, n_envs=n_envs, continuous=continuous)]
    picked = maybe_sample(valid, sample_k=sample_k, seed=seed)

    csv_path = out_dir / f"{env_id}_a2c_grid.csv"
    txt_path = out_dir / f"{env_id}_a2c_cmds.txt"

    rows = picked if picked else valid
    write_csv(csv_path, rows, keys_order)
    write_cmds(txt_path, rows, keys_order, runner_py, "a2c", env_id)

    print(f"[{env_id}] total: {len(combos):,} | valid: {len(valid):,} | written: {len(rows):,} | csv: {csv_path} | cmds: {txt_path}")

# ======================
# 5) CLI
# ======================
def main():
    ap = argparse.ArgumentParser("Make A2C grid and runner commands")
    ap.add_argument("--runner", type=str, default="runner.py", help="Path to your training runner (python script).")
    ap.add_argument("--envs", nargs="+", default=["LunarLander-v3","CartPole-v1","Hopper-v4","LunarLanderContinuous-v3"], help="List of env ids.")
    ap.add_argument("--n-envs", type=int, default=1, help="VecEnv count for rollout size checks.")
    ap.add_argument("--sample", type=int, default=None, help="If set, randomly sample K valid combos per env.")
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    ap.add_argument("--out", type=str, default=None, help="Output folder (default: ./grids next to this script).")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.out) if args.out else (script_dir / "grids")

    for env_id in args.envs:
        make_for_env_a2c(env_id=env_id,
                         runner_py=args.runner,
                         out_dir=out_dir,
                         n_envs=args.n_envs,
                         sample_k=args.sample,
                         seed=args.seed)

if __name__ == "__main__":
    main()

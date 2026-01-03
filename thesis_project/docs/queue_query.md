# Queue Runner Usage Guide

This document describes how to initialise, run, monitor, recover, and export
large-scale reinforcement learning experiments using the SQLite-based queue
system.

All commands assume execution on the RTX3 server.

---

## Environment Setup

```bash
conda activate rl
cd /homes/sohawan2/reinforcement-learning-thesis/thesis_project
```

## Random Seed Sets

The following random seeds are used for independent experimental replications:

```

SEED_SET = [
    0,
    704453266,
    1106730433,
    1374529905,
    1826345644,
    1978735955,
]
```

Each seed corresponds to a separate queue database (queue_seed<seed>.db).

## 1. Import Commands into Queue Databases

Initialise queue databases from pre-generated command lists.

Example (single seed)

```

python init_from_txt.py \
  --cmd-file grids/CartPole-v1_a2c_cmds.txt \
  --db queue_seed1106730433.db \
  --seed 1106730433
```

Full setup for seed 1106730433

```

python init_from_txt.py --cmd-file grids/CartPole-v1_a2c_cmds.txt --db queue_seed1106730433.db --seed 1106730433
python init_from_txt.py --cmd-file grids/CartPole-v1_ppo_cmds.txt --db queue_seed1106730433.db --seed 1106730433

python init_from_txt.py --cmd-file grids/LunarLander-v3_a2c_cmds.txt --db queue_seed1106730433.db --seed 1106730433
python init_from_txt.py --cmd-file grids/LunarLander-v3_ppo_cmds.txt --db queue_seed1106730433.db --seed 1106730433

python init_from_txt.py --cmd-file grids/LunarLanderContinuous-v3_a2c_cmds.txt --db queue_seed1106730433.db --seed 1106730433
python init_from_txt.py --cmd-file grids/LunarLanderContinuous-v3_ppo_cmds.txt --db queue_seed1106730433.db --seed 1106730433

python init_from_txt.py --cmd-file grids/Hopper-v4_a2c_cmds.txt --db queue_seed1106730433.db --seed 1106730433
python init_from_txt.py --cmd-file grids/Hopper-v4_ppo_cmds.txt --db queue_seed1106730433.db --seed 1106730433
```

## 2. Start Multi-process Execution (Local / Light Load)

For small-scale or local execution:
```
python queue_runner.py run \
  --db queue_seed1106730433.db \
  --workdir . \
  --workers 8 \
  --timeout 7200
```

### 3. Monitor Queue Status

Check progress at any time using:

```
date
python queue_runner.py status --db queue_seed1106730433.db
```

Examples for other seeds:

```
python queue_runner.py status --db queue_seed0.db
python queue_runner.py status --db queue_seed704453266.db
```

### 4. Recover Interrupted or Stuck Jobs

If execution is interrupted (e.g. process killed or node rebooted), recover stale
running jobs back into the queue:

```

python queue_runner.py recover \
  --db queue_seed1106730433.db \
  --stale-seconds 10800
```

This operation is idempotent and safe to run multiple times.

### 5. Export Queue State

Export all commands with updated status flags:

```

python queue_runner.py export \
  --db queue_seed1106730433.db \
  --out grids/cmds_marked_seed1106730433.txt
```

### 6. Backup Queue Databases

Regular backups are recommended:

```

cp queue_seed1106730433.db grids/queue_backup_seed1106730433_$(date +%Y%m%d).db
```

### 7. Running on RTX3 (Shared Server, CPU-bound Jobs)

To ensure fair usage on the shared RTX3 server:

Limit numerical libraries to 1 thread per process

Use nice to lower scheduling priority

Restrict total workers to ≈ 70% of available CPU cores

Recommended configuration (21 workers)

```

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

nohup python queue_runner.py run \
  --db queue_seed1106730433.db \
  --workdir /homes/sohawan2/reinforcement-learning-thesis/thesis_project \
  --workers 20\
  --timeout 8400 \
  --logdir logs/cpu20 \
  > run_cpu21_$(date +%F_%H%M).log 2>&1 &

  ```

### end

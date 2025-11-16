#!/usr/bin/env bash
# runner.sh — simple & robust parallel launcher (no wait -n, fixed IFS & arg-splitting)

set -Eeuo pipefail
set -o noglob
trap 'echo; echo "[runner] Interrupted. Cleaning up…"; jobs -pr | xargs -r kill; exit 130' INT

# ---------- Paths ----------
PYTHON=${PYTHON:-/homes/sohawan2/miniconda3/envs/rl/bin/python}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/runner.py"
LOG_DIR="${SCRIPT_DIR}/logs"
[[ -f "$SCRIPT" ]] || { echo "[runner] ERROR: $SCRIPT not found"; exit 1; }
mkdir -p "$LOG_DIR"

# ---------- Config ----------
ENV_ID=${ENV_ID:-CartPole-v1}
ALGO=${ALGO:-ppo}
N_TIMESTEPS=${N_TIMESTEPS:-5e5}
DEVICE=${DEVICE:-auto}
PROGRESS=${PROGRESS:---progress}
NORMALIZE=${NORMALIZE:-}
EVAL_FREQ=${EVAL_FREQ:-20000}   # "" to disable
EVAL_EPS=${EVAL_EPS:-10}
SAVE_FREQ=${SAVE_FREQ:-100000}  # "" to disable
MAX_JOBS=${MAX_JOBS:-4}

# CPU pinning (optional)
BIND_CPUS=${BIND_CPUS:-0}       # 1=on 0=off
CORES_PER_JOB=${CORES_PER_JOB:-8}
NPROC=$(nproc || echo 32)
if (( BIND_CPUS == 1 )) && ! command -v taskset >/dev/null 2>&1; then
  echo "[runner] WARN: taskset not found; disabling CPU pinning."
  BIND_CPUS=0
fi

# Seeds & Grid
SEEDS=(0 1 2)
GRID=(
  "learning_rate:3e-4|learning_rate:1e-4|learning_rate:linear_schedule(3e-4)|learning_rate:linear_schedule(1e-4)"
  "n_steps:512|n_steps:1024|n_steps:2048|n_steps:4096"
  "batch_size:32|batch_size:64|batch_size:128|batch_size:256"
  "clip_range:0.1|clip_range:0.2|clip_range:0.3|clip_range:0.4"
  "gae_lambda:0.85|gae_lambda:0.9|gae_lambda:0.95|gae_lambda:0.99"
  "ent_coef:0.0|ent_coef:0.01|ent_coef:0.02|ent_coef:0.05"
  "policy_kwargs:dict(net_arch=[64,64])|policy_kwargs:dict(net_arch=[128,128])|policy_kwargs:dict(net_arch=[256,256])|policy_kwargs:dict(net_arch=[512,512])"
)

# FAST smoke test
FAST=${FAST:-1}
if (( FAST == 1 )); then
  echo "[runner] FAST=1 (smoke test) — cutting grid & jobs"
  GRID=(
    "learning_rate:3e-4|learning_rate:1e-4"
    "n_steps:1024|n_steps:2048"
    "batch_size:64|batch_size:128"
    "clip_range:0.2"
    "gae_lambda:0.95"
    "ent_coef:0.0|ent_coef:0.02"
    "policy_kwargs:dict(net_arch=[128,128])"
  )
  SEEDS=(0)
fi

echo "[runner] ENV=${ENV_ID}  ALGO=${ALGO}  N=${N_TIMESTEPS}  SEEDS=${SEEDS[*]}"
echo "[runner] DEVICE=${DEVICE}  PROGRESS=${PROGRESS:-<off>}  NORMALIZE=${NORMALIZE:-<off>}"
[[ -n "${EVAL_FREQ}" ]] && echo "[runner] EVAL: freq=${EVAL_FREQ}, episodes=${EVAL_EPS}" || echo "[runner] EVAL: <off>"
[[ -n "${SAVE_FREQ}" ]] && echo "[runner] CKPT: freq=${SAVE_FREQ}" || echo "[runner] CKPT: <off>"
echo "[runner] MAX_JOBS=${MAX_JOBS}  CPU_PINNING=${BIND_CPUS} (cores/job=${CORES_PER_JOB})"

# ---------- Build Cartesian product (restore IFS after each split) ----------
combos=("")
for group in "${GRID[@]}"; do
  OLDIFS=$IFS
  IFS='|' read -r -a opts <<< "$group"
  IFS=$OLDIFS
  new_combos=()
  for base in "${combos[@]}"; do
    for opt in "${opts[@]}"; do
      if [[ -z "$base" ]]; then new_combos+=("$opt"); else new_combos+=("$base $opt"); fi
    done
  done
  combos=("${new_combos[@]}")
done
NUM_COMBOS=${#combos[@]}
TOTAL=$(( ${#SEEDS[@]} * NUM_COMBOS ))
echo "[runner] Will launch ${NUM_COMBOS} hyperparam combos × ${#SEEDS[@]} seeds = ${TOTAL} jobs"

cpu_set_for_idx () {
  local idx="$1"
  local groups=$(( NPROC / CORES_PER_JOB )); (( groups == 0 )) && groups=1
  local g=$(( idx % groups ))
  local start=$(( g * CORES_PER_JOB ))
  local end=$(( start + CORES_PER_JOB - 1 ))
  (( end >= NPROC )) && end=$(( NPROC - 1 ))
  echo "${start}-${end}"
}

launch_one () {
  local seed="$1"
  local combo="$2"
  local tag="s${seed}__$(echo "$combo" | tr ' :[]=(),' '__' | tr -s '_' | cut -c1-200)"
  local log="${LOG_DIR}/${ENV_ID}_${ALGO}_${tag}.log"
  mkdir -p "$(dirname "$log")"

  # 安全拆分 hyperparams（不依赖外部 IFS）
  read -r -a HP_ARR <<< "${combo}"

  local eval_args=(); [[ -n "${EVAL_FREQ}" ]] && eval_args+=( --eval-freq "${EVAL_FREQ}" --eval-episodes "${EVAL_EPS}" )
  local ckpt_args=(); [[ -n "${SAVE_FREQ}" ]] && ckpt_args+=( --save-freq "${SAVE_FREQ}" )

  local CPU_RANGE="all"
  if (( BIND_CPUS == 1 )); then CPU_RANGE="$(cpu_set_for_idx "${job_idx}")"; fi

  echo "[runner] start[#${job_idx}]: seed=${seed} | CPU=${CPU_RANGE} | ${combo}"

  (
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTORCH_JIT=0
    if (( BIND_CPUS == 1 )); then
      taskset -c "${CPU_RANGE}" \
      "${PYTHON}" "${SCRIPT}" \
        --algo "${ALGO}" --env "${ENV_ID}" -n "${N_TIMESTEPS}" \
        --seed "${seed}" --device "${DEVICE}" ${PROGRESS} ${NORMALIZE} \
        "${eval_args[@]}" "${ckpt_args[@]}" \
        --hyperparams "${HP_ARR[@]}"
    else
      "${PYTHON}" "${SCRIPT}" \
        --algo "${ALGO}" --env "${ENV_ID}" -n "${N_TIMESTEPS}" \
        --seed "${seed}" --device "${DEVICE}" ${PROGRESS} ${NORMALIZE} \
        "${eval_args[@]}" "${ckpt_args[@]}" \
        --hyperparams "${HP_ARR[@]}"
    fi
  ) >"$log" 2>&1 &

  echo "[runner] spawned pid=$! -> $log"
}

# ---------- Launch with simple throttle ----------
job_idx=0
for combo in "${combos[@]}"; do
  for seed in "${SEEDS[@]}"; do
    # 限流：直到后台活跃作业数 < MAX_JOBS
    while (( $(jobs -pr | wc -l) >= MAX_JOBS )); do
      sleep 1
    done
    launch_one "${seed}" "${combo}"
    ((job_idx++))
  done
done

# 等所有后台作业结束
wait
echo "[runner] All ${job_idx} jobs finished."

# ---------- Summary (CSV) ----------
echo "[runner] Summarizing results…"
RUNS_DIR="${SCRIPT_DIR}/runs/${ENV_ID}/${ALGO}"
SUMMARY_PATH="${LOG_DIR}/summary_${ENV_ID}_${ALGO}_$(date +%Y%m%d_%H%M%S).csv"

export ENV_ID_EXPORT="${ENV_ID}" ALGO_EXPORT="${ALGO}" RUNS_DIR_EXPORT="${RUNS_DIR}" SUMMARY_PATH_EXPORT="${SUMMARY_PATH}"

"${PYTHON}" - <<'PY'
import os, json, csv, sys
from pathlib import Path
try:
  import yaml
except Exception:
  print("[summary] Missing PyYAML. Install with: pip install pyyaml", file=sys.stderr)
  sys.exit(1)

env=os.environ.get("ENV_ID_EXPORT",""); algo=os.environ.get("ALGO_EXPORT","")
runs_dir=os.environ.get("RUNS_DIR_EXPORT",""); summary_path=os.environ.get("SUMMARY_PATH_EXPORT","")
rows=[]
if not runs_dir or not Path(runs_dir).exists():
  print(f"[summary] Runs directory not found: {runs_dir}", file=sys.stderr)
else:
  for root,dirs,files in os.walk(runs_dir):
    files=set(files)
    if "results.json" in files and "config.yml" in files:
      run=Path(root)
      try: res=json.load(open(run/"results.json"))
      except Exception: continue
      try: cfg=dict(yaml.safe_load(open(run/"config.yml")) or {})
      except Exception: cfg={}
      def flatten(d):
        if not isinstance(d,dict): return str(d)
        parts=[]
        for k,v in d.items():
          if isinstance(v,(dict,list)):
            import json as _j; parts.append(f"{k}={_j.dumps(v,ensure_ascii=False)}")
          else:
            parts.append(f"{k}={v}")
        return "; ".join(parts)
      rows.append({
        "env":env,"algo":algo,"timestamp":cfg.get("timestamp"),"seed":cfg.get("seed"),
        "n_timesteps":cfg.get("n_timesteps"),"device":cfg.get("device"),"normalize":cfg.get("normalize"),
        "eval_freq":cfg.get("eval_freq"),"eval_episodes":cfg.get("eval_episodes"),
        "mean_reward":res.get("mean_reward"),"std_reward":res.get("std_reward"),
        "hyperparams_parsed":flatten(cfg.get("hyperparams_parsed",{})),
        "hyperparams_raw":" ".join(cfg.get("hyperparams_raw",[])) if isinstance(cfg.get("hyperparams_raw",[]),list) else str(cfg.get("hyperparams_raw","")),
        "run_dir":str(run)
      })
if rows:
  rows.sort(key=lambda r: (r.get("mean_reward") is not None, r.get("mean_reward",-1e9)), reverse=True)
  fields=["env","algo","timestamp","seed","n_timesteps","device","normalize","eval_freq","eval_episodes","mean_reward","std_reward","hyperparams_parsed","hyperparams_raw","run_dir"]
  Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
  with open(summary_path,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for r in rows: w.writerow({k:r.get(k) for k in fields})
  print(f"[summary] Wrote {summary_path} with {len(rows)} rows.")
else:
  print("[summary] No results found (results.json + config.yml).")
PY

echo "[runner] Summary saved to: ${SUMMARY_PATH}"
set +o noglob

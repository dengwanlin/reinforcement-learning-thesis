#!/usr/bin/env bash
# run_rl.sh – launch rl_runner.py (non‑interactive) and write logs to $OUTDIR


# 1.  Default environment / algorithm (change them here if you like)

ENV_VAL="Hopper-v4"   # default env
ALG_VAL="PPO"              # default algo


# 2.  Allow the user to override the defaults on the command line

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)   ENV_VAL="$2"; shift 2 ;;
        --algo)  ALG_VAL="$2"; shift 2 ;;
        *)       ARGS+=("$1"); shift ;;   # everything else is passed through
    esac
done


# 3.  Build the log‑file name (env_algo_YYYYMMDD-HHMMSS.log)

OUTDIR="/homes/sohawan2/reinforcement-learning-thesis/code_practice/out"
mkdir -p "$OUTDIR"

TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOGFILE="${OUTDIR}/${ENV_VAL}_${ALG_VAL}_${TIMESTAMP}.log"

echo "[run_rl] LOGFILE = $LOGFILE"
#echo "[run_rl] ENV    = $ENV_VAL"
#echo "[run_rl] ALGO   = $ALG_VAL"


# 4.  Launch the trainer with nohup, forcing non‑interactive mode

nohup python /homes/sohawan2/reinforcement-learning-thesis/code_practice/rl_runner(old).py \
    --env "$ENV_VAL" \
    --algo "$ALG_VAL" \
    --no-interactive \
    "${ARGS[@]}" \
    > "$LOGFILE" 2>&1 < /dev/null &
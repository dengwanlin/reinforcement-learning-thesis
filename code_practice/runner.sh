#!/usr/bin/env bash
# run_rl.sh – launch runner.py (non-interactive) and save logs

# 1. Defaults
ENV_VAL="Hopper-v4"
ALG_VAL="A2C"
ARGS=()

# 2. Parse CLI args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)   ENV_VAL="$2"; shift 2 ;;
        --algo)  ALG_VAL="$2"; shift 2 ;;
        *)       ARGS+=("$1"); shift ;;
    esac
done

# 3. Log path
OUTDIR="/homes/sohawan2/reinforcement-learning-thesis/code_practice/out"
mkdir -p "$OUTDIR"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOGFILE="${OUTDIR}/${ENV_VAL}_${ALG_VAL}_${TIMESTAMP}.log"

printf "[run_rl] Starting %s | %s\n" "$ENV_VAL" "$ALG_VAL"
printf "[run_rl] Logfile: %s\n" "$LOGFILE"

# 4. Run
nohup python /homes/sohawan2/reinforcement-learning-thesis/code_practice/runner.py \
    --env "$ENV_VAL" \
    --algo "$ALG_VAL" \
    "${ARGS[@]}" \
    > "$LOGFILE" 2>&1 < /dev/null &

echo "[run_rl] PID = $!"

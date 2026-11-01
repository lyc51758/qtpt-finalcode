#!/bin/bash
# AntMaze Parallel Experiment: run up to 4 jobs simultaneously

set -e
# Activate your Python environment before running this script if needed.
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
if [ -n "$MUJOCO_PATH" ]; then export LD_LIBRARY_PATH="$MUJOCO_PATH/bin:$LD_LIBRARY_PATH"; fi
export D4RL_SUPPRESS_IMPORT_ERROR=1

cd "$(dirname "$0")"

ENVS="antmaze-umaze-v2 antmaze-umaze-diverse-v2"
METHODS="qtpt spt qsft qdt"
SEEDS="0 1 2"
EPOCHS=50
OUTPUT_DIR=results_antmaze
MAX_PARALLEL=4

run_job() {
    local method=$1 env=$2 seed=$3
    local result_dir="${OUTPUT_DIR}/${method}_${env}_seed${seed}"
    
    if [ -f "${result_dir}/log.csv" ]; then
        lines=$(wc -l < "${result_dir}/log.csv")
        if [ "$lines" -gt "$EPOCHS" ]; then
            echo "[SKIP] ${method} | ${env} | seed=${seed} (${lines} lines)"
            return 0
        fi
    fi
    
    echo "[START] ${method} | ${env} | seed=${seed} | $(date)"
    python3.10 train_antmaze_multitask.py \
        --method $method \
        --env $env \
        --context_len 20 \
        --num_epochs $EPOCHS \
        --batch_size 128 \
        --lr 3e-4 \
        --n_embd 128 \
        --n_layer 4 \
        --n_head 4 \
        --dropout 0.1 \
        --eval_freq 5 \
        --eval_episodes 20 \
        --seed $seed \
        --output_dir $OUTPUT_DIR \
        > "log_${method}_${env}_seed${seed}.txt" 2>&1
    echo "[DONE] ${method} | ${env} | seed=${seed} | $(date)"
}

# Collect all jobs
jobs_list=()
for ENV in $ENVS; do
    for METHOD in $METHODS; do
        for SEED in $SEEDS; do
            jobs_list+=("${METHOD}|${ENV}|${SEED}")
        done
    done
done

echo "Total jobs: ${#jobs_list[@]}"
echo "Max parallel: $MAX_PARALLEL"
echo "Start: $(date)"

# Run in batches of MAX_PARALLEL
running=0
for job in "${jobs_list[@]}"; do
    IFS="|" read -r method env seed <<< "$job"
    
    # Check skip
    result_dir="${OUTPUT_DIR}/${method}_${env}_seed${seed}"
    if [ -f "${result_dir}/log.csv" ]; then
        lines=$(wc -l < "${result_dir}/log.csv")
        if [ "$lines" -gt "$EPOCHS" ]; then
            echo "[SKIP] ${method} | ${env} | seed=${seed}"
            continue
        fi
    fi
    
    run_job "$method" "$env" "$seed" &
    running=$((running + 1))
    
    if [ "$running" -ge "$MAX_PARALLEL" ]; then
        wait -n 2>/dev/null || wait
        running=$((running - 1))
    fi
done

wait
echo "ALL COMPLETE: $(date)"

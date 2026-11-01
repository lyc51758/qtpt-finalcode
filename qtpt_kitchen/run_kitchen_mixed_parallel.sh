#!/bin/bash
# Kitchen mixed-v0: 5 methods x 3 seeds = 15 runs, 3 parallel

set -e
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
if [ -n "$MUJOCO_PATH" ]; then export LD_LIBRARY_PATH="$MUJOCO_PATH/bin:$LD_LIBRARY_PATH"; fi
# Activate your Python environment before running this script if needed.
export D4RL_SUPPRESS_IMPORT_ERROR=1

cd "$(dirname "$0")"

ENV="kitchen-mixed-v0"
METHODS="qtpt spt qsft qdt"
SEEDS="0 1 2"
EPOCHS=100
OUTPUT_DIR=results_kitchen_all
MAX_PARALLEL=3

run_job() {
    local method=$1 seed=$2
    local result_dir="${OUTPUT_DIR}/${method}_${ENV}_seed${seed}"
    
    if [ -f "${result_dir}/log.csv" ]; then
        lines=$(wc -l < "${result_dir}/log.csv")
        if [ "$lines" -gt "$EPOCHS" ]; then
            echo "[SKIP] ${method} | ${ENV} | seed=${seed}"
            return 0
        fi
    fi
    
    echo "[START] ${method} | ${ENV} | seed=${seed} | $(date)"
    python train_kitchen_multitask.py \
        --method $method \
        --env $ENV \
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
        > "log_${method}_${ENV}_seed${seed}.txt" 2>&1
    echo "[DONE] ${method} | ${ENV} | seed=${seed} | $(date)"
}

echo "Kitchen mixed-v0: 5 methods x 3 seeds, max $MAX_PARALLEL parallel"
echo "Start: $(date)"

running=0
for METHOD in $METHODS; do
    for SEED in $SEEDS; do
        # Check skip (CQL dummy files)
        result_dir="${OUTPUT_DIR}/${METHOD}_${ENV}_seed${SEED}"
        if [ -f "${result_dir}/log.csv" ]; then
            lines=$(wc -l < "${result_dir}/log.csv")
            if [ "$lines" -gt "$EPOCHS" ]; then
                echo "[SKIP] ${METHOD} | ${ENV} | seed=${SEED}"
                continue
            fi
        fi
        
        run_job "$METHOD" "$SEED" &
        running=$((running + 1))
        
        if [ "$running" -ge "$MAX_PARALLEL" ]; then
            wait -n 2>/dev/null || wait
            running=$((running - 1))
        fi
    done
done

wait
echo "ALL COMPLETE: $(date)"

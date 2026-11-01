#!/bin/bash
# AntMaze Full Experiment: 2 envs  6 methods  4 seeds = 48 runs
# Skips already-completed runs (checks for log.csv in results dir)

# Activate your Python environment before running this script if needed.
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
if [ -n "$MUJOCO_PATH" ]; then export LD_LIBRARY_PATH="$MUJOCO_PATH/bin:$LD_LIBRARY_PATH"; fi
export D4RL_SUPPRESS_IMPORT_ERROR=1

cd "$(dirname "$0")"

ENVS="antmaze-umaze-v2 antmaze-umaze-diverse-v2"
METHODS="qtpt spt iql qsft qdt"
SEEDS="0 1 2"
EPOCHS=50
OUTPUT_DIR=results_antmaze

total=0
skipped=0
done=0

for ENV in $ENVS; do
    for METHOD in $METHODS; do
        for SEED in $SEEDS; do
            total=$((total + 1))
            RESULT_DIR="${OUTPUT_DIR}/${METHOD}_${ENV}_seed${SEED}"

            # Skip if already completed (log.csv has enough lines = epochs)
            if [ -f "${RESULT_DIR}/log.csv" ]; then
                lines=$(wc -l < "${RESULT_DIR}/log.csv")
                if [ "$lines" -gt "$EPOCHS" ]; then
                    echo "[SKIP] ${METHOD} | ${ENV} | seed=${SEED} (already done, ${lines} lines)"
                    skipped=$((skipped + 1))
                    continue
                fi
            fi

            done=$((done + 1))
            echo "=========================================="
            echo "[${done}/${total}] ${METHOD} | ${ENV} | seed=${SEED}"
            echo "Start: $(date)"
            echo "=========================================="

            python train_antmaze_multitask.py \
                --method $METHOD \
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
                --seed $SEED \
                --output_dir $OUTPUT_DIR \
                2>&1 | tee "log_${METHOD}_${ENV}_seed${SEED}.txt"

            echo "[DONE] ${METHOD} | ${ENV} | seed=${SEED} | $(date)"
        done
    done
done

echo "=========================================="
echo "ALL COMPLETE: ${done} ran, ${skipped} skipped, ${total} total"
echo "Finished: $(date)"
echo "=========================================="

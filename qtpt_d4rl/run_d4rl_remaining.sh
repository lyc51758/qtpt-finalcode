#!/bin/bash
# Run QTPT-Continuous on remaining 2 HalfCheetah datasets (50 epochs each)
# Activate your Python environment before running this script if needed.
if [ -n "$MUJOCO_PATH" ]; then export LD_LIBRARY_PATH="$MUJOCO_PATH/bin:$LD_LIBRARY_PATH"; fi
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export D4RL_SUPPRESS_IMPORT_ERROR=1

cd "$(dirname "$0")"

for ENV in halfcheetah-medium-replay-v2 halfcheetah-medium-expert-v2; do
    echo "============================================"
    echo "Starting: $ENV"
    echo "Time: $(date)"
    echo "============================================"

    python3.10 qtpt_continuous.py \
        --env $ENV \
        --context_len 20 \
        --num_epochs 50 \
        --batch_size 256 \
        --lr 3e-4 \
        --n_embd 128 \
        --n_layer 4 \
        --n_head 4 \
        --dropout 0.1 \
        --gamma 0.99 \
        --tau 0.005 \
        --bc_alpha 2.5 \
        --eval_episodes 10 \
        --eval_freq 5 \
        --seed 0 \
        --output_dir results_d4rl \
        --gpu 0

    echo "Finished: $ENV at $(date)"
    echo ""
done

echo "ALL DONE at $(date)"

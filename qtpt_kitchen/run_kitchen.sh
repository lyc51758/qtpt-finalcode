#!/bin/bash
# Run QTPT on Kitchen environments (partial + mixed)
# context_len=30, other params same as antmaze

set -e

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
if [ -n "$MUJOCO_PATH" ]; then export LD_LIBRARY_PATH="$MUJOCO_PATH/bin:$LD_LIBRARY_PATH"; fi
# Activate your Python environment before running this script if needed.
export D4RL_SUPPRESS_IMPORT_ERROR=1

cd "$(dirname "$0")"

echo "=========================================="
echo "Kitchen QTPT Experiment (context_len=30)"
echo "Start: $(date)"
echo "=========================================="

python train_kitchen_qtpt.py \
    --env all \
    --context_len 20 \
    --num_epochs 100 \
    --batch_size 128 \
    --lr 3e-4 \
    --n_embd 128 \
    --n_layer 4 \
    --n_head 4 \
    --dropout 0.1 \
    --gamma 0.99 \
    --eval_freq 5 \
    --eval_episodes 20 \
    --seed 0 \
    --output_dir results_kitchen_v6 \
    --gpu 0 \
    2>&1 | tee run_kitchen_v6.log

echo "=========================================="
echo "Done: $(date)"
echo "=========================================="

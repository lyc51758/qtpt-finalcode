#!/bin/bash
# AntMaze Multi-Task Experiment
# Fix GLIBCXX issue
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
if [ -n "$MUJOCO_PATH" ]; then export LD_LIBRARY_PATH="$MUJOCO_PATH/bin:$LD_LIBRARY_PATH"; fi
export D4RL_SUPPRESS_IMPORT_ERROR=1

cd "$(dirname "$0")"

# Run each method sequentially on antmaze-umaze-v2
for METHOD in qtpt spt cql iql qsft qdt; do
    echo "=========================================="
    echo "Running $METHOD on antmaze-umaze-v2"
    echo "=========================================="
    python3.10 train_antmaze_multitask.py \
        --method $METHOD \
        --env antmaze-umaze-v2 \
        --context_len 20 \
        --num_epochs 50 \
        --batch_size 128 \
        --lr 3e-4 \
        --n_embd 128 \
        --n_layer 4 \
        --n_head 4 \
        --dropout 0.1 \
        --eval_freq 5 \
        --eval_episodes 20 \
        --seed 0 \
        --output_dir results_antmaze \
        2>&1 | tee "log_${METHOD}_umaze.txt"
done

echo "All methods complete!"

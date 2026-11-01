#!/bin/bash
# Activate your Python environment before running this script if needed.
cd "$(dirname "$0")"

echo "=== Starting mix=0.5 experiments at $(date) ==="

for method in qtpt spt cql iql qsft qdt; do
    echo ">>> Starting $method mix=0.5 at $(date)"
    python3.10 train_baselines_darkroom.py \
        --method $method --data_mix 0.5 --seed 0 \
        --num_epochs 50 --batch_size 256 --lr 5e-5 \
        > log_${method}_mix05.txt 2>&1
    echo ">>> Finished $method mix=0.5 at $(date)"
done

echo "=== All mix=0.5 done at $(date) ==="

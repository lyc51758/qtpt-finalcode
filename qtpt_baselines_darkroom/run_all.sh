#!/bin/bash
# Activate your Python environment before running this script if needed.
cd "$(dirname "$0")"

echo "=== Starting all experiments at $(date) ==="

# Random data experiments
for method in qtpt spt cql iql qsft qdt; do
    echo ">>> Starting $method on random data at $(date)"
    python3.10 train_baselines_darkroom.py \
        --method $method --data_mix 0.0 --seed 0 \
        --num_epochs 50 --batch_size 256 --lr 5e-5 \
        > log_${method}_random.txt 2>&1
    echo ">>> Finished $method on random data at $(date)"
done

# Expert data experiments
for method in qtpt spt cql iql qsft qdt; do
    echo ">>> Starting $method on expert data at $(date)"
    python3.10 train_baselines_darkroom.py \
        --method $method --data_mix 1.0 --seed 0 \
        --num_epochs 50 --batch_size 256 --lr 5e-5 \
        > log_${method}_expert.txt 2>&1
    echo ">>> Finished $method on expert data at $(date)"
done

echo "=== All experiments completed at $(date) ==="

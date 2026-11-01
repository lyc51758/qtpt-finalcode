#!/bin/bash
# Run all QTPT ablation experiments on Darkroom
# 5 variants  2 data conditions = 10 experiments

cd "$(dirname "$0")"
# Activate your Python environment before running this script if needed.

METHODS="qtpt qtpt_mc qtpt_noctx"
EPOCHS=50
BS=256
LR=5e-5

echo "=== Starting ablation experiments at $(date) ==="

# Random data
for method in $METHODS; do
    echo ">>> Starting $method on random data at $(date)"
    python3.10 train_ablation_darkroom.py \
        --method $method --data_mix 0.0 --seed 0 \
        --num_epochs $EPOCHS --batch_size $BS --lr $LR
    echo ">>> Finished $method on random data at $(date)"
done

# Expert data
for method in $METHODS; do
    echo ">>> Starting $method on expert data at $(date)"
    python3.10 train_ablation_darkroom.py \
        --method $method --data_mix 1.0 --seed 0 \
        --num_epochs $EPOCHS --batch_size $BS --lr $LR
    echo ">>> Finished $method on expert data at $(date)"
done

echo "=== All ablation experiments completed at $(date) ==="

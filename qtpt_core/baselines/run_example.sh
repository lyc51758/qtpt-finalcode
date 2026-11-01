#!/bin/bash
# Example script for running IQL/CQL baselines on QTPT environments.

cd "$(dirname "$0")/.."

ENV="linear_bandit"
SOURCE="random"
NUM_ENVS=100000
NUM_PER_TASK=1
GPU=0
USE_WANDB="--use_wandb"
WANDB_PROJECT="qtpt-comparison"

echo "=========================================="
echo "Running IQL and CQL on QTPT environments"
echo "=========================================="
echo "Environment: $ENV"
echo "Data source: $SOURCE"
echo "GPU: $GPU"
echo ""

python baselines/test_iql_cql_on_existing_envs.py \
    --env "$ENV" \
    --source "$SOURCE" \
    --num_envs "$NUM_ENVS" \
    --num_per_task "$NUM_PER_TASK" \
    --method both \
    $USE_WANDB \
    --wandb_project "$WANDB_PROJECT" \
    --gpu "$GPU" \
    --num_epochs 250

echo ""
echo "=========================================="
echo "Test complete."
echo "Check wandb project: $WANDB_PROJECT"
echo "=========================================="

#!/bin/bash
# Activate your Python environment before running this script if needed.
cd "$(dirname "$0")/../qtpt_core"

echo "Generating mix=0.5 darkroom data at $(date)"
python3.10 generate_data.py \
    --env darkroom --envs 100000 --dim 10 --H 100 \
    --hists 1 --samples 1 --mix 0.5 \
    --data_path data/darkroom/
echo "Done at $(date)"

QTPT NeurIPS 2026 Supplementary Code
====================================

This archive contains the code used for the QTPT experiments in the paper
"From Weak Data to Strong Policy: Q-Targets Enable Provable In-Context
Reinforcement Learning".

The package is intended for anonymous review. It excludes generated datasets,
trained checkpoints, logs, cached files, local machine paths, server-specific
paths, and personal information.


1. Package Structure
--------------------

- qtpt_core/
  Core QTPT model code, bandit and Darkroom utilities, data generation scripts,
  and matched Transformer baseline adapters.

- qtpt_baselines_darkroom/
  Darkroom baseline experiments for QTPT, SPT, CQL, IQL, Q-SFT, and QDT under
  the same context-conditioned evaluation protocol.

- qtpt_ablation_darkroom/
  Darkroom ablations used to isolate context conditioning and Bellman/TD target
  design, including QTPT, Monte Carlo targets, and no-context variants.

- qtpt_kitchen/
  D4RL Kitchen experiments under matched Transformer baselines.

- qtpt_antmaze/
  D4RL AntMaze experiments under matched Transformer baselines.

- qtpt_d4rl/
  Continuous-action D4RL prototype code.

- diagnostics/stitching_toy/
  A small diagnostic toy MDP illustrating the value-stitching mechanism behind
  the QTPT-vs-QTPT-MC ablation. This is not a benchmark; it is included only as
  a compact mechanism check.


2. Environment Setup
--------------------

Recommended environment:

    Python 3.10
    PyTorch with CUDA support
    One A100 40GB GPU for the main experiments

Install the base dependencies:

    cd qtpt_core
    pip install -r requirements.txt

For D4RL Kitchen and AntMaze, install MuJoCo and D4RL-compatible dependencies
according to your local CUDA/MuJoCo setup. If MuJoCo is installed outside the
default loader path, set MUJOCO_PATH before running the D4RL scripts.

WandB logging is optional. If logging is desired, set:

    export WANDB_API_KEY=<your_wandb_key>

Otherwise, use the no-WandB flags provided by the scripts where available.


3. Data Generation
------------------

Generated data is not included in this archive. Scripts create local data,
model, and result directories at runtime.

Bandit data can be generated from qtpt_core:

    cd qtpt_core
    python generate_data.py --env linear_bandit --algos random linucb \
        --envs 100000 --num_per_task 1 --data_path data/

Darkroom data can be generated from qtpt_core. For example:

    cd qtpt_core
    python generate_data.py --env darkroom --envs 100000 --dim 10 --H 100 \
        --hists 1 --samples 1 --mix 0.0 --data_path data/darkroom/

Use --mix 1.0 for expert roll-in data. The helper script below generates the
mixed-data setting used by the Darkroom baseline experiments:

    cd qtpt_baselines_darkroom
    bash gen_mix05.sh


4. Main QTPT Training
---------------------

The main QTPT pretraining entry point is:

    cd qtpt_core
    python pretrain_v2.py --help

Example usage:

    python pretrain_v2.py --env linear_bandit --source random --Q \
        --num_envs 100000 --num_per_task 1 --num_epochs 250


5. Darkroom Baselines and Ablations
-----------------------------------

Run matched Transformer baselines on Darkroom:

    cd qtpt_baselines_darkroom
    bash run_all.sh

Run the mixed-data Darkroom setting:

    bash run_mix05.sh

Run Darkroom ablations:

    cd qtpt_ablation_darkroom
    bash run_ablation.sh

Individual scripts also expose command-line help:

    python train_baselines_darkroom.py --help
    python train_ablation_darkroom.py --help


6. Diagnostic Stitching Experiment
----------------------------------

The diagnostic stitching toy MDP illustrates why TD-style QTPT can benefit from
partial offline trajectory fragments. The dataset contains no complete
start-to-goal optimal trajectory; QTPT can propagate value across fragments via
Bellman backup, while Monte Carlo targets and supervised imitation are more
limited.

Run the diagnostic experiment:

    cd diagnostics/stitching_toy
    python run_stitching_multiseed.py --seeds 0 1 2 3 4 --n-steps 1000

Generated CSV/JSON summaries are written under diagnostics/stitching_toy/results/
and are intentionally not included in this archive.


7. D4RL Experiments
-------------------

Kitchen:

    cd qtpt_kitchen
    python train_kitchen_multitask.py --help
    bash run_kitchen_all.sh

AntMaze:

    cd qtpt_antmaze
    python train_antmaze_multitask.py --help
    bash run_antmaze_all.sh

The D4RL scripts compare QTPT against matched Transformer baselines under the
same revised experimental protocol. Result directories and checkpoints are
created locally when the scripts are run.


8. Outputs
----------

Training scripts write outputs to local directories such as:

    results/
    models/
    qtpt_baselines_darkroom/results/
    qtpt_ablation_darkroom/results/
    qtpt_kitchen/results/
    qtpt_antmaze/results/
    diagnostics/stitching_toy/results/

These generated artifacts are intentionally not included in the review archive.


9. Notes on Scope
-----------------

This supplement focuses on the QTPT reinforcement-learning experiments reported
in the paper, including the controlled bandit/Darkroom experiments, Darkroom
ablations, D4RL Kitchen/AntMaze extensions, and a compact diagnostic stitching
toy MDP that explains the TD-vs-Monte-Carlo ablation mechanism.

Incomplete exploratory code, result records, notebooks, cached files, trained
checkpoints, and unrelated experimental branches are intentionally omitted to
keep the submission compact, anonymous, and reproducible.

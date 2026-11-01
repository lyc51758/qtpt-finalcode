# Q-Target Pretrained Transformers (QTPT)

This repository contains supplementary code for **From Weak Data to Strong Policy: Q-Targets Enable Provable In-Context Reinforcement Learning** (NeurIPS 2026).

QTPT trains a Transformer to predict history-conditioned action values from offline trajectories. It uses reward and Bellman targets rather than directly imitating the actions in the dataset.

## Contents

| Directory | Description |
| --- | --- |
| `qtpt_core/` | Core models, training, environments, data generation, and baseline adapters |
| `qtpt_ablation_darkroom/` | Darkroom ablations comparing QTPT, Monte Carlo targets, and a no-context variant |
| `qtpt_baselines_darkroom/` | Darkroom baseline experiments |
| `qtpt_kitchen/` | D4RL Kitchen experiments |
| `qtpt_antmaze/` | D4RL AntMaze experiments |
| `qtpt_d4rl/` | Continuous-action D4RL prototype |
| `diagnostics/stitching_toy/` | A small value-stitching diagnostic |

## Setup

The original experiments use Python, PyTorch, and additional packages listed in `qtpt_core/requirements.txt`. Install a PyTorch build compatible with your system before installing the remaining dependencies. Kitchen and AntMaze additionally require a compatible MuJoCo and D4RL installation.

## Running experiments

The original supplementary instructions are in [README.txt](README.txt). They include data-generation and training commands, as well as the Darkroom and D4RL entry points. Generated datasets, checkpoints, and raw results are not included in this repository.

For a small diagnostic, the stitching toy can be run with:

```bash
cd diagnostics/stitching_toy
python run_stitching_multiseed.py --seeds 0 1 2 3 4 --n-steps 1000
```

The commands and their correspondence to every reported paper result are being checked. In particular, the current local environment does not have PyTorch, so the diagnostic command has not yet been verified here. Please consult the paper and the supplementary instructions for each experiment's settings.

## Acknowledgments

The implementation in `qtpt_core/model.py` notes that it is based largely on [in-context-rl/net.py](https://github.com/licong-lin/in-context-rl/blob/main/net.py). Licensing and attribution for this code and other third-party components are being reviewed before public release.

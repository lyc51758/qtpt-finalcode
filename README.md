# Q-Target Pretrained Transformers (QTPT)

[English](README.md) | [中文](README_zh-CN.md)

This repository contains supplementary code for **From Weak Data to Strong Policy: Q-Targets Enable Provable In-Context Reinforcement Learning** (NeurIPS 2026).

QTPT trains a Transformer to predict history-conditioned action values from offline trajectories. It uses reward and Bellman targets rather than directly imitating the actions in the dataset.

**Paper:** [arXiv:2609.30391](https://arxiv.org/abs/2609.30391)

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

The experiments use Python and PyTorch. Install a PyTorch build compatible with your system, then install the dependencies listed in `qtpt_core/requirements.txt`:

```bash
pip install -r qtpt_core/requirements.txt
```

Kitchen and AntMaze additionally require a compatible MuJoCo and D4RL installation. The dependency file includes D4RL, which may require a separate environment setup.

## Running experiments

The original supplementary instructions are in [README.txt](README.txt). They include data-generation and training commands, as well as the Darkroom and D4RL entry points. Generated datasets, checkpoints, and raw results are not included in this repository.

For a small diagnostic, the stitching toy can be run with:

```bash
cd diagnostics/stitching_toy
python run_stitching_multiseed.py --seeds 0 1 2 3 4 --n-steps 1000
```

Please consult the paper and the supplementary instructions for the settings used in each experiment.

## Citation

If you find this work useful, please cite:

```bibtex
@article{lin2026qtpt,
  title   = {From Weak Data to Strong Policy: Q-Targets Enable Provable In-Context Reinforcement Learning},
  author  = {Lin, Yichen and Xiong, Xuyuan and Wang, Xue and Meng, Xiangfu and Wei, Mike Mingcheng and Yao, Tao},
  journal = {arXiv preprint arXiv:2609.30391},
  year    = {2026}
}
```

## Acknowledgments

The implementation in `qtpt_core/model.py` is based on [in-context-rl](https://github.com/licong-lin/in-context-rl).

## License

Contributions by the QTPT authors are available under the [MIT License](LICENSE). This license does not grant rights in third-party material. `qtpt_core/model.py` is based on upstream `in-context-rl/net.py`; no license was found in that upstream repository, so rights to the upstream material require separate clarification.

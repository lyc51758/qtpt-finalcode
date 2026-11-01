# Q-Target Pretrained Transformers（QTPT）

[English](README.md) | [中文](README_zh-CN.md)

本仓库包含论文 **From Weak Data to Strong Policy: Q-Targets Enable Provable In-Context Reinforcement Learning**（NeurIPS 2026）的补充代码。

QTPT 利用离线轨迹训练 Transformer，预测以交互历史为条件的动作价值。它通过奖励和 Bellman 目标学习，而不是直接模仿数据中的动作。

**论文：** [arXiv:2609.30391](https://arxiv.org/abs/2609.30391)

## 目录

| 目录 | 内容 |
| --- | --- |
| `qtpt_core/` | 核心模型、训练、环境、数据生成和基线适配代码 |
| `qtpt_ablation_darkroom/` | Darkroom 上 QTPT、Monte Carlo 目标和无上下文变体的消融实验 |
| `qtpt_baselines_darkroom/` | Darkroom 基线实验 |
| `qtpt_kitchen/` | D4RL Kitchen 实验 |
| `qtpt_antmaze/` | D4RL AntMaze 实验 |
| `qtpt_d4rl/` | 连续动作 D4RL 原型代码 |
| `diagnostics/stitching_toy/` | 一个用于分析价值拼接的小型诊断实验 |

## 环境配置

实验使用 Python 和 PyTorch。请先安装与你的系统兼容的 PyTorch，再安装 `qtpt_core/requirements.txt` 中的依赖：

```bash
pip install -r qtpt_core/requirements.txt
```

Kitchen 和 AntMaze 实验还需要兼容的 MuJoCo 和 D4RL。依赖列表包含 D4RL，它可能需要单独配置环境。

## 运行实验

数据生成、训练、Darkroom 和 D4RL 入口的说明见 [README.txt](README.txt)。仓库不包含生成的数据集、模型检查点和原始实验结果。

运行 stitching toy 诊断实验：

```bash
cd diagnostics/stitching_toy
python run_stitching_multiseed.py --seeds 0 1 2 3 4 --n-steps 1000
```

各实验的设置请结合论文和补充说明查看。

## 引用

如果你觉得本工作对你的研究有所帮助，欢迎引用：

```bibtex
@article{lin2026qtpt,
  title   = {From Weak Data to Strong Policy: Q-Targets Enable Provable In-Context Reinforcement Learning},
  author  = {Lin, Yichen and Xiong, Xuyuan and Wang, Xue and Meng, Xiangfu and Wei, Mike Mingcheng and Yao, Tao},
  journal = {arXiv preprint arXiv:2609.30391},
  year    = {2026}
}
```

## 致谢

`qtpt_core/model.py` 的实现基于 [in-context-rl](https://github.com/licong-lin/in-context-rl)。

## 许可证

QTPT 作者的贡献采用 [MIT License](LICENSE)。该许可证不授予第三方材料的权利。`qtpt_core/model.py` 基于上游 `in-context-rl/net.py`，该上游仓库未发现许可证，因此上游材料的授权仍需单独确认。

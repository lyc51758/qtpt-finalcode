# QTPT supplementary code (private review)

Code prepared as supplementary material for "From Weak Data to Strong Policy: Q-Targets Enable Provable In-Context Reinforcement Learning" (NeurIPS 2026).

This repository is a private staging copy, not a verified reproduction release. The package contains experimental implementations for Linear Bandit, Darkroom, Kitchen and AntMaze, plus an illustrative stitching toy. It does not include generated datasets, checkpoints or raw results. Its scope and commands have not yet been independently verified against every result in the paper.

The original package instructions are retained in [README.txt](README.txt) for audit; the references to anonymous review and reproducibility there have not been revalidated for a public release. Dependencies are listed in [qtpt_core/requirements.txt](qtpt_core/requirements.txt). The stitching toy command in README.txt requires PyTorch; it could not be run in the current local environment because PyTorch is not installed.

The source notes that [qtpt_core/model.py](qtpt_core/model.py) is largely based on the upstream [in-context-rl/net.py](https://github.com/licong-lin/in-context-rl/blob/main/net.py). No license was found in that upstream repository during the preliminary check. Permission and licensing must be resolved before public release. Other third-party dependencies and source attributions also require review.

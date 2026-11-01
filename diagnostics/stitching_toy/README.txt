Diagnostic Stitching Toy MDP
============================

This directory contains a small diagnostic experiment for the QTPT paper. It
is intended to illustrate the mechanism behind the main-text QTPT-MC ablation:
bootstrapped TD targets can propagate value information across partial offline
trajectory fragments, whereas Monte Carlo targets and behavior imitation are
limited to the returns or actions observed within each fragment.

The toy dataset intentionally contains no full start-to-goal optimal trajectory.
It contains:

- Type-A fragments starting from the initial state, which reveal local
  transition structure but no downstream reward.
- Type-B fragments starting from an intermediate state, which reveal which
  terminal goal is rewarding for the current task.

QTPT can use Bellman backup to propagate the value of Type-B fragments back to
the useful Type-A transition. This is the diagnostic "stitching" mechanism.


Files
-----

- stitching_toy_mdp.py
  Defines the toy MDP, dataset generator, tiny context-conditioned Transformer,
  QTPT/QTPT-MC/SPT objectives, and single-seed evaluation.

- run_stitching_multiseed.py
  Runs the diagnostic experiment over multiple seeds and writes local CSV/JSON
  summaries under the chosen output directory.


Example
-------

From this directory:

    python run_stitching_multiseed.py --seeds 0 1 2 3 4 --n-steps 1000

The script writes generated outputs to:

    results/stitching_multiseed_results.csv
    results/stitching_multiseed_summary.json

Generated result files are not included in the review archive.


Scope
-----

This is not a new benchmark and is not used to claim broad empirical
superiority. It is a controlled diagnostic that explains one mechanism behind
the TD-vs-Monte-Carlo gap reported in the paper's ablation study.

"""
Re-evaluate best model checkpoints to compute mean  std over 100 eval environments.
Covers both:
  1. Darkroom baseline comparison (qtpt, spt, cql, iql, qsft, qdt)
  2. QTPT ablation (qtpt, qtpt_mc, qtpt_noctx)

For each method  data condition, loads best_model.pth and runs evaluate_darkroom
to get all 100 per-environment rewards, then computes mean and std.

Usage:
    python reeval_best_models.py \
        --baseline_dir ../qtpt_baselines_darkroom/results \
        --ablation_dir ./results \
        --output_dir ./reeval_results
"""

import os
import sys
import json
import copy
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'qtpt_core'))
from model import Transformer
from pretrain_v2 import evaluate_darkroom

#  shared config 

BASE_CONFIG = {
    'env': 'darkroom',
    'horizon': 100,
    'dim': 10,
    'act_num': 5,
    'action_dim': 5,
    'state_dim': 2,
    'n_embd': 128,
    'n_layer': 6,
    'n_head': 4,
    'dropout': 0.2,
    'activation': 'relu',
    'pred_q': True,
    'test': False,
    'lr': 5e-5,
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'qtpt_core', 'data', 'darkroom')

#  model wrappers (minimal, just enough for evaluate_darkroom) 

class SingleNetWrapper(nn.Module):
    """Wraps a single Transformer for evaluate_darkroom compatibility."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)

    @property
    def test(self): return self.net.test
    @test.setter
    def test(self, v): self.net.test = v
    def eval(self): self.net.eval(); return self
    def train(self, mode=True): self.net.train(mode); return self


def load_q_model(ckpt_path, device):
    """Load a Q-network checkpoint (used by qtpt, cql, iql, qsft, ablations)."""
    cfg = BASE_CONFIG.copy()
    cfg['pred_q'] = True
    net = Transformer(cfg, device)
    state = torch.load(ckpt_path, map_location=device)
    # checkpoints may be full model state_dict or just q_net
    # try loading directly, then strip prefix
    try:
        net.load_state_dict(state)
    except RuntimeError:
        # strip "q_net." prefix if saved from wrapper
        new_state = {k.replace('q_net.', ''): v for k, v in state.items() if 'q_net.' in k}
        if not new_state:
            # try other common prefixes
            new_state = {k.replace('net.', ''): v for k, v in state.items() if 'net.' in k}
        net.load_state_dict(new_state, strict=False)
    net.to(device)
    return SingleNetWrapper(net).to(device)


def load_spt_model(ckpt_path, device):
    """Load an SPT (behavior cloning) checkpoint."""
    cfg = BASE_CONFIG.copy()
    cfg['pred_q'] = False
    net = Transformer(cfg, device)
    state = torch.load(ckpt_path, map_location=device)
    try:
        net.load_state_dict(state)
    except RuntimeError:
        new_state = {k.replace('net.', ''): v for k, v in state.items() if 'net.' in k}
        net.load_state_dict(new_state, strict=False)
    net.to(device)
    return SingleNetWrapper(net).to(device)


#  evaluation 

def reeval(model, eval_trajs, device, n_repeats=1):
    """
    Run evaluate_darkroom n_repeats times and collect all per-env rewards.
    With n_repeats=1 we get 100 values; with n_repeats=3 we get 300.
    Returns (mean, std, all_rewards).
    """
    all_rewards = []
    for _ in range(n_repeats):
        rewards = evaluate_darkroom(model, BASE_CONFIG, device, eval_trajs, mode='greedy')
        all_rewards.extend(rewards)
    return float(np.mean(all_rewards)), float(np.std(all_rewards)), all_rewards


def load_eval_trajs(mix):
    path = os.path.join(DATA_DIR, f'trajs_darkroom_envs100_H100_d10_mix{mix}_eval.pkl')
    if not os.path.exists(path):
        path = os.path.join(DATA_DIR, 'trajs_darkroom_envs100_H100_d10_mix0.0_eval.pkl')
    return pickle.load(open(path, 'rb'))


#  main 

BASELINE_METHODS = ['qtpt', 'spt', 'cql', 'iql', 'qsft', 'qdt']
ABLATION_METHODS = ['qtpt', 'qtpt_mc', 'qtpt_noctx']

METHOD_LABELS = {
    'qtpt':      'QTPT (Ours)',
    'spt':       'SPT',
    'cql':       'CQL',
    'iql':       'IQL',
    'qsft':      'Q-SFT',
    'qdt':       'QDT',
    'qtpt_mc':   'QTPT-MC',
    'qtpt_noctx':'QTPT-NoCtx',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline_dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'qtpt_baselines_darkroom', 'results'))
    parser.add_argument('--ablation_dir', type=str,
                        default='results')
    parser.add_argument('--output_dir', type=str,
                        default='reeval_results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_repeats', type=int, default=3,
                        help='Number of eval repeats per model (each uses 100 envs)')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    results = {}  # {group: {mix: {method: {mean, std, n}}}}

    for mix in ['0.0', '1.0']:
        eval_trajs = load_eval_trajs(mix)
        mix_label = 'random' if mix == '0.0' else 'expert'
        print(f'\n{"="*60}')
        print(f'Data: {mix_label} (mix={mix}), {len(eval_trajs)} eval envs')
        print(f'{"="*60}')

        #  Baseline methods 
        for method in BASELINE_METHODS:
            ckpt = os.path.join(args.baseline_dir,
                                f'{method}_mix{mix}_seed{args.seed}', 'best_model.pth')
            if not os.path.exists(ckpt):
                print(f'  [SKIP] {method}: no checkpoint at {ckpt}')
                continue

            print(f'  Evaluating baseline/{method} ...', end=' ', flush=True)
            try:
                if method == 'spt':
                    model = load_spt_model(ckpt, device)
                else:
                    model = load_q_model(ckpt, device)

                mean, std, all_r = reeval(model, eval_trajs, device, args.n_repeats)
                n = len(all_r)
                ci95 = 1.96 * std / np.sqrt(n)
                print(f'mean={mean:.2f}, std={std:.2f}, 95%CI={ci95:.2f} (n={n})')

                key = f'baseline/{method}/mix{mix}'
                results[key] = {
                    'method': METHOD_LABELS.get(method, method),
                    'group': 'baseline',
                    'data': mix_label,
                    'mean': mean,
                    'std': std,
                    'ci95': ci95,
                    'n': n,
                }
            except Exception as e:
                print(f'ERROR: {e}')

        #  Ablation methods 
        for method in ABLATION_METHODS:
            ckpt = os.path.join(args.ablation_dir,
                                f'{method}_mix{mix}_seed{args.seed}', 'best_model.pth')
            if not os.path.exists(ckpt):
                print(f'  [SKIP] ablation/{method}: no checkpoint at {ckpt}')
                continue

            print(f'  Evaluating ablation/{method} ...', end=' ', flush=True)
            try:
                model = load_q_model(ckpt, device)
                mean, std, all_r = reeval(model, eval_trajs, device, args.n_repeats)
                n = len(all_r)
                ci95 = 1.96 * std / np.sqrt(n)
                print(f'mean={mean:.2f}, std={std:.2f}, 95%CI={ci95:.2f} (n={n})')

                key = f'ablation/{method}/mix{mix}'
                results[key] = {
                    'method': METHOD_LABELS.get(method, method),
                    'group': 'ablation',
                    'data': mix_label,
                    'mean': mean,
                    'std': std,
                    'ci95': ci95,
                    'n': n,
                }
            except Exception as e:
                print(f'ERROR: {e}')

    #  Save JSON 
    json_path = os.path.join(args.output_dir, 'reeval_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved raw results to {json_path}')

    #  Print tables 
    for group, methods in [('baseline', BASELINE_METHODS), ('ablation', ABLATION_METHODS)]:
        print(f'\n{"="*70}')
        print(f'{group.upper()} RESULTS  (mean  std, n={args.n_repeats * 100} eval envs)')
        print(f'{"="*70}')
        print(f'{"Method":<16} {"Random MeanStd":>20} {"Expert MeanStd":>20}')
        print('-' * 58)
        for method in methods:
            r_key = f'{group}/{method}/mix0.0'
            e_key = f'{group}/{method}/mix1.0'
            r = results.get(r_key)
            e = results.get(e_key)
            r_str = f'{r["mean"]:.2f}{r["std"]:.2f}' if r else 'N/A'
            e_str = f'{e["mean"]:.2f}{e["std"]:.2f}' if e else 'N/A'
            label = METHOD_LABELS.get(method, method)
            print(f'{label:<16} {r_str:>20} {e_str:>20}')

    #  Save CSV tables 
    import csv
    for group, methods in [('baseline', BASELINE_METHODS), ('ablation', ABLATION_METHODS)]:
        csv_path = os.path.join(args.output_dir, f'{group}_table.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Method', 'Random_Mean', 'Random_Std', 'Random_CI95',
                             'Expert_Mean', 'Expert_Std', 'Expert_CI95'])
            for method in methods:
                r = results.get(f'{group}/{method}/mix0.0', {})
                e = results.get(f'{group}/{method}/mix1.0', {})
                writer.writerow([
                    METHOD_LABELS.get(method, method),
                    f'{r.get("mean", ""):.4f}' if r else '',
                    f'{r.get("std", ""):.4f}' if r else '',
                    f'{r.get("ci95", ""):.4f}' if r else '',
                    f'{e.get("mean", ""):.4f}' if e else '',
                    f'{e.get("std", ""):.4f}' if e else '',
                    f'{e.get("ci95", ""):.4f}' if e else '',
                ])
        print(f'Saved {group} table to {csv_path}')


if __name__ == '__main__':
    main()

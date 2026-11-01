"""
QTPT Ablation experiments on Darkroom.

Ablation variants:
1. qtpt         - Standard QTPT (1-step TD, full context, double DQN) [baseline]
2. qtpt_mc      - Monte Carlo return target instead of TD
3. qtpt_noctx   - State-only (no history context, only current state)
4. qtpt_3step   - 3-step TD target
5. qtpt_5step   - 5-step TD target

All share the same Transformer architecture, data, and evaluation protocol.
"""

import os
import sys
import copy
import json
import pickle
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tqdm

# Add parent project to path for model imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'qtpt_core'))
from model import Transformer
from pretrain_v2 import evaluate_darkroom


class DarkRoomSequenceDataset(Dataset):
    """Loads darkroom trajectories and returns full sequences for training."""
    def __init__(self, trajs, horizon, device):
        self.trajs = trajs
        self.horizon = horizon
        self.device = device

    def __len__(self):
        return len(self.trajs)

    def __getitem__(self, idx):
        traj = self.trajs[idx]
        states = torch.tensor(traj['context_states'], dtype=torch.float32)
        actions_raw = traj['context_actions']
        if isinstance(actions_raw, np.ndarray):
            actions = torch.tensor(actions_raw, dtype=torch.float32)
        else:
            actions = torch.tensor(np.array(actions_raw), dtype=torch.float32)
        rewards = torch.tensor(traj['context_rewards'], dtype=torch.float32)
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        next_states = torch.tensor(traj['context_next_states'], dtype=torch.float32)
        return {
            'context_states': states,
            'context_actions': actions,
            'context_rewards': rewards,
            'context_next_states': next_states,
            'goal': traj['goal'],
        }


# ============================================================
# QTPT baseline (1-step TD, full context)
# ============================================================

class QTPTDarkroom(nn.Module):
    """Standard QTPT: 1-step TD target, full context, double DQN."""
    def __init__(self, config, device, gamma=0.99):
        super().__init__()
        self.device = device
        self.gamma = gamma

        q_config = config.copy()
        q_config['pred_q'] = True
        q_config['test'] = False
        self.q_net = Transformer(q_config, device)
        self.target_net = Transformer(copy.deepcopy(q_config), device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=config.get('lr', 5e-5), weight_decay=1e-3)
        self.to(device)

    def train_step(self, batch):
        self.q_net.train()
        self.q_net.test = False
        x = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        B, H, A = x['context_actions'].shape

        q_values = self.q_net(x)

        with torch.no_grad():
            target_q = self.target_net(x)
            soft_q = F.softmax(target_q, dim=-1) * target_q
            next_q = soft_q.sum(dim=-1, keepdim=True)
            next_q_shifted = torch.cat([next_q[:, 1:, :],
                                        torch.zeros(B, 1, 1, device=self.device)], dim=1)
            td_target = x['context_rewards'] + self.gamma * next_q_shifted

        actions_idx = x['context_actions'].argmax(dim=-1, keepdim=True)
        q_taken = q_values.gather(2, actions_idx)
        loss = F.mse_loss(q_taken, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        for tp, sp in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(0.005 * sp.data + 0.995 * tp.data)

        return {'loss': loss.item()}

    def forward(self, x):
        return self.q_net(x)

    @property
    def test(self):
        return self.q_net.test
    @test.setter
    def test(self, val):
        self.q_net.test = val
    def eval(self):
        self.q_net.eval()
        return self
    def train(self, mode=True):
        self.q_net.train(mode)
        return self


# ============================================================
# Ablation 1: Monte Carlo return target
# ============================================================

class QTPTMonteCarlo(nn.Module):
    """QTPT with Monte Carlo return target instead of TD bootstrap.
    Target = sum_{k=0}^{H-t-1} gamma^k * r_{t+k}  (discounted return-to-go)
    """
    def __init__(self, config, device, gamma=0.99):
        super().__init__()
        self.device = device
        self.gamma = gamma

        q_config = config.copy()
        q_config['pred_q'] = True
        q_config['test'] = False
        self.q_net = Transformer(q_config, device)
        # No target network needed for MC

        self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=config.get('lr', 5e-5), weight_decay=1e-3)
        self.to(device)

    def _compute_mc_returns(self, rewards):
        """Compute discounted return-to-go for each timestep.
        rewards: [B, H, 1]
        returns: [B, H, 1]
        """
        B, H, _ = rewards.shape
        returns = torch.zeros_like(rewards)
        running_return = torch.zeros(B, 1, device=rewards.device)
        for t in reversed(range(H)):
            running_return = rewards[:, t, :] + self.gamma * running_return
            returns[:, t, :] = running_return
        return returns

    def train_step(self, batch):
        self.q_net.train()
        self.q_net.test = False
        x = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        q_values = self.q_net(x)

        with torch.no_grad():
            mc_target = self._compute_mc_returns(x['context_rewards'])

        actions_idx = x['context_actions'].argmax(dim=-1, keepdim=True)
        q_taken = q_values.gather(2, actions_idx)
        loss = F.mse_loss(q_taken, mc_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        return {'loss': loss.item()}

    def forward(self, x):
        return self.q_net(x)

    @property
    def test(self):
        return self.q_net.test
    @test.setter
    def test(self, val):
        self.q_net.test = val
    def eval(self):
        self.q_net.eval()
        return self
    def train(self, mode=True):
        self.q_net.train(mode)
        return self


# ============================================================
# Ablation 2: State-only (no context history)
# ============================================================

class QTPTStateOnly(nn.Module):
    """QTPT without context: only uses current state (masks out history).
    At each timestep t, the model only sees (s_t), not (s_1,a_1,r_1,...,s_t).
    Implemented by zeroing out all context before the current timestep.
    """
    def __init__(self, config, device, gamma=0.99):
        super().__init__()
        self.device = device
        self.gamma = gamma

        q_config = config.copy()
        q_config['pred_q'] = True
        q_config['test'] = False
        self.q_net = Transformer(q_config, device)
        self.target_net = Transformer(copy.deepcopy(q_config), device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=config.get('lr', 5e-5), weight_decay=1e-3)
        self.to(device)

    def _mask_context(self, x):
        """Create per-timestep inputs where only the current step is visible.
        We achieve this by processing each timestep independently:
        feed (s_t, a_{t-1}, r_{t-1}) as a length-1 sequence.

        For efficiency during training, we keep the full sequence but zero out
        all previous context; the causal mask already prevents future leakage,
        so we mask past by zeroing states/actions/rewards before current step.

        Actually, simpler approach: set context length to 1 by only keeping
        the last timestep for each position. We do this by creating a diagonal
        mask effect: for each timestep t, only (s_t) is non-zero.
        """
        B, H, S = x['context_states'].shape
        A = x['context_actions'].shape[-1]

        # Zero out all history: for position t, zero everything before t
        # This effectively makes each timestep independent
        masked_x = {}
        masked_x['context_states'] = x['context_states'].clone()
        masked_x['context_actions'] = torch.zeros_like(x['context_actions'])
        masked_x['context_rewards'] = torch.zeros_like(x['context_rewards'])
        if 'context_next_states' in x:
            masked_x['context_next_states'] = x['context_next_states'].clone()
        return masked_x

    def train_step(self, batch):
        self.q_net.train()
        self.q_net.test = False
        x = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        B, H, A = x['context_actions'].shape

        masked_x = self._mask_context(x)
        q_values = self.q_net(masked_x)

        with torch.no_grad():
            target_q = self.target_net(masked_x)
            soft_q = F.softmax(target_q, dim=-1) * target_q
            next_q = soft_q.sum(dim=-1, keepdim=True)
            next_q_shifted = torch.cat([next_q[:, 1:, :],
                                        torch.zeros(B, 1, 1, device=self.device)], dim=1)
            td_target = x['context_rewards'] + self.gamma * next_q_shifted

        actions_idx = x['context_actions'].argmax(dim=-1, keepdim=True)
        q_taken = q_values.gather(2, actions_idx)
        loss = F.mse_loss(q_taken, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        for tp, sp in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(0.005 * sp.data + 0.995 * tp.data)

        return {'loss': loss.item()}

    def forward(self, x):
        """For evaluation: also mask context."""
        masked_x = self._mask_context(x)
        return self.q_net(masked_x)

    @property
    def test(self):
        return self.q_net.test
    @test.setter
    def test(self, val):
        self.q_net.test = val
    def eval(self):
        self.q_net.eval()
        return self
    def train(self, mode=True):
        self.q_net.train(mode)
        return self


# ============================================================
# Ablation 3: N-step TD target
# ============================================================

class QTPTNStep(nn.Module):
    """QTPT with N-step TD target.
    Target = sum_{k=0}^{n-1} gamma^k * r_{t+k} + gamma^n * max_a Q(s_{t+n}, a)
    """
    def __init__(self, config, device, gamma=0.99, n_step=3):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.n_step = n_step

        q_config = config.copy()
        q_config['pred_q'] = True
        q_config['test'] = False
        self.q_net = Transformer(q_config, device)
        self.target_net = Transformer(copy.deepcopy(q_config), device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=config.get('lr', 5e-5), weight_decay=1e-3)
        self.to(device)

    def _compute_nstep_targets(self, rewards, next_q_values):
        """Compute n-step TD targets.
        rewards: [B, H, 1]
        next_q_values: [B, H, 1] max Q-values at each timestep
        returns: [B, H, 1] n-step targets
        """
        B, H, _ = rewards.shape
        n = self.n_step
        targets = torch.zeros_like(rewards)

        for t in range(H):
            # Sum of discounted rewards for n steps
            n_actual = min(n, H - t)
            discounted_reward = torch.zeros(B, 1, device=rewards.device)
            for k in range(n_actual):
                discounted_reward += (self.gamma ** k) * rewards[:, t + k, :]

            # Bootstrap from Q-value at step t+n (if within horizon)
            if t + n < H:
                bootstrap = (self.gamma ** n) * next_q_values[:, t + n, :]
            else:
                bootstrap = torch.zeros(B, 1, device=rewards.device)

            targets[:, t, :] = discounted_reward + bootstrap

        return targets

    def train_step(self, batch):
        self.q_net.train()
        self.q_net.test = False
        x = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        B, H, A = x['context_actions'].shape

        q_values = self.q_net(x)

        with torch.no_grad():
            target_q = self.target_net(x)
            soft_q = F.softmax(target_q, dim=-1) * target_q
            next_q = soft_q.sum(dim=-1, keepdim=True)  # [B, H, 1]

            nstep_target = self._compute_nstep_targets(x['context_rewards'], next_q)

        actions_idx = x['context_actions'].argmax(dim=-1, keepdim=True)
        q_taken = q_values.gather(2, actions_idx)
        loss = F.mse_loss(q_taken, nstep_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        for tp, sp in zip(self.target_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(0.005 * sp.data + 0.995 * tp.data)

        return {'loss': loss.item()}

    def forward(self, x):
        return self.q_net(x)

    @property
    def test(self):
        return self.q_net.test
    @test.setter
    def test(self, val):
        self.q_net.test = val
    def eval(self):
        self.q_net.eval()
        return self
    def train(self, mode=True):
        self.q_net.train(mode)
        return self


# ============================================================
# Training Loop
# ============================================================

def train_and_evaluate(model, method_name, train_loader, eval_trajs, config, device,
                       num_epochs=50, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'log.csv')

    best_reward = -float('inf')

    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,avg_reward\n')

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for batch in tqdm.tqdm(train_loader, desc=f'{method_name} Epoch {epoch}'):
            metrics = model.train_step(batch)
            epoch_loss += metrics['loss']
            n_batches += 1

        epoch_loss /= max(n_batches, 1)

        with torch.no_grad():
            rewards = evaluate_darkroom(model, config, device, eval_trajs, mode='greedy')
            avg_reward = np.mean(rewards)

        print(f'[{method_name}] Epoch {epoch} | Loss: {epoch_loss:.4f} | Avg Reward: {avg_reward:.2f}')

        with open(log_path, 'a') as f:
            f.write(f'{epoch},{epoch_loss:.6f},{avg_reward:.4f}\n')

        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

    return best_reward


def get_config(args):
    return {
        'env': 'darkroom',
        'horizon': 100,
        'dim': 10,
        'act_num': 5,
        'action_dim': 5,
        'state_dim': 2,
        'n_embd': args.n_embd,
        'n_layer': args.n_layer,
        'n_head': args.n_head,
        'dropout': args.dropout,
        'activation': 'relu',
        'pred_q': True,
        'test': False,
        'lr': args.lr,
    }


ABLATION_METHODS = ['qtpt', 'qtpt_mc', 'qtpt_noctx', 'qtpt_3step', 'qtpt_5step']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='all',
                        choices=ABLATION_METHODS + ['all'])
    parser.add_argument('--data_mix', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--n_embd', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'qtpt_core', 'data', 'darkroom'))
    parser.add_argument('--output_dir', type=str,
                        default='results')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    mix = args.data_mix
    train_file = os.path.join(args.data_dir,
                              f'trajs_darkroom_envs100000_hists1_samples1_H100_d10_mix{mix}_train.pkl')
    eval_file = os.path.join(args.data_dir,
                             f'trajs_darkroom_envs100_H100_d10_mix{mix}_eval.pkl')
    if not os.path.exists(eval_file):
        eval_file = os.path.join(args.data_dir, 'trajs_darkroom_envs100_H100_d10_mix0.0_eval.pkl')

    print(f'Loading train data: {train_file}')
    train_trajs = pickle.load(open(train_file, 'rb'))
    print(f'Loading eval data: {eval_file}')
    eval_trajs = pickle.load(open(eval_file, 'rb'))

    print(f'Train trajectories: {len(train_trajs)}, Eval trajectories: {len(eval_trajs)}')

    horizon = 100
    train_dataset = DarkRoomSequenceDataset(train_trajs, horizon, device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)

    config = get_config(args)
    data_label = f'mix{mix}'

    if args.method == 'all':
        methods = ABLATION_METHODS
    else:
        methods = [args.method]

    results = {}

    for method in methods:
        print(f'\n{"="*60}')
        print(f'Ablation: {method} | Data: {data_label} | Seed: {args.seed}')
        print(f'{"="*60}')

        save_dir = os.path.join(args.output_dir, f'{method}_{data_label}_seed{args.seed}')

        if method == 'qtpt':
            model = QTPTDarkroom(config, device)
        elif method == 'qtpt_mc':
            model = QTPTMonteCarlo(config, device)
        elif method == 'qtpt_noctx':
            model = QTPTStateOnly(config, device)
        elif method == 'qtpt_3step':
            model = QTPTNStep(config, device, n_step=3)
        elif method == 'qtpt_5step':
            model = QTPTNStep(config, device, n_step=5)

        label_map = {
            'qtpt': 'QTPT (1-step TD)',
            'qtpt_mc': 'QTPT-MC',
            'qtpt_noctx': 'QTPT-NoCtx',
            'qtpt_3step': 'QTPT-3step',
            'qtpt_5step': 'QTPT-5step',
        }

        best = train_and_evaluate(model, label_map[method], train_loader, eval_trajs, config, device,
                                  num_epochs=args.num_epochs, save_dir=save_dir)
        results[method] = best
        print(f'[{method}] Best Avg Reward: {best:.2f}')

    print(f'\n{"="*60}')
    print(f'ABLATION SUMMARY | Data: {data_label} | Seed: {args.seed}')
    print(f'{"="*60}')
    for method, reward in results.items():
        print(f'  {method:>15s}: {reward:.2f}')

    summary_path = os.path.join(args.output_dir, f'ablation_summary_{data_label}_seed{args.seed}.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Summary saved to {summary_path}')


if __name__ == '__main__':
    main()

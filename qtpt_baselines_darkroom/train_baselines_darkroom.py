"""
Baseline comparison on DarkRoom: CQL, IQL, Q-SFT, QDT, QTPT, SPT.

All methods use the same Transformer architecture and are evaluated with the
same in-context protocol (evaluate_darkroom) for fair comparison.

Usage:
    python train_baselines_darkroom.py --method cql --data_mix 0.0 --seed 0
    python train_baselines_darkroom.py --method all --data_mix 0.0 --seed 0
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'qtpt_core'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import argparse
import random
import json
import copy
import tqdm
from torch.utils.data import Dataset, DataLoader
from model import Transformer, CosineAnnealingWarmup
from envs.darkroom import DarkroomEnv
from pretrain_v2 import evaluate_darkroom

# ============================================================
# Dataset
# ============================================================

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
        states = torch.tensor(traj['context_states'], dtype=torch.float32)        # [H, 2]
        actions_raw = traj['context_actions']                                       # [H, 5]
        if isinstance(actions_raw, np.ndarray):
            actions = torch.tensor(actions_raw, dtype=torch.float32)
        else:
            actions = torch.tensor(np.array(actions_raw), dtype=torch.float32)
        rewards = torch.tensor(traj['context_rewards'], dtype=torch.float32)       # [H,]
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)                                         # [H, 1]

        next_states = torch.tensor(traj['context_next_states'], dtype=torch.float32)  # [H, 2]

        return {
            'context_states': states,
            'context_actions': actions,
            'context_rewards': rewards,
            'context_next_states': next_states,
            'goal': traj['goal'],
        }


# ============================================================
# Baseline Implementations
# ============================================================

class CQLDarkroom(nn.Module):
    """Conservative Q-Learning for Darkroom."""
    def __init__(self, config, device, alpha_cql=1.0, gamma=0.99, tau=0.005):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha_cql = alpha_cql
        self.act_num = config['act_num']

        q_config = config.copy()
        q_config['pred_q'] = True
        q_config['test'] = False
        self.q_net = Transformer(q_config, device)
        self.target_q_net = Transformer(copy.deepcopy(q_config), device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=config.get('lr', 5e-5), weight_decay=1e-3)
        self.to(device)

    def _soft_update(self):
        for tp, sp in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def train_step(self, batch):
        self.q_net.train()
        self.q_net.test = False

        x = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

        # Q-values for all timesteps: [B, H, A]
        q_values = self.q_net(x)
        B, H, A = q_values.shape

        # Target Q-values
        with torch.no_grad():
            target_q = self.target_q_net(x)  # [B, H, A]
            next_q_max = target_q.max(dim=-1, keepdim=True)[0]  # [B, H, 1]
            # Shift: use Q(t+1) for target at t
            next_q_shifted = torch.cat([next_q_max[:, 1:, :],
                                        torch.zeros(B, 1, 1, device=self.device)], dim=1)
            td_target = x['context_rewards'] + self.gamma * next_q_shifted  # [B, H, 1]

        # Q-value for taken actions
        actions_idx = x['context_actions'].argmax(dim=-1, keepdim=True)  # [B, H, 1]
        q_taken = q_values.gather(2, actions_idx)  # [B, H, 1]

        # Bellman loss
        bellman_loss = F.mse_loss(q_taken, td_target)

        # CQL penalty: logsumexp(Q) - Q_data
        cql_loss = (torch.logsumexp(q_values, dim=-1) - q_taken.squeeze(-1)).mean()

        loss = bellman_loss + self.alpha_cql * cql_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        self._soft_update()

        return {'loss': loss.item(), 'bellman': bellman_loss.item(), 'cql': cql_loss.item()}

    def forward(self, x):
        """For evaluate_darkroom compatibility: returns [B, A] or [B, T, A]."""
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


class IQLDarkroom(nn.Module):
    """Implicit Q-Learning for Darkroom."""
    def __init__(self, config, device, expectile=0.7, temperature=3.0, gamma=0.99, tau=0.005):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.expectile = expectile
        self.temperature = temperature
        self.act_num = config['act_num']
        n_embd = config['n_embd']

        # Q-network
        q_config = config.copy()
        q_config['pred_q'] = True
        q_config['test'] = False
        self.q_net = Transformer(q_config, device)
        self.target_q_net = Transformer(copy.deepcopy(q_config), device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # V-network: reuses transformer backbone + value head
        v_config = config.copy()
        v_config['pred_q'] = False
        v_config['test'] = False
        self.v_backbone = Transformer(v_config, device)
        self.v_head = nn.Linear(n_embd, 1).to(device)

        # Policy network
        pi_config = config.copy()
        pi_config['pred_q'] = False
        pi_config['test'] = False
        self.pi_net = Transformer(pi_config, device)

        self.q_optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=config.get('lr', 5e-5), weight_decay=1e-3)
        self.v_optimizer = torch.optim.AdamW(
            list(self.v_backbone.parameters()) + list(self.v_head.parameters()),
            lr=config.get('lr', 5e-5), weight_decay=1e-3)
        self.pi_optimizer = torch.optim.AdamW(self.pi_net.parameters(), lr=config.get('lr', 5e-5), weight_decay=1e-3)

        self.to(device)

    def _soft_update(self):
        for tp, sp in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def _get_v(self, x):
        """Get V-values: backbone features -> v_head."""
        self.v_backbone.test = self.q_net.test
        logits = self.v_backbone(x)  # [B, H, A] or [B, A]
        # Average over action dim to get a feature, then apply v_head
        # Actually, since backbone with pred_q=False outputs logits [B, H, A],
        # we use the mean across actions as a proxy state representation
        if logits.dim() == 3:
            feat = logits.mean(dim=-1, keepdim=True)  # [B, H, 1]
        else:
            feat = logits.mean(dim=-1, keepdim=True)  # [B, 1]
        return feat

    def _expectile_loss(self, pred, target):
        diff = target - pred
        weight = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        return (weight * diff.pow(2)).mean()

    def train_step(self, batch):
        x = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        self.q_net.train()
        self.q_net.test = False

        B, H, A = x['context_actions'].shape

        # 1) V-network update: expectile regression against Q-values
        with torch.no_grad():
            q_vals = self.target_q_net(x)  # [B, H, A]
            actions_idx = x['context_actions'].argmax(dim=-1, keepdim=True)
            q_data = q_vals.gather(2, actions_idx)  # [B, H, 1]

        v_pred = self._get_v(x)  # [B, H, 1]
        v_loss = self._expectile_loss(v_pred, q_data)

        self.v_optimizer.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.v_backbone.parameters()) + list(self.v_head.parameters()), 1.0)
        self.v_optimizer.step()

        # 2) Q-network update: Bellman with V-target bootstrap
        with torch.no_grad():
            v_next = self._get_v(x).detach()  # [B, H, 1]
            v_next_shifted = torch.cat([v_next[:, 1:, :],
                                        torch.zeros(B, 1, 1, device=self.device)], dim=1)
            td_target = x['context_rewards'] + self.gamma * v_next_shifted

        q_values = self.q_net(x)
        q_taken = q_values.gather(2, x['context_actions'].argmax(dim=-1, keepdim=True))
        q_loss = F.mse_loss(q_taken, td_target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.q_optimizer.step()
        self._soft_update()

        # 3) Policy update: advantage-weighted regression
        with torch.no_grad():
            adv = q_data - v_pred.detach()
            weights = torch.exp(adv / self.temperature).clamp(max=100.0)

        pi_logits = self.pi_net(x)  # [B, H, A]
        log_probs = F.log_softmax(pi_logits, dim=-1)
        actions_idx = x['context_actions'].argmax(dim=-1, keepdim=True)
        log_pi_a = log_probs.gather(2, actions_idx)  # [B, H, 1]
        pi_loss = -(weights * log_pi_a).mean()

        self.pi_optimizer.zero_grad()
        pi_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.pi_net.parameters(), 1.0)
        self.pi_optimizer.step()

        return {'loss': (v_loss + q_loss + pi_loss).item(), 'v_loss': v_loss.item(),
                'q_loss': q_loss.item(), 'pi_loss': pi_loss.item()}

    def forward(self, x):
        """For evaluate_darkroom: use pi_net for action selection."""
        return self.pi_net(x)

    @property
    def test(self):
        return self.pi_net.test

    @test.setter
    def test(self, val):
        self.pi_net.test = val
        self.q_net.test = val

    def eval(self):
        self.pi_net.eval()
        self.q_net.eval()
        self.v_backbone.eval()
        return self

    def train(self, mode=True):
        self.pi_net.train(mode)
        self.q_net.train(mode)
        self.v_backbone.train(mode)
        return self


class QSFTDarkroom(nn.Module):
    """Q-SFT: Q-learning via Supervised Fine-Tuning (weighted cross-entropy)."""
    def __init__(self, config, device, gamma=0.99, tau=0.005):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.act_num = config['act_num']

        q_config = config.copy()
        q_config['pred_q'] = True
        q_config['test'] = False
        self.q_net = Transformer(q_config, device)
        self.target_q_net = Transformer(copy.deepcopy(q_config), device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=config.get('lr', 5e-5), weight_decay=1e-3)
        self.to(device)

    def _soft_update(self):
        for tp, sp in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def train_step(self, batch):
        self.q_net.train()
        self.q_net.test = False

        x = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        B, H, A = x['context_actions'].shape

        q_values = self.q_net(x)  # [B, H, A]

        # Compute TD targets using target network
        with torch.no_grad():
            target_q = self.target_q_net(x)  # [B, H, A]
            next_q_max = target_q.max(dim=-1, keepdim=True)[0]
            next_q_shifted = torch.cat([next_q_max[:, 1:, :],
                                        torch.zeros(B, 1, 1, device=self.device)], dim=1)
            td_target = (x['context_rewards'] + self.gamma * next_q_shifted).squeeze(-1)  # [B, H]

        # Weighted cross-entropy loss (Q-SFT formulation)
        actions_idx = x['context_actions'].argmax(dim=-1)  # [B, H]
        log_probs = F.log_softmax(q_values, dim=-1)  # [B, H, A]

        # Weight = softmax(td_target) for taken action, uniform for others
        weights = torch.zeros_like(q_values)  # [B, H, A]
        # For the taken action, weight = td_target (normalized)
        td_normalized = torch.sigmoid(td_target)  # normalize to [0, 1]
        weights.scatter_(2, actions_idx.unsqueeze(-1), td_normalized.unsqueeze(-1))
        # For other actions, use current Q-values (lower weight)
        other_mask = 1 - F.one_hot(actions_idx, A).float()
        q_softmax = F.softmax(q_values.detach(), dim=-1)
        weights = weights + other_mask * q_softmax * 0.1

        loss = -(weights * log_probs).sum(dim=-1).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()
        self._soft_update()

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


class QDTDarkroom(nn.Module):
    """QDT: Q-learning Decision Transformer (two-phase)."""
    def __init__(self, config, device, gamma=0.99, tau=0.005):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.act_num = config['act_num']

        # Phase 1: Q-network
        q_config = config.copy()
        q_config['pred_q'] = True
        q_config['test'] = False
        self.q_net = Transformer(q_config, device)
        self.target_q_net = Transformer(copy.deepcopy(q_config), device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())

        # Phase 2: DT (policy) network - uses rewards channel for return-to-go
        dt_config = config.copy()
        dt_config['pred_q'] = False
        dt_config['test'] = False
        self.dt_net = Transformer(dt_config, device)

        self.q_optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=config.get('lr', 5e-5), weight_decay=1e-3)
        self.dt_optimizer = torch.optim.AdamW(self.dt_net.parameters(), lr=config.get('lr', 5e-5), weight_decay=1e-3)

        self.phase = 'q'  # 'q' or 'dt'
        self.to(device)

    def _soft_update(self):
        for tp, sp in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

    def train_step_q(self, batch):
        """Phase 1: Train Q-network."""
        self.q_net.train()
        self.q_net.test = False

        x = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        B, H, A = x['context_actions'].shape

        q_values = self.q_net(x)

        with torch.no_grad():
            target_q = self.target_q_net(x)
            next_q_max = target_q.max(dim=-1, keepdim=True)[0]
            next_q_shifted = torch.cat([next_q_max[:, 1:, :],
                                        torch.zeros(B, 1, 1, device=self.device)], dim=1)
            td_target = x['context_rewards'] + self.gamma * next_q_shifted

        actions_idx = x['context_actions'].argmax(dim=-1, keepdim=True)
        q_taken = q_values.gather(2, actions_idx)
        loss = F.mse_loss(q_taken, td_target)

        self.q_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.q_optimizer.step()
        self._soft_update()

        return {'q_loss': loss.item()}

    def train_step_dt(self, batch):
        """Phase 2: Train DT with Q-relabeled returns."""
        self.dt_net.train()
        self.dt_net.test = False
        self.q_net.eval()
        self.q_net.test = False

        x = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        B, H, A = x['context_actions'].shape

        # Relabel: use Q-values as return-to-go
        with torch.no_grad():
            q_vals = self.q_net(x)  # [B, H, A]
            actions_idx = x['context_actions'].argmax(dim=-1, keepdim=True)
            rtg = q_vals.gather(2, actions_idx)  # [B, H, 1]

        # Replace rewards with return-to-go for DT conditioning
        x_dt = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in x.items()}
        x_dt['context_rewards'] = rtg

        logits = self.dt_net(x_dt)  # [B, H, A]
        loss = F.cross_entropy(logits.reshape(-1, A), actions_idx.squeeze(-1).reshape(-1))

        self.dt_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dt_net.parameters(), 1.0)
        self.dt_optimizer.step()

        return {'dt_loss': loss.item()}

    def train_step(self, batch):
        if self.phase == 'q':
            return self.train_step_q(batch)
        else:
            return self.train_step_dt(batch)

    def forward(self, x):
        """For evaluation: use Q-net to compute RTG, then DT for action."""
        if self.phase == 'dt':
            # At test time, set desired RTG high
            x_eval = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in x.items()}
            # Use max Q-value as RTG signal
            with torch.no_grad():
                self.q_net.test = self.dt_net.test
                q_vals = self.q_net(x_eval)
                if q_vals.dim() == 2:
                    rtg = q_vals.max(dim=-1, keepdim=True)[0]
                else:
                    rtg = q_vals.max(dim=-1, keepdim=True)[0]
            # Reshape RTG to match context_rewards
            if 'context_rewards' in x_eval and x_eval['context_rewards'].dim() == 3:
                H = x_eval['context_rewards'].shape[1]
                if rtg.dim() == 2:
                    rtg = rtg.unsqueeze(1).expand(-1, H, -1)
                x_eval['context_rewards'] = rtg[:, :H, :]
            return self.dt_net(x_eval)
        else:
            return self.q_net(x)

    @property
    def test(self):
        if self.phase == 'dt':
            return self.dt_net.test
        return self.q_net.test

    @test.setter
    def test(self, val):
        self.q_net.test = val
        self.dt_net.test = val

    def eval(self):
        self.q_net.eval()
        self.dt_net.eval()
        return self

    def train(self, mode=True):
        if self.phase == 'q':
            self.q_net.train(mode)
        else:
            self.dt_net.train(mode)
        return self


# ============================================================
# Training Loop
# ============================================================

def train_and_evaluate(model, method_name, train_loader, eval_trajs, config, device,
                       num_epochs=100, save_dir='results', scheduler=None):
    """Unified training + evaluation loop."""
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'log.csv')

    best_reward = -float('inf')
    best_epoch = 0

    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,avg_reward\n')

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        epoch_loss = 0
        n_batches = 0

        for batch in tqdm.tqdm(train_loader, desc=f'{method_name} Epoch {epoch}'):
            metrics = model.train_step(batch)
            epoch_loss += metrics.get('loss', metrics.get('q_loss', metrics.get('dt_loss', 0)))
            n_batches += 1

        epoch_loss /= max(n_batches, 1)

        # Evaluation
        with torch.no_grad():
            rewards = evaluate_darkroom(model, config, device, eval_trajs, mode='greedy')
            avg_reward = np.mean(rewards)

        print(f'[{method_name}] Epoch {epoch} | Loss: {epoch_loss:.4f} | Avg Reward: {avg_reward:.2f}')

        with open(log_path, 'a') as f:
            f.write(f'{epoch},{epoch_loss:.6f},{avg_reward:.4f}\n')

        if avg_reward > best_reward:
            best_reward = avg_reward
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'ckpt_{epoch}.pth'))

        if scheduler is not None:
            scheduler.step()

    print(f'[{method_name}] Best reward: {best_reward:.2f} at epoch {best_epoch}')
    return best_reward


def train_qdt(model, train_loader, eval_trajs, config, device,
              q_epochs=50, dt_epochs=50, save_dir='results'):
    """Two-phase QDT training."""
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'log.csv')

    with open(log_path, 'w') as f:
        f.write('phase,epoch,train_loss,avg_reward\n')

    # Phase 1: Q-network
    model.phase = 'q'
    print("=== QDT Phase 1: Training Q-network ===")
    for epoch in range(1, q_epochs + 1):
        model.train()
        epoch_loss = 0
        n_batches = 0
        for batch in tqdm.tqdm(train_loader, desc=f'QDT-Q Epoch {epoch}'):
            metrics = model.train_step(batch)
            epoch_loss += metrics.get('q_loss', 0)
            n_batches += 1
        epoch_loss /= max(n_batches, 1)

        with torch.no_grad():
            rewards = evaluate_darkroom(model, config, device, eval_trajs, mode='greedy')
            avg_reward = np.mean(rewards)

        print(f'[QDT-Q] Epoch {epoch} | Loss: {epoch_loss:.4f} | Avg Reward: {avg_reward:.2f}')
        with open(log_path, 'a') as f:
            f.write(f'q,{epoch},{epoch_loss:.6f},{avg_reward:.4f}\n')

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'q_ckpt_{epoch}.pth'))

    # Phase 2: DT with relabeled returns
    model.phase = 'dt'
    print("=== QDT Phase 2: Training Decision Transformer ===")
    best_reward = -float('inf')
    for epoch in range(1, dt_epochs + 1):
        model.train()
        epoch_loss = 0
        n_batches = 0
        for batch in tqdm.tqdm(train_loader, desc=f'QDT-DT Epoch {epoch}'):
            metrics = model.train_step(batch)
            epoch_loss += metrics.get('dt_loss', 0)
            n_batches += 1
        epoch_loss /= max(n_batches, 1)

        with torch.no_grad():
            rewards = evaluate_darkroom(model, config, device, eval_trajs, mode='greedy')
            avg_reward = np.mean(rewards)

        print(f'[QDT-DT] Epoch {epoch} | Loss: {epoch_loss:.4f} | Avg Reward: {avg_reward:.2f}')
        with open(log_path, 'a') as f:
            f.write(f'dt,{epoch},{epoch_loss:.6f},{avg_reward:.4f}\n')

        if avg_reward > best_reward:
            best_reward = avg_reward
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

    return best_reward


# ============================================================
# QTPT and SPT (for reference comparison)
# ============================================================

class QTPTDarkroom(nn.Module):
    """QTPT (our method): Q-learning pretrained transformer."""
    def __init__(self, config, device, gamma=0.99, softmax_target=True, double_dqn=True):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.softmax_target = softmax_target
        self.double_dqn = double_dqn

        q_config = config.copy()
        q_config['pred_q'] = True
        q_config['test'] = False
        self.q_net = Transformer(q_config, device)

        if double_dqn:
            self.target_net = Transformer(copy.deepcopy(q_config), device)
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.AdamW(self.q_net.parameters(), lr=config.get('lr', 5e-5), weight_decay=1e-3)
        self.to(device)

    def train_step(self, batch):
        self.q_net.train()
        self.q_net.test = False

        x = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        B, H, A = x['context_actions'].shape

        q_values = self.q_net(x)  # [B, H, A]

        with torch.no_grad():
            if self.double_dqn:
                target_q = self.target_net(x)
            else:
                target_q = q_values.detach()

            if self.softmax_target:
                soft_q = F.softmax(target_q, dim=-1) * target_q
                next_q = soft_q.sum(dim=-1, keepdim=True)
            else:
                next_q = target_q.max(dim=-1, keepdim=True)[0]

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

        if self.double_dqn:
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


class SPTDarkroom(nn.Module):
    """SPT: Supervised Pretrained Transformer (behavior cloning)."""
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.act_num = config['act_num']

        spt_config = config.copy()
        spt_config['pred_q'] = False
        spt_config['test'] = False
        self.net = Transformer(spt_config, device)

        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=config.get('lr', 5e-5), weight_decay=1e-3)
        self.to(device)

    def train_step(self, batch):
        self.net.train()
        self.net.test = False

        x = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        B, H, A = x['context_actions'].shape

        logits = self.net(x)  # [B, H, A]
        targets = x['context_actions'].argmax(dim=-1)  # [B, H]
        loss = F.cross_entropy(logits.reshape(-1, A), targets.reshape(-1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.optimizer.step()

        return {'loss': loss.item()}

    def forward(self, x):
        return self.net(x)

    @property
    def test(self):
        return self.net.test

    @test.setter
    def test(self, val):
        self.net.test = val

    def eval(self):
        self.net.eval()
        return self

    def train(self, mode=True):
        self.net.train(mode)
        return self


# ============================================================
# Main
# ============================================================

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='all',
                        choices=['cql', 'iql', 'qsft', 'qdt', 'qtpt', 'spt', 'all'])
    parser.add_argument('--data_mix', type=float, default=0.0,
                        help='0.0=random, 1.0=expert')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--n_embd', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=6)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'qtpt_core', 'data', 'darkroom'))
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'qtpt_baselines_darkroom', 'results'))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--alpha_cql', type=float, default=1.0)
    parser.add_argument('--max_trajs', type=int, default=-1,
                        help='Max training trajectories (-1 = use all)')
    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load data
    mix = args.data_mix
    train_file = os.path.join(args.data_dir,
                              f'trajs_darkroom_envs100000_hists1_samples1_H100_d10_mix{mix}_train.pkl')
    eval_file = os.path.join(args.data_dir,
                             f'trajs_darkroom_envs100_H100_d10_mix{mix}_eval.pkl')

    # Fallback: eval file might not have mix suffix
    if not os.path.exists(eval_file):
        eval_file = os.path.join(args.data_dir, 'trajs_darkroom_envs100_H100_d10_mix0.0_eval.pkl')

    print(f'Loading train data: {train_file}')
    train_trajs = pickle.load(open(train_file, 'rb'))
    print(f'Loading eval data: {eval_file}')
    eval_trajs = pickle.load(open(eval_file, 'rb'))

    if args.max_trajs > 0:
        train_trajs = train_trajs[:args.max_trajs]

    print(f'Train trajectories: {len(train_trajs)}, Eval trajectories: {len(eval_trajs)}')

    # Dataset and DataLoader
    horizon = 100
    train_dataset = DarkRoomSequenceDataset(train_trajs, horizon, device)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)

    config = get_config(args)
    data_label = f'mix{mix}'

    # Determine which methods to run
    if args.method == 'all':
        methods = ['qtpt', 'spt', 'cql', 'iql', 'qsft', 'qdt']
    else:
        methods = [args.method]

    results = {}

    for method in methods:
        print(f'\n{"="*60}')
        print(f'Training: {method.upper()} | Data: {data_label} | Seed: {args.seed}')
        print(f'{"="*60}')

        save_dir = os.path.join(args.output_dir, f'{method}_{data_label}_seed{args.seed}')

        if method == 'cql':
            model = CQLDarkroom(config, device, alpha_cql=args.alpha_cql)
            best = train_and_evaluate(model, 'CQL', train_loader, eval_trajs, config, device,
                                      num_epochs=args.num_epochs, save_dir=save_dir)
        elif method == 'iql':
            model = IQLDarkroom(config, device)
            best = train_and_evaluate(model, 'IQL', train_loader, eval_trajs, config, device,
                                      num_epochs=args.num_epochs, save_dir=save_dir)
        elif method == 'qsft':
            model = QSFTDarkroom(config, device)
            best = train_and_evaluate(model, 'Q-SFT', train_loader, eval_trajs, config, device,
                                      num_epochs=args.num_epochs, save_dir=save_dir)
        elif method == 'qdt':
            model = QDTDarkroom(config, device)
            best = train_qdt(model, train_loader, eval_trajs, config, device,
                             q_epochs=args.num_epochs // 2, dt_epochs=args.num_epochs // 2,
                             save_dir=save_dir)
        elif method == 'qtpt':
            model = QTPTDarkroom(config, device)
            best = train_and_evaluate(model, 'QTPT', train_loader, eval_trajs, config, device,
                                      num_epochs=args.num_epochs, save_dir=save_dir)
        elif method == 'spt':
            model = SPTDarkroom(config, device)
            best = train_and_evaluate(model, 'SPT', train_loader, eval_trajs, config, device,
                                      num_epochs=args.num_epochs, save_dir=save_dir)

        results[method] = best
        print(f'[{method.upper()}] Best Avg Reward: {best:.2f}')

    # Summary
    print(f'\n{"="*60}')
    print(f'SUMMARY | Data: {data_label} | Seed: {args.seed}')
    print(f'{"="*60}')
    for method, reward in results.items():
        print(f'  {method.upper():>8s}: {reward:.2f}')

    # Save summary
    summary_path = os.path.join(args.output_dir, f'summary_{data_label}_seed{args.seed}.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Summary saved to {summary_path}')


if __name__ == '__main__':
    main()

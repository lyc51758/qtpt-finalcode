"""
Multi-Task AntMaze Experiment: QTPT vs Baselines

AntMaze is a goal-conditioned navigation task where an ant robot must reach
different goal positions in a maze. Each episode has a different goal, making
it a natural multi-task benchmark.

Key multi-task property:
  - Different goals = different tasks
  - Sparse reward (1 if near goal, 0 otherwise)
  - Agent must infer the current goal from context (reward signals)
  - QTPT's in-context learning should excel at goal inference

Methods: QTPT, SPT, CQL, IQL, Q-SFT, QDT (all continuous action)

Usage:
    python train_antmaze_multitask.py --method qtpt --env antmaze-umaze-v2
    python train_antmaze_multitask.py --method all --env antmaze-umaze-v2
"""

import os
import sys
import copy
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import tqdm

os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
import gym
import d4rl.gym_mujoco
import d4rl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'qtpt_core'))
from model import DecoderTransformerBackbone
from transformers import GPT2Config


# ============================================================
# State Normalizer
# ============================================================

class StateNormalizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = np.maximum(std, 1e-6)

    def normalize(self, states):
        if isinstance(states, torch.Tensor):
            mean = torch.tensor(self.mean, dtype=states.dtype, device=states.device)
            std = torch.tensor(self.std, dtype=states.dtype, device=states.device)
            return (states - mean) / std
        return (states - self.mean) / self.std


# ============================================================
# AntMaze Multi-Task Dataset
# ============================================================

class AntMazeMultiTaskDataset(Dataset):
    """
    AntMaze dataset with multi-task structure.

    Each episode has a specific goal position. We segment episodes into
    context windows of length context_len. The goal info is NOT given
    to the model; it must be inferred from the reward signal in context.
    """
    def __init__(self, env_name, context_len=20):
        env = gym.make(env_name)
        dataset = env.get_dataset()
        env.close()

        self.states = dataset['observations'].astype(np.float32)
        self.actions = dataset['actions'].astype(np.float32)
        self.rewards = dataset['rewards'].astype(np.float32)
        self.terminals = dataset['terminals'].astype(np.bool_)
        self.timeouts = dataset.get('timeouts', np.zeros_like(self.terminals)).astype(np.bool_)
        self.goals = dataset.get('infos/goal', None)

        # Build next_states by shifting
        self.next_states = np.zeros_like(self.states)
        self.next_states[:-1] = self.states[1:]
        # At episode boundaries, next_state = current state (will be masked by terminal)
        dones = np.logical_or(self.terminals, self.timeouts)
        done_idx = np.where(dones)[0]
        self.next_states[done_idx] = self.states[done_idx]

        self.state_dim = self.states.shape[1]
        self.action_dim = self.actions.shape[1]
        self.context_len = context_len

        # State normalization
        self.normalizer = StateNormalizer(
            mean=self.states.mean(axis=0),
            std=self.states.std(axis=0),
        )
        self.states = self.normalizer.normalize(self.states)
        self.next_states = self.normalizer.normalize(self.next_states)

        # Segment into episodes
        ends = np.where(dones)[0] + 1
        self.episodes = []
        self.episode_goals = []
        start = 0
        for end in ends:
            if end - start >= 2:
                self.episodes.append((start, end))
                if self.goals is not None:
                    self.episode_goals.append(self.goals[start].copy())
            start = end
        if start < len(self.states) and len(self.states) - start >= 2:
            self.episodes.append((start, len(self.states)))
            if self.goals is not None:
                self.episode_goals.append(self.goals[start].copy())

        # Build valid sampling indices
        self.valid_indices = []
        for ep_idx, (ep_start, ep_end) in enumerate(self.episodes):
            ep_len = ep_end - ep_start
            if ep_len >= context_len:
                for t in range(ep_len - context_len + 1):
                    self.valid_indices.append((ep_idx, t))
            else:
                self.valid_indices.append((ep_idx, 0))

        # Multi-task statistics
        if self.goals is not None:
            unique_goals = np.unique(np.round(np.array(self.episode_goals), 1), axis=0)
            n_unique = len(unique_goals)
        else:
            n_unique = 0

        print(f'AntMaze: {env_name} | {len(self.states)} transitions, '
              f'{len(self.episodes)} episodes, {len(self.valid_indices)} segments | '
              f'state_dim={self.state_dim}, action_dim={self.action_dim} | '
              f'~{n_unique} unique goals (tasks)')

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        ep_idx, t_start = self.valid_indices[idx]
        ep_start, ep_end = self.episodes[ep_idx]
        K = self.context_len

        abs_start = ep_start + t_start
        abs_end = min(abs_start + K, ep_end)
        actual_len = abs_end - abs_start

        states = self.states[abs_start:abs_end]
        actions = self.actions[abs_start:abs_end]
        rewards = self.rewards[abs_start:abs_end]
        next_states = self.next_states[abs_start:abs_end]
        terminals = self.terminals[abs_start:abs_end]

        # Left-pad short sequences
        if actual_len < K:
            pad = K - actual_len
            states = np.concatenate([np.zeros((pad, self.state_dim), dtype=np.float32), states])
            actions = np.concatenate([np.zeros((pad, self.action_dim), dtype=np.float32), actions])
            rewards = np.concatenate([np.zeros(pad, dtype=np.float32), rewards])
            next_states = np.concatenate([np.zeros((pad, self.state_dim), dtype=np.float32), next_states])
            terminals = np.concatenate([np.zeros(pad, dtype=np.bool_), terminals])

        return {
            'context_states': torch.tensor(states),
            'context_actions': torch.tensor(actions),
            'context_rewards': torch.tensor(rewards).unsqueeze(-1),
            'context_next_states': torch.tensor(next_states),
            'terminals': torch.tensor(terminals.astype(np.float32)).unsqueeze(-1),
        }


# ============================================================
# Input Sequence Builder (shared by all Transformer methods)
# ============================================================

def build_input_sequence(states, actions, rewards, device, query_state=None):
    """
    Build interleaved input: [s_0, (a_0,r_0), s_1, (a_1,r_1), ..., s_K]
    Even positions = state, odd positions = action+reward.
    """
    B, K, state_dim = states.shape
    action_dim = actions.shape[-1]

    state_seq = torch.zeros(B, 2 * K, state_dim, device=device)
    state_seq[:, 0::2, :] = states

    ar_seq = torch.zeros(B, 2 * K, action_dim + 1, device=device)
    ar_seq[:, 1::2, :] = torch.cat([actions, rewards], dim=-1)

    one_seq = torch.ones(B, 2 * K, 1, device=device)
    pos_seq = torch.arange(1, 2 * K + 1, dtype=torch.float32, device=device)
    pos_seq = pos_seq.view(1, -1, 1).expand(B, -1, -1)

    seq = torch.cat([state_seq, ar_seq, one_seq, pos_seq], dim=-1)

    if query_state is not None:
        query = torch.zeros(B, 1, seq.shape[-1], device=device)
        query[:, 0, :state_dim] = query_state
        query[:, 0, -2] = 1.0
        query[:, 0, -1] = 2 * K + 1
        seq = torch.cat([seq, query], dim=1)

    return seq


# ============================================================
# Actor and Critic (shared architecture)
# ============================================================

class ActorTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, n_embd, n_layer, n_head,
                 dropout, context_len, max_action=1.0):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_embd = n_embd
        self.context_len = context_len
        self.max_action = max_action

        cfg = GPT2Config(
            n_positions=4 * (1 + context_len),
            n_embd=n_embd, n_layer=n_layer, n_head=n_head,
            resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout,
            use_cache=False,
        )
        self.backbone = DecoderTransformerBackbone(cfg, activation="relu")
        self.embed = nn.Linear(3 + state_dim + action_dim, n_embd)
        self.head = nn.Sequential(
            nn.Linear(n_embd, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh(),
        )

    def forward(self, states, actions, rewards, query_state=None):
        device = states.device
        seq = build_input_sequence(states, actions, rewards, device, query_state)
        h = self.backbone(inputs_embeds=self.embed(seq))
        if query_state is not None:
            return self.max_action * self.head(h[:, -1, :])
        else:
            return self.max_action * self.head(h[:, 0::2, :])


class CriticTransformer(nn.Module):
    def __init__(self, state_dim, action_dim, n_embd, n_layer, n_head,
                 dropout, context_len):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_embd = n_embd
        self.context_len = context_len

        cfg = GPT2Config(
            n_positions=4 * (1 + context_len),
            n_embd=n_embd, n_layer=n_layer, n_head=n_head,
            resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout,
            use_cache=False,
        )
        self.backbone = DecoderTransformerBackbone(cfg, activation="relu")
        self.embed = nn.Linear(3 + state_dim + action_dim, n_embd)
        self.q1 = nn.Sequential(
            nn.Linear(n_embd + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(n_embd + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, states, actions, rewards, eval_actions):
        device = states.device
        seq = build_input_sequence(states, actions, rewards, device)
        h = self.backbone(inputs_embeds=self.embed(seq))
        h_states = h[:, 0::2, :]
        h_sa = torch.cat([h_states, eval_actions], dim=-1)
        return self.q1(h_sa), self.q2(h_sa)


# ============================================================
# Method 1: QTPT (our method)
# ============================================================

class QTPTContinuous(nn.Module):
    """QTPT: In-context Q-learning with Transformer, TD3-style actor-critic."""
    def __init__(self, state_dim, action_dim, config, device,
                 gamma=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5,
                 actor_update_freq=2, max_action=1.0, bc_alpha=2.5):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.actor_update_freq = actor_update_freq
        self.max_action = max_action
        self.bc_alpha = bc_alpha
        self.state_dim = state_dim
        self.action_dim = action_dim

        n_embd = config['n_embd']
        n_layer = config['n_layer']
        n_head = config['n_head']
        dropout = config['dropout']
        ctx_len = config['context_len']
        lr = config['lr']

        self.actor = ActorTransformer(
            state_dim, action_dim, n_embd, n_layer, n_head,
            dropout, ctx_len, max_action).to(device)
        self.critic = CriticTransformer(
            state_dim, action_dim, n_embd, n_layer, n_head,
            dropout, ctx_len).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=lr, weight_decay=1e-4)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=lr, weight_decay=1e-4)
        self.step_count = 0

    def train_step(self, batch):
        self.step_count += 1
        states = batch['context_states'].to(self.device)
        actions = batch['context_actions'].to(self.device)
        rewards = batch['context_rewards'].to(self.device)
        terminals = batch['terminals'].to(self.device)
        B, K, _ = states.shape

        # Critic loss
        with torch.no_grad():
            next_actions = self.actor_target(states, actions, rewards)
            noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)
            tq1, tq2 = self.critic_target(states, actions, rewards, next_actions)
            target_q = torch.min(tq1, tq2)
            target_q_next = torch.cat([
                target_q[:, 1:, :],
                torch.zeros(B, 1, 1, device=self.device)
            ], dim=1)
            td_target = rewards + self.gamma * (1 - terminals) * target_q_next

        q1, q2 = self.critic(states, actions, rewards, actions)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        metrics = {'critic_loss': critic_loss.item(), 'actor_loss': 0.0}

        # Actor loss (delayed)
        if self.step_count % self.actor_update_freq == 0:
            pi = self.actor(states, actions, rewards)
            q1_pi, _ = self.critic(states, actions, rewards, pi)
            lmbda = self.bc_alpha / q1_pi.abs().mean().detach()
            actor_loss = -lmbda * q1_pi.mean() + F.mse_loss(pi, actions)

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()
            metrics['actor_loss'] = actor_loss.item()

            for tp, sp in zip(self.actor_target.parameters(), self.actor.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
            for tp, sp in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

        return metrics


# ============================================================
# Method 2: SPT (Supervised Pretrained Transformer - BC)
# ============================================================

class SPTContinuous(nn.Module):
    """Behavior cloning with Transformer context."""
    def __init__(self, state_dim, action_dim, config, device, max_action=1.0):
        super().__init__()
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.actor = ActorTransformer(
            state_dim, action_dim, config['n_embd'], config['n_layer'],
            config['n_head'], config['dropout'], config['context_len'],
            max_action).to(device)
        self.optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=config['lr'], weight_decay=1e-4)

    def train_step(self, batch):
        states = batch['context_states'].to(self.device)
        actions = batch['context_actions'].to(self.device)
        rewards = batch['context_rewards'].to(self.device)

        pred_actions = self.actor(states, actions, rewards)
        loss = F.mse_loss(pred_actions, actions)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.optimizer.step()

        return {'critic_loss': loss.item(), 'actor_loss': loss.item()}


# ============================================================
# Method 3: CQL (Conservative Q-Learning, continuous)
# ============================================================

class CQLContinuous(nn.Module):
    """CQL with Transformer backbone for continuous actions."""
    def __init__(self, state_dim, action_dim, config, device,
                 gamma=0.99, tau=0.005, alpha_cql=0.1, max_action=1.0,
                 n_random_actions=10):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha_cql = alpha_cql
        self.max_action = max_action
        self.n_random_actions = n_random_actions
        self.state_dim = state_dim
        self.action_dim = action_dim

        n_embd = config['n_embd']
        n_layer = config['n_layer']
        n_head = config['n_head']
        dropout = config['dropout']
        ctx_len = config['context_len']
        lr = config['lr']

        self.actor = ActorTransformer(
            state_dim, action_dim, n_embd, n_layer, n_head,
            dropout, ctx_len, max_action).to(device)
        self.critic = CriticTransformer(
            state_dim, action_dim, n_embd, n_layer, n_head,
            dropout, ctx_len).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=lr, weight_decay=1e-4)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=lr, weight_decay=1e-4)
        self.step_count = 0

    def train_step(self, batch):
        self.step_count += 1
        states = batch['context_states'].to(self.device)
        actions = batch['context_actions'].to(self.device)
        rewards = batch['context_rewards'].to(self.device)
        terminals = batch['terminals'].to(self.device)
        B, K, _ = states.shape

        # Critic: Bellman + CQL penalty
        with torch.no_grad():
            next_actions = self.actor(states, actions, rewards)
            tq1, tq2 = self.critic_target(states, actions, rewards, next_actions)
            target_q = torch.min(tq1, tq2)
            target_q_next = torch.cat([
                target_q[:, 1:, :],
                torch.zeros(B, 1, 1, device=self.device)
            ], dim=1)
            td_target = rewards + self.gamma * (1 - terminals) * target_q_next

        q1_data, q2_data = self.critic(states, actions, rewards, actions)
        bellman_loss = F.mse_loss(q1_data, td_target) + F.mse_loss(q2_data, td_target)

        # CQL: penalize Q on random actions
        random_actions = torch.FloatTensor(
            B, K, self.n_random_actions, self.action_dim
        ).uniform_(-self.max_action, self.max_action).to(self.device)

        cql_loss = 0
        for i in range(self.n_random_actions):
            q1_rand, q2_rand = self.critic(
                states, actions, rewards, random_actions[:, :, i, :])
            cql_loss = cql_loss + (q1_rand.mean() + q2_rand.mean())
        cql_loss = cql_loss / self.n_random_actions
        cql_loss = cql_loss - (q1_data.mean() + q2_data.mean())

        critic_loss = bellman_loss + self.alpha_cql * cql_loss

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # Actor update
        metrics = {'critic_loss': critic_loss.item(), 'actor_loss': 0.0}
        if self.step_count % 2 == 0:
            pi = self.actor(states, actions, rewards)
            q1_pi, _ = self.critic(states, actions, rewards, pi)
            actor_loss = -q1_pi.mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()
            metrics['actor_loss'] = actor_loss.item()

            for tp, sp in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

        return metrics


# ============================================================
# Method 4: IQL (Implicit Q-Learning, continuous)
# ============================================================

class IQLContinuous(nn.Module):
    """IQL with Transformer backbone."""
    def __init__(self, state_dim, action_dim, config, device,
                 gamma=0.99, tau=0.005, expectile=0.7, temperature=3.0,
                 max_action=1.0):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.expectile = expectile
        self.temperature = temperature
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim

        n_embd = config['n_embd']
        n_layer = config['n_layer']
        n_head = config['n_head']
        dropout = config['dropout']
        ctx_len = config['context_len']
        lr = config['lr']

        self.actor = ActorTransformer(
            state_dim, action_dim, n_embd, n_layer, n_head,
            dropout, ctx_len, max_action).to(device)
        self.critic = CriticTransformer(
            state_dim, action_dim, n_embd, n_layer, n_head,
            dropout, ctx_len).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        # V-network: simple MLP on top of transformer features
        cfg = GPT2Config(
            n_positions=4 * (1 + ctx_len),
            n_embd=n_embd, n_layer=n_layer, n_head=n_head,
            resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout,
            use_cache=False,
        )
        self.v_backbone = DecoderTransformerBackbone(cfg, activation="relu").to(device)
        self.v_embed = nn.Linear(3 + state_dim + action_dim, n_embd).to(device)
        self.v_head = nn.Sequential(
            nn.Linear(n_embd, 256), nn.ReLU(),
            nn.Linear(256, 1),
        ).to(device)

        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=lr, weight_decay=1e-4)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=lr, weight_decay=1e-4)
        self.v_opt = torch.optim.AdamW(
            list(self.v_backbone.parameters()) + list(self.v_embed.parameters()) +
            list(self.v_head.parameters()), lr=lr, weight_decay=1e-4)

    def _get_v(self, states, actions, rewards):
        device = states.device
        seq = build_input_sequence(states, actions, rewards, device)
        h = self.v_backbone(inputs_embeds=self.v_embed(seq))
        h_states = h[:, 0::2, :]
        return self.v_head(h_states)

    def _expectile_loss(self, pred, target):
        diff = target - pred
        weight = torch.where(diff > 0, self.expectile, 1 - self.expectile)
        return (weight * diff.pow(2)).mean()

    def train_step(self, batch):
        states = batch['context_states'].to(self.device)
        actions = batch['context_actions'].to(self.device)
        rewards = batch['context_rewards'].to(self.device)
        terminals = batch['terminals'].to(self.device)
        B, K, _ = states.shape

        # 1) V-network: expectile regression against Q
        with torch.no_grad():
            q1, q2 = self.critic_target(states, actions, rewards, actions)
            q_data = torch.min(q1, q2)

        v_pred = self._get_v(states, actions, rewards)
        v_loss = self._expectile_loss(v_pred, q_data)

        self.v_opt.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.v_backbone.parameters()) + list(self.v_embed.parameters()) +
            list(self.v_head.parameters()), 1.0)
        self.v_opt.step()

        # 2) Q-network: Bellman with V bootstrap
        with torch.no_grad():
            v_next = self._get_v(states, actions, rewards).detach()
            v_next_shifted = torch.cat([
                v_next[:, 1:, :],
                torch.zeros(B, 1, 1, device=self.device)
            ], dim=1)
            td_target = rewards + self.gamma * (1 - terminals) * v_next_shifted

        q1, q2 = self.critic(states, actions, rewards, actions)
        q_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.critic_opt.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        for tp, sp in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

        # 3) Actor: advantage-weighted regression
        with torch.no_grad():
            adv = q_data - v_pred.detach()
            weights = torch.exp(adv / self.temperature).clamp(max=100.0)

        pi = self.actor(states, actions, rewards)
        actor_loss = (weights * (pi - actions).pow(2).sum(dim=-1, keepdim=True)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        return {'critic_loss': q_loss.item(), 'actor_loss': actor_loss.item()}


# ============================================================
# Method 5: Q-SFT (Q-value Supervised Fine-Tuning, continuous)
# ============================================================

class QSFTContinuous(nn.Module):
    """Q-SFT: Train actor with Q-weighted BC loss."""
    def __init__(self, state_dim, action_dim, config, device,
                 gamma=0.99, tau=0.005, max_action=1.0):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim

        n_embd = config['n_embd']
        n_layer = config['n_layer']
        n_head = config['n_head']
        dropout = config['dropout']
        ctx_len = config['context_len']
        lr = config['lr']

        self.actor = ActorTransformer(
            state_dim, action_dim, n_embd, n_layer, n_head,
            dropout, ctx_len, max_action).to(device)
        self.critic = CriticTransformer(
            state_dim, action_dim, n_embd, n_layer, n_head,
            dropout, ctx_len).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=lr, weight_decay=1e-4)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=lr, weight_decay=1e-4)

    def train_step(self, batch):
        states = batch['context_states'].to(self.device)
        actions = batch['context_actions'].to(self.device)
        rewards = batch['context_rewards'].to(self.device)
        terminals = batch['terminals'].to(self.device)
        B, K, _ = states.shape

        # Critic update (standard Bellman)
        with torch.no_grad():
            tq1, tq2 = self.critic_target(states, actions, rewards, actions)
            target_q = torch.min(tq1, tq2)
            target_q_next = torch.cat([
                target_q[:, 1:, :],
                torch.zeros(B, 1, 1, device=self.device)
            ], dim=1)
            td_target = rewards + self.gamma * (1 - terminals) * target_q_next

        q1, q2 = self.critic(states, actions, rewards, actions)
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        for tp, sp in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

        # Actor: Q-weighted BC
        with torch.no_grad():
            q1_data, q2_data = self.critic(states, actions, rewards, actions)
            q_vals = torch.min(q1_data, q2_data)
            weights = torch.sigmoid(q_vals)  # normalize to [0,1]

        pi = self.actor(states, actions, rewards)
        actor_loss = (weights * (pi - actions).pow(2).sum(dim=-1, keepdim=True)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        return {'critic_loss': critic_loss.item(), 'actor_loss': actor_loss.item()}


# ============================================================
# Method 6: QDT (Q-guided Decision Transformer, continuous)
# ============================================================

class QDTContinuous(nn.Module):
    """Two-phase: (1) train Q-network, (2) train DT conditioned on Q-values."""
    def __init__(self, state_dim, action_dim, config, device,
                 gamma=0.99, tau=0.005, max_action=1.0):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.state_dim = state_dim
        self.action_dim = action_dim

        n_embd = config['n_embd']
        n_layer = config['n_layer']
        n_head = config['n_head']
        dropout = config['dropout']
        ctx_len = config['context_len']
        lr = config['lr']

        self.critic = CriticTransformer(
            state_dim, action_dim, n_embd, n_layer, n_head,
            dropout, ctx_len).to(device)
        self.critic_target = copy.deepcopy(self.critic)

        # DT actor: same architecture but trained with BC conditioned on RTG
        self.actor = ActorTransformer(
            state_dim, action_dim, n_embd, n_layer, n_head,
            dropout, ctx_len, max_action).to(device)

        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=lr, weight_decay=1e-4)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=lr, weight_decay=1e-4)

        self.phase = 'q'  # 'q' or 'dt'

    def train_step(self, batch):
        if self.phase == 'q':
            return self._train_q(batch)
        else:
            return self._train_dt(batch)

    def _train_q(self, batch):
        states = batch['context_states'].to(self.device)
        actions = batch['context_actions'].to(self.device)
        rewards = batch['context_rewards'].to(self.device)
        terminals = batch['terminals'].to(self.device)
        B, K, _ = states.shape

        with torch.no_grad():
            tq1, tq2 = self.critic_target(states, actions, rewards, actions)
            target_q = torch.min(tq1, tq2)
            target_q_next = torch.cat([
                target_q[:, 1:, :],
                torch.zeros(B, 1, 1, device=self.device)
            ], dim=1)
            td_target = rewards + self.gamma * (1 - terminals) * target_q_next

        q1, q2 = self.critic(states, actions, rewards, actions)
        loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.critic_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        for tp, sp in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

        return {'critic_loss': loss.item(), 'actor_loss': 0.0}

    def _train_dt(self, batch):
        states = batch['context_states'].to(self.device)
        actions = batch['context_actions'].to(self.device)
        rewards = batch['context_rewards'].to(self.device)

        # Relabel rewards with Q-values as return-to-go
        with torch.no_grad():
            q1, q2 = self.critic(states, actions, rewards, actions)
            rtg = torch.min(q1, q2)

        pi = self.actor(states, actions, rtg)  # condition on Q-values
        loss = F.mse_loss(pi, actions)

        self.actor_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        return {'critic_loss': 0.0, 'actor_loss': loss.item()}


# ============================================================
# Evaluation
# ============================================================

def evaluate_antmaze(model, env_name, normalizer, context_len, device,
                     num_episodes=20, max_action=1.0):
    """
    Evaluate in AntMaze with in-context learning.
    Each episode has a random goal; the agent uses context to adapt.
    Returns success rate (primary metric for AntMaze).
    """
    env = gym.make(env_name)
    successes = []
    returns = []

    for ep in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        ctx_s, ctx_a, ctx_r = [], [], []
        episode_return = 0
        done = False

        for t in range(700):  # AntMaze max steps
            if done:
                break

            norm_state = normalizer.normalize(state)

            if len(ctx_s) == 0:
                s_t = torch.zeros(1, 1, model.state_dim, device=device)
                a_t = torch.zeros(1, 1, model.action_dim, device=device)
                r_t = torch.zeros(1, 1, 1, device=device)
            else:
                K = min(len(ctx_s), context_len)
                s_t = torch.tensor(np.array(ctx_s[-K:]), dtype=torch.float32,
                                   device=device).unsqueeze(0)
                a_t = torch.tensor(np.array(ctx_a[-K:]), dtype=torch.float32,
                                   device=device).unsqueeze(0)
                r_t = torch.tensor(np.array(ctx_r[-K:]), dtype=torch.float32,
                                   device=device).unsqueeze(0).unsqueeze(-1)

            query = torch.tensor(norm_state, dtype=torch.float32,
                                 device=device).unsqueeze(0)

            with torch.no_grad():
                action = model.actor(s_t, a_t, r_t, query_state=query)
                action = action.cpu().numpy().flatten()

            action = np.clip(action, -max_action, max_action)

            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                next_state, reward, done, info = result

            episode_return += reward

            ctx_s.append(norm_state.copy() if isinstance(norm_state, np.ndarray)
                         else np.array(norm_state))
            ctx_a.append(action.copy())
            ctx_r.append(float(reward))

            state = next_state

        # AntMaze success = reached goal at least once
        success = float(episode_return > 0)
        successes.append(success)
        returns.append(episode_return)

    env.close()

    return {
        'success_rate': float(np.mean(successes)),
        'mean_return': float(np.mean(returns)),
        'std_return': float(np.std(returns)),
    }


# ============================================================
# Training Loop
# ============================================================

def train_and_evaluate(model, method_name, dataloader, dataset, config,
                       device, args):
    save_dir = os.path.join(args.output_dir, f'{method_name}_{args.env}_seed{args.seed}')
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    log_path = os.path.join(save_dir, 'log.csv')
    with open(log_path, 'w') as f:
        f.write('epoch,critic_loss,actor_loss,success_rate,mean_return\n')

    best_success = -1.0

    for epoch in range(1, args.num_epochs + 1):
        model.train() if hasattr(model, 'train') else None
        epoch_critic = 0
        epoch_actor = 0
        n_batches = 0

        for batch in tqdm.tqdm(dataloader, desc=f'{method_name} Epoch {epoch}/{args.num_epochs}'):
            metrics = model.train_step(batch)
            epoch_critic += metrics.get('critic_loss', 0)
            epoch_actor += metrics.get('actor_loss', 0)
            n_batches += 1

        epoch_critic /= max(n_batches, 1)
        epoch_actor /= max(n_batches, 1)

        # Evaluate
        sr = mr = 0.0
        if epoch % args.eval_freq == 0 or epoch == 1:
            res = evaluate_antmaze(
                model, args.env, dataset.normalizer, args.context_len,
                device, num_episodes=args.eval_episodes, max_action=1.0)
            sr = res['success_rate']
            mr = res['mean_return']

            print(f'[{method_name}] Epoch {epoch} | Critic: {epoch_critic:.4f} | '
                  f'Actor: {epoch_actor:.4f} | Success: {sr:.2%} | Return: {mr:.2f}')

            if sr > best_success:
                best_success = sr
                torch.save({
                    'actor': model.actor.state_dict() if hasattr(model, 'actor') else
                             model.state_dict(),
                    'epoch': epoch,
                    'success_rate': sr,
                }, os.path.join(save_dir, 'best_model.pth'))
        else:
            print(f'[{method_name}] Epoch {epoch} | Critic: {epoch_critic:.4f} | '
                  f'Actor: {epoch_actor:.4f}')

        with open(log_path, 'a') as f:
            f.write(f'{epoch},{epoch_critic:.6f},{epoch_actor:.6f},{sr:.4f},{mr:.4f}\n')

    print(f'[{method_name}] Best success rate: {best_success:.2%}')
    return best_success


def train_qdt(model, dataloader, dataset, config, device, args):
    """Two-phase QDT training."""
    save_dir = os.path.join(args.output_dir, f'qdt_{args.env}_seed{args.seed}')
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    log_path = os.path.join(save_dir, 'log.csv')
    with open(log_path, 'w') as f:
        f.write('phase,epoch,critic_loss,actor_loss,success_rate,mean_return\n')

    q_epochs = args.num_epochs // 2
    dt_epochs = args.num_epochs - q_epochs

    # Phase 1: Q-network
    model.phase = 'q'
    print(f'=== QDT Phase 1: Q-network ({q_epochs} epochs) ===')
    for epoch in range(1, q_epochs + 1):
        epoch_loss = 0
        n = 0
        for batch in tqdm.tqdm(dataloader, desc=f'QDT-Q Epoch {epoch}'):
            metrics = model.train_step(batch)
            epoch_loss += metrics['critic_loss']
            n += 1
        epoch_loss /= max(n, 1)

        sr = mr = 0.0
        if epoch % args.eval_freq == 0:
            res = evaluate_antmaze(
                model, args.env, dataset.normalizer, args.context_len,
                device, num_episodes=args.eval_episodes)
            sr, mr = res['success_rate'], res['mean_return']
            print(f'[QDT-Q] Epoch {epoch} | Loss: {epoch_loss:.4f} | Success: {sr:.2%}')

        with open(log_path, 'a') as f:
            f.write(f'q,{epoch},{epoch_loss:.6f},0.0,{sr:.4f},{mr:.4f}\n')

    # Phase 2: DT
    model.phase = 'dt'
    best_success = -1.0
    print(f'=== QDT Phase 2: Decision Transformer ({dt_epochs} epochs) ===')
    for epoch in range(1, dt_epochs + 1):
        epoch_loss = 0
        n = 0
        for batch in tqdm.tqdm(dataloader, desc=f'QDT-DT Epoch {epoch}'):
            metrics = model.train_step(batch)
            epoch_loss += metrics['actor_loss']
            n += 1
        epoch_loss /= max(n, 1)

        sr = mr = 0.0
        if epoch % args.eval_freq == 0:
            res = evaluate_antmaze(
                model, args.env, dataset.normalizer, args.context_len,
                device, num_episodes=args.eval_episodes)
            sr, mr = res['success_rate'], res['mean_return']
            print(f'[QDT-DT] Epoch {epoch} | Loss: {epoch_loss:.4f} | Success: {sr:.2%}')

            if sr > best_success:
                best_success = sr
                torch.save({
                    'actor': model.actor.state_dict(),
                    'critic': model.critic.state_dict(),
                    'epoch': epoch,
                    'success_rate': sr,
                }, os.path.join(save_dir, 'best_model.pth'))

        with open(log_path, 'a') as f:
            f.write(f'dt,{epoch},0.0,{epoch_loss:.6f},{sr:.4f},{mr:.4f}\n')

    print(f'[QDT] Best success rate: {best_success:.2%}')
    return best_success


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='AntMaze Multi-Task Baselines')
    parser.add_argument('--method', type=str, default='all',
                        choices=['qtpt', 'spt', 'cql', 'iql', 'qsft', 'qdt', 'all'])
    parser.add_argument('--env', type=str, default='antmaze-umaze-v2',
                        choices=['antmaze-umaze-v2', 'antmaze-umaze-diverse-v2',
                                 'antmaze-medium-play-v2', 'antmaze-medium-diverse-v2'])
    parser.add_argument('--context_len', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--n_embd', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--eval_episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='results_antmaze')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Environment: {args.env}')

    # Dataset
    dataset = AntMazeMultiTaskDataset(args.env, context_len=args.context_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)

    config = {
        'n_embd': args.n_embd,
        'n_layer': args.n_layer,
        'n_head': args.n_head,
        'dropout': args.dropout,
        'context_len': args.context_len,
        'lr': args.lr,
    }

    if args.method == 'all':
        methods = ['qtpt', 'spt', 'cql', 'iql', 'qsft', 'qdt']
    else:
        methods = [args.method]

    results = {}

    for method in methods:
        print(f'\n{"="*60}')
        print(f'Training: {method.upper()} | Env: {args.env} | Seed: {args.seed}')
        print(f'{"="*60}')

        sd = dataset.state_dim
        ad = dataset.action_dim

        if method == 'qtpt':
            model = QTPTContinuous(sd, ad, config, device, gamma=args.gamma)
        elif method == 'spt':
            model = SPTContinuous(sd, ad, config, device)
        elif method == 'cql':
            model = CQLContinuous(sd, ad, config, device, gamma=args.gamma)
        elif method == 'iql':
            model = IQLContinuous(sd, ad, config, device, gamma=args.gamma)
        elif method == 'qsft':
            model = QSFTContinuous(sd, ad, config, device, gamma=args.gamma)
        elif method == 'qdt':
            model = QDTContinuous(sd, ad, config, device, gamma=args.gamma)

        if method == 'qdt':
            best = train_qdt(model, dataloader, dataset, config, device, args)
        else:
            best = train_and_evaluate(model, method, dataloader, dataset,
                                      config, device, args)

        results[method] = best
        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # Summary
    print(f'\n{"="*60}')
    print(f'SUMMARY | Env: {args.env} | Seed: {args.seed}')
    print(f'{"="*60}')
    for method, sr in results.items():
        print(f'  {method.upper():>8s}: {sr:.2%}')

    summary_path = os.path.join(args.output_dir, f'summary_{args.env}_seed{args.seed}.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Summary saved to {summary_path}')


if __name__ == '__main__':
    main()

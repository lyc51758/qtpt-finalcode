"""
Kitchen Multi-Task Experiment: QTPT

Franka Kitchen is a multi-task environment where a 9-DoF robot must complete
different combinations of subtasks (microwave, kettle, light, slider, etc.).
Different episodes involve different subtask orderings/combinations.

Key multi-task property:
  - Different subtask combinations = different tasks
  - Sparse reward: +1 per completed subtask (max 4 per episode)
  - Agent must infer current task from context (reward signals)
  - kitchen-mixed has lowest data quality -> tests "weak data to strong policy"

Env specs:
  - Obs: 60-dim (robot joints + object states), Act: 9-dim continuous [-1,1]
  - Episode length: 280 steps
  - Datasets: kitchen-partial-v0 (partial demos), kitchen-mixed-v0 (mixed quality)

Usage:
    python train_kitchen_qtpt.py --env kitchen-partial-v0
    python train_kitchen_qtpt.py --env kitchen-mixed-v0
    python train_kitchen_qtpt.py --env all
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
# Kitchen Multi-Task Dataset
# ============================================================

class KitchenMultiTaskDataset(Dataset):
    """
    Kitchen dataset with multi-task structure.

    Each episode involves completing a subset of kitchen subtasks.
    Different episodes have different subtask orderings/completions,
    making this a natural multi-task benchmark.
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

        # Build next_states
        self.next_states = np.zeros_like(self.states)
        self.next_states[:-1] = self.states[1:]
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
        start = 0
        for end in ends:
            if end - start >= 2:
                self.episodes.append((start, end))
            start = end
        if start < len(self.states) and len(self.states) - start >= 2:
            self.episodes.append((start, len(self.states)))

        # Build valid sampling indices
        self.valid_indices = []
        for ep_idx, (ep_start, ep_end) in enumerate(self.episodes):
            ep_len = ep_end - ep_start
            if ep_len >= context_len:
                for t in range(ep_len - context_len + 1):
                    self.valid_indices.append((ep_idx, t))
            else:
                self.valid_indices.append((ep_idx, 0))

        # Dataset statistics
        total_reward = self.rewards.sum()
        ep_rewards = []
        for ep_start, ep_end in self.episodes:
            ep_rewards.append(self.rewards[ep_start:ep_end].sum())
        ep_rewards = np.array(ep_rewards)

        print(f'Kitchen: {env_name} | {len(self.states)} transitions, '
              f'{len(self.episodes)} episodes, {len(self.valid_indices)} segments | '
              f'state_dim={self.state_dim}, action_dim={self.action_dim} | '
              f'avg_ep_reward={ep_rewards.mean():.2f}, max={ep_rewards.max():.1f}')

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
# Input Sequence Builder
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
# Actor and Critic
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
# QTPT (our method)
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
# Evaluation
# ============================================================

def evaluate_kitchen(model, env_name, normalizer, context_len, device,
                     num_episodes=20, max_action=1.0):
    """
    Evaluate in Kitchen with in-context learning.
    Returns average score (number of completed subtasks, 0-4).
    """
    env = gym.make(env_name)
    scores = []
    returns = []

    for ep in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        ctx_s, ctx_a, ctx_r = [], [], []
        episode_return = 0
        done = False

        for t in range(280):  # Kitchen max steps
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

        # Kitchen score = total completed subtasks (episode return)
        scores.append(episode_return)
        returns.append(episode_return)

    env.close()

    # D4RL normalized score (kitchen max=4, min=0)
    mean_score = float(np.mean(scores))
    # Normalized: score / 4 * 100 (percentage of max)
    normalized_score = mean_score / 4.0 * 100.0

    return {
        'mean_score': mean_score,
        'std_score': float(np.std(scores)),
        'normalized_score': normalized_score,
        'mean_return': float(np.mean(returns)),
    }


# ============================================================
# Training Loop
# ============================================================

def train_and_evaluate(model, env_name, dataloader, dataset, config,
                       device, args):
    save_dir = os.path.join(args.output_dir, f'qtpt_{env_name}_seed{args.seed}')
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    log_path = os.path.join(save_dir, 'log.csv')
    with open(log_path, 'w') as f:
        f.write('epoch,critic_loss,actor_loss,mean_score,normalized_score\n')

    best_score = -1.0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_critic = 0
        epoch_actor = 0
        n_batches = 0

        for batch in tqdm.tqdm(dataloader, desc=f'QTPT Epoch {epoch}/{args.num_epochs}'):
            metrics = model.train_step(batch)
            epoch_critic += metrics.get('critic_loss', 0)
            epoch_actor += metrics.get('actor_loss', 0)
            n_batches += 1

        epoch_critic /= max(n_batches, 1)
        epoch_actor /= max(n_batches, 1)

        # Evaluate
        ms = ns = 0.0
        if epoch % args.eval_freq == 0 or epoch == 1:
            res = evaluate_kitchen(
                model, env_name, dataset.normalizer, args.context_len,
                device, num_episodes=args.eval_episodes, max_action=1.0)
            ms = res['mean_score']
            ns = res['normalized_score']

            print(f'[QTPT] Epoch {epoch} | Critic: {epoch_critic:.4f} | '
                  f'Actor: {epoch_actor:.4f} | Score: {ms:.2f}/4 ({ns:.1f}%)')

            if ms > best_score:
                best_score = ms
                torch.save({
                    'actor': model.actor.state_dict(),
                    'critic': model.critic.state_dict(),
                    'epoch': epoch,
                    'mean_score': ms,
                    'normalized_score': ns,
                }, os.path.join(save_dir, 'best_model.pth'))
        else:
            print(f'[QTPT] Epoch {epoch} | Critic: {epoch_critic:.4f} | '
                  f'Actor: {epoch_actor:.4f}')

        with open(log_path, 'a') as f:
            f.write(f'{epoch},{epoch_critic:.6f},{epoch_actor:.6f},{ms:.4f},{ns:.4f}\n')

    print(f'[QTPT] Best score on {env_name}: {best_score:.2f}/4 ({best_score/4*100:.1f}%)')
    return best_score


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Kitchen QTPT Experiment')
    parser.add_argument('--env', type=str, default='all',
                        choices=['kitchen-partial-v0', 'kitchen-mixed-v0',
                                 'kitchen-complete-v0', 'all'])
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
    parser.add_argument('--output_dir', type=str, default='results_kitchen')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if args.env == 'all':
        envs = ['kitchen-partial-v0', 'kitchen-mixed-v0']
    else:
        envs = [args.env]

    config = {
        'n_embd': args.n_embd,
        'n_layer': args.n_layer,
        'n_head': args.n_head,
        'dropout': args.dropout,
        'context_len': args.context_len,
        'lr': args.lr,
    }

    results = {}

    for env_name in envs:
        print(f'\n{"="*60}')
        print(f'Training QTPT | Env: {env_name} | Seed: {args.seed}')
        print(f'{"="*60}')

        dataset = KitchenMultiTaskDataset(env_name, context_len=args.context_len)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=4, pin_memory=True)

        sd = dataset.state_dim
        ad = dataset.action_dim

        model = QTPTContinuous(sd, ad, config, device, gamma=args.gamma)
        best = train_and_evaluate(model, env_name, dataloader, dataset,
                                  config, device, args)
        results[env_name] = best

        del model
        del dataset
        del dataloader
        torch.cuda.empty_cache()

    # Summary
    print(f'\n{"="*60}')
    print(f'SUMMARY | QTPT Kitchen Results')
    print(f'{"="*60}')
    for env_name, score in results.items():
        print(f'  {env_name}: {score:.2f}/4 ({score/4*100:.1f}%)')

    summary_path = os.path.join(args.output_dir, f'summary_qtpt_seed{args.seed}.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Summary saved to {summary_path}')


if __name__ == '__main__':
    main()

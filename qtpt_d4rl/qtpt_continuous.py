"""
QTPT for Continuous Action Spaces - D4RL Benchmarks

Extends QTPT to continuous actions via Actor-Critic architecture:
  Critic: Transformer(context) + action -> Q(s, a; context)
  Actor:  Transformer(context) -> continuous action (tanh-squashed)

The actor approximates argmax_a Q(s,a), replacing enumeration in the discrete case.
Everything else (TD target, in-context learning, Transformer backbone) stays the same.

Training follows TD3-style:
  - Twin Q-heads (shared backbone) to reduce overestimation
  - Delayed actor update (every 2 critic steps)
  - Target action smoothing (clipped noise on target actions)
  - BC regularization for offline stability

Usage:
    python qtpt_continuous.py \
        --env halfcheetah-medium-v2 \
        --context_len 20 --num_epochs 200 --batch_size 256
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


#  State Normalizer 

class StateNormalizer:
    """Normalize states using dataset statistics."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = np.maximum(std, 1e-6)

    def normalize(self, states):
        if isinstance(states, torch.Tensor):
            mean = torch.tensor(self.mean, dtype=states.dtype, device=states.device)
            std = torch.tensor(self.std, dtype=states.dtype, device=states.device)
            return (states - mean) / std
        return (states - self.mean) / self.std


#  D4RL Sequence Dataset 

class D4RLSequenceDataset(Dataset):
    """
    D4RL dataset segmented into trajectory chunks of length context_len.
    Each sample is a consecutive segment from a single episode.
    """
    def __init__(self, env_name, context_len=20):
        env = gym.make(env_name)
        dataset = d4rl.qlearning_dataset(env)
        env.close()

        self.states = dataset['observations'].astype(np.float32)
        self.actions = dataset['actions'].astype(np.float32)
        self.rewards = dataset['rewards'].astype(np.float32)
        self.next_states = dataset['next_observations'].astype(np.float32)
        self.terminals = dataset['terminals'].astype(np.bool_)
        self.timeouts = dataset.get('timeouts', np.zeros_like(self.terminals)).astype(np.bool_)

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
        ends = np.where(self.terminals | self.timeouts)[0] + 1
        self.episodes = []
        start = 0
        for end in ends:
            if end - start >= 2:
                self.episodes.append((start, end))
            start = end
        if start < len(self.states):
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

        print(f'D4RL: {env_name} | {len(self.states)} transitions, '
              f'{len(self.episodes)} episodes, {len(self.valid_indices)} segments | '
              f'state_dim={self.state_dim}, action_dim={self.action_dim}')

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


#  Input Sequence Builder 

def build_input_sequence(states, actions, rewards, device, query_state=None):
    """
    Build interleaved input sequence for the Transformer backbone.

    Per timestep t:
      Even position 2t:   [state_t,           zeros(action_dim+1), 1, pos]
      Odd  position 2t+1: [zeros(state_dim),   action_t, reward_t, 1, pos]

    Causal mask ensures position 2t sees: s_0,a_0,r_0, ..., s_t (not a_t, r_t).
    This is correct for both actor (predict a_t from s_t + history) and
    critic (evaluate Q(s_t, a) given context).

    Args:
        states:  [B, K, state_dim]
        actions: [B, K, action_dim]
        rewards: [B, K, 1]
        query_state: [B, state_dim], optional extra state for test-time inference
    Returns:
        seq: [B, 2K (+1 if query), input_dim]
    """
    B, K, state_dim = states.shape
    action_dim = actions.shape[-1]

    # Even positions: state
    state_seq = torch.zeros(B, 2 * K, state_dim, device=device)
    state_seq[:, 0::2, :] = states

    # Odd positions: action + reward
    ar_seq = torch.zeros(B, 2 * K, action_dim + 1, device=device)
    ar_seq[:, 1::2, :] = torch.cat([actions, rewards], dim=-1)

    one_seq = torch.ones(B, 2 * K, 1, device=device)
    pos_seq = torch.arange(1, 2 * K + 1, dtype=torch.float32, device=device)
    pos_seq = pos_seq.view(1, -1, 1).expand(B, -1, -1)

    seq = torch.cat([state_seq, ar_seq, one_seq, pos_seq], dim=-1)
    # seq: [B, 2K, state_dim + action_dim + 3]

    if query_state is not None:
        # Append one extra even position for the query state
        query = torch.zeros(B, 1, seq.shape[-1], device=device)
        query[:, 0, :state_dim] = query_state
        query[:, 0, -2] = 1.0
        query[:, 0, -1] = 2 * K + 1
        seq = torch.cat([seq, query], dim=1)

    return seq


#  Actor (Transformer-based) 

class ActorTransformer(nn.Module):
    """Transformer actor: context -> continuous action (tanh-squashed)."""

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
        """
        Training: returns [B, K, action_dim] predicted actions at each context step.
        Test:     returns [B, action_dim] action for the query state.
        """
        device = states.device
        seq = build_input_sequence(states, actions, rewards, device, query_state)
        h = self.backbone(inputs_embeds=self.embed(seq))

        if query_state is not None:
            return self.max_action * self.head(h[:, -1, :])
        else:
            return self.max_action * self.head(h[:, 0::2, :])


#  Critic (Transformer-based, twin Q-heads) 

class CriticTransformer(nn.Module):
    """Transformer critic: context + action -> twin Q-values."""

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
        """
        Args:
            states, actions, rewards: context from dataset [B, K, ...]
            eval_actions: actions to evaluate Q for [B, K, action_dim]
        Returns:
            q1, q2: [B, K, 1]
        """
        device = states.device
        seq = build_input_sequence(states, actions, rewards, device)
        h = self.backbone(inputs_embeds=self.embed(seq))

        h_states = h[:, 0::2, :]  # [B, K, n_embd], even positions
        h_sa = torch.cat([h_states, eval_actions], dim=-1)

        return self.q1(h_sa), self.q2(h_sa)


#  QTPT Continuous (TD3-style training) 

class QTPTContinuous(nn.Module):
    """
    QTPT for continuous action spaces.

    Core idea unchanged: Transformer does in-context Q-learning.
    The actor replaces discrete argmax with a learned continuous policy.

    Training:
      Critic loss: MSE( Q(s_t, a_t; ctx), r_t + (1-d) * Q_target(s_{t+1}, _target(s_{t+1})) )
      Actor loss:  -Q(s_t, (s_t; ctx); ctx) +  * BC_loss
    """
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
            dropout, ctx_len, max_action,
        ).to(device)

        self.critic = CriticTransformer(
            state_dim, action_dim, n_embd, n_layer, n_head,
            dropout, ctx_len,
        ).to(device)

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=lr, weight_decay=1e-4)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=lr, weight_decay=1e-4)

        self.step_count = 0

    def select_action(self, states, actions, rewards, query_state):
        """Inference: get action for a single query state given context."""
        with torch.no_grad():
            return self.actor(states, actions, rewards, query_state)

    def train_step(self, batch):
        self.step_count += 1
        states = batch['context_states'].to(self.device)
        actions = batch['context_actions'].to(self.device)
        rewards = batch['context_rewards'].to(self.device)
        terminals = batch['terminals'].to(self.device)
        B, K, _ = states.shape

        #  Critic loss 
        with torch.no_grad():
            # Target actor: predict next actions at each position
            next_actions = self.actor_target(states, actions, rewards)
            # TD3 target smoothing
            noise = (torch.randn_like(next_actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip)
            next_actions = (next_actions + noise).clamp(-self.max_action, self.max_action)

            # Target Q at each position
            tq1, tq2 = self.critic_target(states, actions, rewards, next_actions)
            target_q = torch.min(tq1, tq2)

            # Shift by 1: Q at position t+1 is the bootstrap for step t
            target_q_next = torch.cat([
                target_q[:, 1:, :],
                torch.zeros(B, 1, 1, device=self.device)
            ], dim=1)

            td_target = rewards + self.gamma * (1 - terminals) * target_q_next

        q1, q2 = self.critic(states, actions, rewards, actions)  # Q for dataset actions
        critic_loss = F.mse_loss(q1, td_target) + F.mse_loss(q2, td_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        metrics = {'critic_loss': critic_loss.item()}

        #  Actor loss (delayed) 
        if self.step_count % self.actor_update_freq == 0:
            pi = self.actor(states, actions, rewards)
            q1_pi, _ = self.critic(states, actions, rewards, pi)

            # TD3+BC: normalize Q scale, add BC regularization
            lmbda = self.bc_alpha / q1_pi.abs().mean().detach()
            actor_loss = -lmbda * q1_pi.mean() + F.mse_loss(pi, actions)

            self.actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_opt.step()

            metrics['actor_loss'] = actor_loss.item()

            # EMA target update
            for tp, sp in zip(self.actor_target.parameters(), self.actor.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
            for tp, sp in zip(self.critic_target.parameters(), self.critic.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

        return metrics


#  Evaluation 

def evaluate_d4rl(model, env_name, normalizer, context_len, device,
                  num_episodes=10, max_action=1.0):
    """Evaluate actor in D4RL environment with rolling context buffer."""
    env = gym.make(env_name)
    returns = []

    for ep in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        ctx_s, ctx_a, ctx_r = [], [], []
        episode_return = 0
        done = False

        for t in range(env._max_episode_steps or 1000):
            if done:
                break

            norm_state = normalizer.normalize(state)

            # Build context tensors
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
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            episode_return += reward

            ctx_s.append(norm_state.copy() if isinstance(norm_state, np.ndarray)
                         else np.array(norm_state))
            ctx_a.append(action.copy())
            ctx_r.append(float(reward))

            state = next_state

        returns.append(episode_return)

    env.close()

    # Normalized D4RL score
    try:
        norm_scores = [d4rl.get_normalized_score(env_name, r) * 100 for r in returns]
    except Exception:
        norm_scores = [0.0] * len(returns)

    return {
        'mean_return': float(np.mean(returns)),
        'std_return': float(np.std(returns)),
        'mean_normalized': float(np.mean(norm_scores)),
        'std_normalized': float(np.std(norm_scores)),
        'all_returns': returns,
    }


#  Main 

def main():
    parser = argparse.ArgumentParser(description='QTPT for D4RL continuous control')
    parser.add_argument('--env', type=str, default='halfcheetah-medium-v2')
    parser.add_argument('--context_len', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--n_embd', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=4)
    parser.add_argument('--n_head', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--policy_noise', type=float, default=0.2)
    parser.add_argument('--noise_clip', type=float, default=0.5)
    parser.add_argument('--bc_alpha', type=float, default=2.5,
                        help='BC regularization weight (like TD3+BC)')
    parser.add_argument('--actor_update_freq', type=int, default=2)
    parser.add_argument('--eval_episodes', type=int, default=10)
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output_dir', type=str, default='results_d4rl')
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
    dataset = D4RLSequenceDataset(args.env, context_len=args.context_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)

    # Environment info
    env = gym.make(args.env)
    max_action = float(env.action_space.high[0])
    env.close()

    config = {
        'n_embd': args.n_embd,
        'n_layer': args.n_layer,
        'n_head': args.n_head,
        'dropout': args.dropout,
        'context_len': args.context_len,
        'lr': args.lr,
    }

    model = QTPTContinuous(
        state_dim=dataset.state_dim,
        action_dim=dataset.action_dim,
        config=config,
        device=device,
        gamma=args.gamma,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        actor_update_freq=args.actor_update_freq,
        max_action=max_action,
        bc_alpha=args.bc_alpha,
    )

    save_dir = os.path.join(args.output_dir, f'{args.env}_seed{args.seed}')
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    log_path = os.path.join(save_dir, 'log.csv')
    with open(log_path, 'w') as f:
        f.write('epoch,critic_loss,actor_loss,eval_return,eval_std,'
                'normalized_score,normalized_std\n')

    best_normalized = -float('inf')

    for epoch in range(1, args.num_epochs + 1):
        epoch_critic = 0
        epoch_actor = 0
        n_batches = 0
        n_actor = 0

        for batch in tqdm.tqdm(dataloader, desc=f'Epoch {epoch}/{args.num_epochs}'):
            metrics = model.train_step(batch)
            epoch_critic += metrics['critic_loss']
            if 'actor_loss' in metrics:
                epoch_actor += metrics['actor_loss']
                n_actor += 1
            n_batches += 1

        epoch_critic /= max(n_batches, 1)
        epoch_actor /= max(n_actor, 1)

        # Evaluate
        eval_ret = eval_std = norm_score = norm_std = 0.0
        if epoch % args.eval_freq == 0 or epoch == 1:
            res = evaluate_d4rl(
                model, args.env, dataset.normalizer, args.context_len,
                device, args.eval_episodes, max_action)
            eval_ret = res['mean_return']
            eval_std = res['std_return']
            norm_score = res['mean_normalized']
            norm_std = res['std_normalized']

            print(f'[Epoch {epoch}] Critic: {epoch_critic:.4f} | '
                  f'Actor: {epoch_actor:.4f} | '
                  f'Return: {eval_ret:.1f}{eval_std:.1f} | '
                  f'Normalized: {norm_score:.1f}{norm_std:.1f}')

            if norm_score > best_normalized:
                best_normalized = norm_score
                torch.save({
                    'actor': model.actor.state_dict(),
                    'critic': model.critic.state_dict(),
                    'epoch': epoch,
                    'normalized_score': norm_score,
                }, os.path.join(save_dir, 'best_model.pth'))
        else:
            print(f'[Epoch {epoch}] Critic: {epoch_critic:.4f} | '
                  f'Actor: {epoch_actor:.4f}')

        with open(log_path, 'a') as f:
            f.write(f'{epoch},{epoch_critic:.6f},{epoch_actor:.6f},'
                    f'{eval_ret:.4f},{eval_std:.4f},'
                    f'{norm_score:.4f},{norm_std:.4f}\n')

    print(f'\nBest normalized score: {best_normalized:.1f}')
    print(f'Results saved to {save_dir}')


if __name__ == '__main__':
    main()
